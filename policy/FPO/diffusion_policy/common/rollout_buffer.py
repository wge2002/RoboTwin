# diffusion_policy/common/rollout_buffer.py

import torch
import numpy as np
from typing import Dict, Optional, Union, Tuple
from diffusion_policy.common.pytorch_util import dict_apply

class RolloutBuffer:
    def __init__(
        self, 
        num_envs: int,
        buffer_size: int,
        obs_shape: Dict[str, Tuple], # e.g. {'image': (3, 96, 96), 'agent_pos': (2,)}
        action_shape: Tuple,
        device: str = "cpu",
        # FPO specific dims
        feature_dim: int = None, 
    ):
        self.num_envs = num_envs
        self.buffer_size = buffer_size
        self.device = device
        self.ptr = 0
        self.full = False

        # --- 初始化存储空间 ---
        
        # 1. Observation (Nested Dict)
        self.obs = {}
        for key, shape in obs_shape.items():
            self.obs[key] = torch.zeros((buffer_size, num_envs) + shape).to(device)
            
        # 2. Action & Rewards
        self.actions = torch.zeros((buffer_size, num_envs) + action_shape).to(device)
        self.rewards = torch.zeros((buffer_size, num_envs)).to(device)
        self.dones = torch.zeros((buffer_size, num_envs)).to(device)
        
        # 3. Value Function (Critic 输出)
        self.values = torch.zeros((buffer_size, num_envs)).to(device)
        
        # 4. FPO/PPO Specifics
        # FPO 需要存储生成动作时的 Loss 快照，而不是 log_prob
        # info['loss'] shape: (B, N_probes)
        # 我们假设 N_probes 是动态的，或者这里先不预分配，用 List 存？
        # 为了效率，建议预分配。假设 policy.sample_action_with_info 返回的 info 结构固定
        # 这里先用 List 暂存 info，最后 stack，或者简单起见只存核心 tensor
        self.fpo_loss_snapshots = [] # 暂时用 list，因为 num_probes 可能变
        self.fpo_noise = []
        self.fpo_timesteps = []

        # 5. Advantages & Returns (Computed later)
        self.advantages = torch.zeros((buffer_size, num_envs)).to(device)
        self.returns = torch.zeros((buffer_size, num_envs)).to(device)

    def add(
        self, 
        obs: Dict[str, torch.Tensor], 
        action: torch.Tensor, 
        reward: torch.Tensor, 
        done: torch.Tensor, 
        value: torch.Tensor,
        info: Dict[str, torch.Tensor] # FPO info
    ):
        if self.ptr >= self.buffer_size:
            raise IndexError("RolloutBuffer is full")

        # 存 Obs
        for key in self.obs.keys():
            self.obs[key][self.ptr] = obs[key].detach().to(self.device)
            
        # 存基础数据
        self.actions[self.ptr] = action.detach().to(self.device)
        self.rewards[self.ptr] = reward.detach().to(self.device)
        self.dones[self.ptr] = done.detach().to(self.device)
        self.values[self.ptr] = value.detach().to(self.device).squeeze(-1) # (B, 1) -> (B,)

        # 存 FPO Info (Loss snapshot, etc)
        # info['loss']: (B, N)
        self.fpo_loss_snapshots.append(info['loss'].detach().to(self.device))
        self.fpo_noise.append(info['noise'].detach().to(self.device))
        self.fpo_timesteps.append(info['timesteps'].detach().to(self.device))

        self.ptr += 1

    def compute_gae(self, next_value, gamma=0.99, gae_lambda=0.95):
        """
        计算 Generalized Advantage Estimation (GAE)
        next_value: 最后一个状态的 Value，用于 bootstrap
        """
        self.values = self.values.to(self.device)
        next_value = next_value.to(self.device).squeeze(-1)
        self.advantages = torch.zeros_like(self.rewards).to(self.device)
        
        last_gae_lam = 0
        for step in reversed(range(self.buffer_size)):
            if step == self.buffer_size - 1:
                next_non_terminal = 1.0 - 0.0 # 假设 rollout 结束时不一定是 done，或者是截断
                # 如果是 done，next_value 应该是 0 或者 masked。
                # 这里简化处理：如果在 buffer 边界，我们用传入的 next_value
                next_val = next_value
            else:
                next_non_terminal = 1.0 - self.dones[step + 1]
                next_val = self.values[step + 1]

            # Delta = r + gamma * V(s') * (1-done) - V(s)
            delta = self.rewards[step] + gamma * next_val * next_non_terminal - self.values[step]
            
            # GAE = delta + gamma * lambda * (1-done) * last_gae
            last_gae_lam = delta + gamma * gae_lambda * next_non_terminal * last_gae_lam
            self.advantages[step] = last_gae_lam

        # Returns = Advantage + Value
        self.returns = self.advantages + self.values

    def get_generator(self, mini_batch_size=None):
        """
        生成器，用于 PPO Update 循环
        打乱数据并分批返回
        """
        # 1. 整理 FPO Info
        # List -> Tensor: (T, B, ...) -> (T*B, ...)
        fpo_loss = torch.stack(self.fpo_loss_snapshots) # (T, B, N)
        fpo_noise = torch.stack(self.fpo_noise)
        fpo_time = torch.stack(self.fpo_timesteps)
        
        # 2. Flatten 所有数据: (Buffer_Size, Num_Envs, ...) -> (Batch_Size, ...)
        batch_size = self.buffer_size * self.num_envs
        
        # Helper to flatten: (T, B, ...) -> (T*B, ...)
        def flatten_env(x):
            return x.reshape(batch_size, *x.shape[2:])

        flat_obs = dict_apply(self.obs, flatten_env)
        flat_actions = flatten_env(self.actions)
        flat_log_probs = flatten_env(fpo_loss) # 这里借用 log_probs 名字存 loss snapshot
        flat_noise = flatten_env(fpo_noise)
        flat_time = flatten_env(fpo_time)
        
        flat_advantages = self.advantages.reshape(batch_size)
        flat_returns = self.returns.reshape(batch_size)
        flat_values = self.values.reshape(batch_size)

        # 3. Mini-batch 迭代
        indices = np.arange(batch_size)
        np.random.shuffle(indices)

        if mini_batch_size is None:
            mini_batch_size = batch_size

        for start in range(0, batch_size, mini_batch_size):
            end = start + mini_batch_size
            mb_inds = indices[start:end]

            yield {
                'obs': dict_apply(flat_obs, lambda x: x[mb_inds]),
                'action': flat_actions[mb_inds],
                'value': flat_values[mb_inds],
                'advantage': flat_advantages[mb_inds],
                'return': flat_returns[mb_inds],
                # FPO Specifics
                'old_loss': flat_log_probs[mb_inds], # 旧的 Loss 快照
                'noise': flat_noise[mb_inds],        # 生成当时的噪声
                'timesteps': flat_time[mb_inds]      # 生成当时的时间步
            }

    def clear(self):
        self.ptr = 0
        self.fpo_loss_snapshots = []
        self.fpo_noise = []
        self.fpo_timesteps = []