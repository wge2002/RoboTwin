import torch
import numpy as np
from typing import Dict, Optional, Union, Tuple, List
from diffusion_policy.common.pytorch_util import dict_apply

class RolloutBuffer:
    def __init__(
        self, 
        num_envs: int,
        buffer_size: int,
        obs_shape: Dict[str, Tuple], 
        action_shape: Tuple,

        num_probes: int,  # 新增参数，用于预分配 FPO 数据空间 (默认为 50)
        horizon: int,

        device: str = "cpu",
    ):
        self.num_envs = num_envs
        self.buffer_size = buffer_size
        self.device = device
        self.ptr = 0
        self.full = False
        action_dim = action_shape[-1]

        # --- 1. Observation (Nested Dict) ---
        self.obs = {}
        for key, shape in obs_shape.items():
            # 预分配显存/内存
            self.obs[key] = torch.zeros((buffer_size, num_envs) + shape).to(device)
            
        # --- 2. Action, Reward, Done ---
        self.actions = torch.zeros((buffer_size, num_envs) + action_shape).to(device)
        self.rewards = torch.zeros((buffer_size, num_envs)).to(device)
        self.dones = torch.zeros((buffer_size, num_envs)).to(device)
        
        # --- 3. Value Function (Critic) ---
        self.values = torch.zeros((buffer_size, num_envs)).to(device)
        
        # --- 4. FPO Specifics (Loss Snapshot, Noise, Time) ---
        # FPO 需要复现生成动作时的 ODE 状态
        # self.fpo_loss_snapshots = [] 
        # self.fpo_noise = []
        # self.fpo_timesteps = []

        # --- 4. FPO Specifics (Loss Snapshot, Noise, Time) ---
        # 修正: 替换为预分配的 Tensors
        # Loss Snapshot (Old Loss): (buffer_size, num_envs, num_probes)
        self.fpo_loss_snapshots = torch.zeros((buffer_size, num_envs, num_probes, horizon)).to(device)
        
        # Noise: (buffer_size, num_envs, num_probes, H, D)
        self.fpo_noise = torch.zeros((buffer_size, num_envs, num_probes, horizon, action_dim)).to(device)
        
        # Timesteps: (buffer_size, num_envs, num_probes, H)
        self.fpo_timesteps = torch.zeros((buffer_size, num_envs, num_probes, horizon)).to(device)

        # --- 5. Advantages & Returns ---
        self.advantages = torch.zeros((buffer_size, num_envs)).to(device)
        self.returns = torch.zeros((buffer_size, num_envs)).to(device)

    def add(
        self, 
        obs: Dict[str, torch.Tensor], 
        action: torch.Tensor, 
        reward: torch.Tensor, 
        done: torch.Tensor, 
        value: torch.Tensor,
        info: Dict[str, torch.Tensor]
    ):
        if self.ptr >= self.buffer_size:
            raise IndexError("RolloutBuffer is full")

        # 存 Observation
        for key in self.obs.keys():
            self.obs[key][self.ptr] = obs[key].detach().to(self.device)
            
        # 存基础 RL 数据
        self.actions[self.ptr] = action.detach().to(self.device) # action 形状为 (1, 8, 14)，现在可以存入 (1, 8, 14) 的槽位
        # reward, done, value 形状修正不变
        self.rewards[self.ptr] = reward.detach().to(self.device).squeeze(0)
        self.dones[self.ptr] = done.detach().to(self.device).squeeze(0)    
        self.values[self.ptr] = value.detach().to(self.device).squeeze(0)   
        
        # 补全 FPO 专用数据存储逻辑
        horizon = self.fpo_noise.shape[-2] # 从预分配的张量中获取 H=8

        # FPO Noise: Policy 输出 (1, 50, 8, 14)，Buffer 期望 (1, 50, 8, 14)
        # 移除 .squeeze(0)
        self.fpo_noise[self.ptr] = info['noise'].detach().to(self.device)

        # FPO Loss: Policy 输出 (1, 50)。需要扩展到 (1, 50, 8)
        # (1, 50) -> (1, 50, 1) -> repeat H 次 -> (1, 50, H)
        loss_padded = info['loss'].unsqueeze(-1).repeat(1, 1, horizon)
        self.fpo_loss_snapshots[self.ptr] = loss_padded.detach().to(self.device)

        # FPO Timesteps: Policy 输出 (1, 50)。需要扩展到 (1, 50, 8)
        # (1, 50) -> (1, 50, 1) -> repeat H 次 -> (1, 50, H)
        time_padded = info['timesteps'].unsqueeze(-1).repeat(1, 1, horizon)
        self.fpo_timesteps[self.ptr] = time_padded.detach().to(self.device)

        # 更新指针
        self.ptr += 1
        self.full = self.ptr >= self.buffer_size
        # self.rewards[self.ptr] = reward.detach().to(self.device)
        # self.dones[self.ptr] = done.detach().to(self.device)
        # self.values[self.ptr] = value.detach().to(self.device).squeeze(-1)

        # # 存 FPO 快照数据 (用于计算概率比率)
        # self.fpo_loss_snapshots.append(info['loss'].detach().to(self.device))
        # self.fpo_noise.append(info['noise'].detach().to(self.device))
        # self.fpo_timesteps.append(info['timesteps'].detach().to(self.device))

        # self.ptr += 1

    def compute_gae(self, next_value, gamma=0.99, gae_lambda=0.95):
        """
        计算 GAE (Generalized Advantage Estimation)
        """
        self.values = self.values.to(self.device)
        next_value = next_value.to(self.device).squeeze(-1)
        self.advantages = torch.zeros_like(self.rewards).to(self.device)
        
        last_gae_lam = 0
        for step in reversed(range(self.buffer_size)):
            if step == self.buffer_size - 1:
                next_non_terminal = 1.0 - 0.0 
                next_val = next_value
            else:
                next_non_terminal = 1.0 - self.dones[step + 1]
                next_val = self.values[step + 1]

            delta = self.rewards[step] + gamma * next_val * next_non_terminal - self.values[step]
            last_gae_lam = delta + gamma * gae_lambda * next_non_terminal * last_gae_lam
            self.advantages[step] = last_gae_lam

        self.returns = self.advantages + self.values

    def get_generator(self, mini_batch_size=None):

        """
        将 (Steps, Envs) 数据展平并打乱，生成 Mini-batch
        """
        # 【FIX】使用实际收集的步数 self.ptr
        actual_buffer_size = self.ptr 
        
        # 【FIX】直接使用预分配的 Tensor，并进行切片
        fpo_loss = self.fpo_loss_snapshots[:actual_buffer_size] # (T_actual, B, N)
        fpo_noise = self.fpo_noise[:actual_buffer_size]
        fpo_time = self.fpo_timesteps[:actual_buffer_size]
        
        # 【FIX】Batch size 基于实际收集的数据
        batch_size = actual_buffer_size * self.num_envs
        
        def flatten_env_and_slice(x):
            # 先切片，再展平
            return x[:actual_buffer_size].reshape(batch_size, *x.shape[2:])

        flat_obs = dict_apply(self.obs, flatten_env_and_slice)
        flat_actions = flatten_env_and_slice(self.actions)
        
        # FPO data is not nested obs, so we flatten directly
        flat_old_loss = fpo_loss.reshape(batch_size, *fpo_loss.shape[2:])
        flat_noise = fpo_noise.reshape(batch_size, *fpo_noise.shape[2:])
        flat_time = fpo_time.reshape(batch_size, *fpo_time.shape[2:])

        # 【FIX】Value/Advantage/Return 也使用切片
        flat_advantages = self.advantages[:actual_buffer_size].reshape(batch_size)
        flat_returns = self.returns[:actual_buffer_size].reshape(batch_size)
        flat_values = self.values[:actual_buffer_size].reshape(batch_size)

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
                'old_loss': flat_old_loss[mb_inds],
                'noise': flat_noise[mb_inds],
                'timesteps': flat_time[mb_inds]
            }

    def clear(self):
        self.ptr = 0
        self.full = False
        #self.fpo_loss_snapshots = []
        #self.fpo_noise = []
        #self.fpo_timesteps = []