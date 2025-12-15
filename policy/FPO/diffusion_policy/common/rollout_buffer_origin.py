# import torch
# import numpy as np
# from typing import Dict, Optional, Union, Tuple, List
# from diffusion_policy.common.pytorch_util import dict_apply

# class RolloutBuffer:
#     def __init__(
#         self, 
#         num_envs: int,
#         buffer_size: int,
#         obs_shape: Dict[str, Tuple], 
#         action_shape: Tuple,
#         device: str = "cpu",
#     ):
#         self.num_envs = num_envs
#         self.buffer_size = buffer_size
#         self.device = device
#         self.ptr = 0
#         self.full = False

#         # --- 1. Observation (Nested Dict) ---
#         self.obs = {}
#         for key, shape in obs_shape.items():
#             # 预分配显存/内存
#             self.obs[key] = torch.zeros((buffer_size, num_envs) + shape).to(device)
            
#         # --- 2. Action, Reward, Done ---
#         self.actions = torch.zeros((buffer_size, num_envs) + action_shape).to(device)
#         self.rewards = torch.zeros((buffer_size, num_envs)).to(device)
#         self.dones = torch.zeros((buffer_size, num_envs)).to(device)
        
#         # --- 3. Value Function (Critic) ---
#         self.values = torch.zeros((buffer_size, num_envs)).to(device)
        
#         # --- 4. FPO Specifics (Loss Snapshot, Noise, Time) ---
#         # FPO 需要复现生成动作时的 ODE 状态
#         self.fpo_loss_snapshots = [] 
#         self.fpo_noise = []
#         self.fpo_timesteps = []

#         # --- 5. Advantages & Returns ---
#         self.advantages = torch.zeros((buffer_size, num_envs)).to(device)
#         self.returns = torch.zeros((buffer_size, num_envs)).to(device)

#     def add(
#         self, 
#         obs: Dict[str, torch.Tensor], 
#         action: torch.Tensor, 
#         reward: torch.Tensor, 
#         done: torch.Tensor, 
#         value: torch.Tensor,
#         info: Dict[str, torch.Tensor]
#     ):
#         if self.ptr >= self.buffer_size:
#             raise IndexError("RolloutBuffer is full")

#         # 存 Observation
#         for key in self.obs.keys():
#             self.obs[key][self.ptr] = obs[key].detach().to(self.device)
            
#         # 存基础 RL 数据
#         self.actions[self.ptr] = action.detach().to(self.device)
#         self.rewards[self.ptr] = reward.detach().to(self.device)
#         self.dones[self.ptr] = done.detach().to(self.device)
#         self.values[self.ptr] = value.detach().to(self.device).squeeze(-1)

#         # 存 FPO 快照数据 (用于计算概率比率)
#         self.fpo_loss_snapshots.append(info['loss'].detach().to(self.device))
#         self.fpo_noise.append(info['noise'].detach().to(self.device))
#         self.fpo_timesteps.append(info['timesteps'].detach().to(self.device))

#         self.ptr += 1

#     def compute_gae(self, next_value, gamma=0.99, gae_lambda=0.95):
#         """
#         计算 GAE (Generalized Advantage Estimation)
#         """
#         self.values = self.values.to(self.device)
#         next_value = next_value.to(self.device).squeeze(-1)
#         self.advantages = torch.zeros_like(self.rewards).to(self.device)
        
#         last_gae_lam = 0
#         for step in reversed(range(self.buffer_size)):
#             if step == self.buffer_size - 1:
#                 next_non_terminal = 1.0 - 0.0 
#                 next_val = next_value
#             else:
#                 next_non_terminal = 1.0 - self.dones[step + 1]
#                 next_val = self.values[step + 1]

#             delta = self.rewards[step] + gamma * next_val * next_non_terminal - self.values[step]
#             last_gae_lam = delta + gamma * gae_lambda * next_non_terminal * last_gae_lam
#             self.advantages[step] = last_gae_lam

#         self.returns = self.advantages + self.values

#     def get_generator(self, mini_batch_size=None):
#         """
#         将 (Steps, Envs) 数据展平并打乱，生成 Mini-batch
#         """
#         # Stack FPO info lists
#         fpo_loss = torch.stack(self.fpo_loss_snapshots) # (T, B, N)
#         fpo_noise = torch.stack(self.fpo_noise)
#         fpo_time = torch.stack(self.fpo_timesteps)
        
#         batch_size = self.buffer_size * self.num_envs
        
#         def flatten_env(x):
#             return x.reshape(batch_size, *x.shape[2:])

#         flat_obs = dict_apply(self.obs, flatten_env)
#         flat_actions = flatten_env(self.actions)
#         flat_old_loss = flatten_env(fpo_loss)
#         flat_noise = flatten_env(fpo_noise)
#         flat_time = flatten_env(fpo_time)
        
#         flat_advantages = self.advantages.reshape(batch_size)
#         flat_returns = self.returns.reshape(batch_size)
#         flat_values = self.values.reshape(batch_size)

#         indices = np.arange(batch_size)
#         np.random.shuffle(indices)

#         if mini_batch_size is None:
#             mini_batch_size = batch_size

#         for start in range(0, batch_size, mini_batch_size):
#             end = start + mini_batch_size
#             mb_inds = indices[start:end]

#             yield {
#                 'obs': dict_apply(flat_obs, lambda x: x[mb_inds]),
#                 'action': flat_actions[mb_inds],
#                 'value': flat_values[mb_inds],
#                 'advantage': flat_advantages[mb_inds],
#                 'return': flat_returns[mb_inds],
#                 'old_loss': flat_old_loss[mb_inds],
#                 'noise': flat_noise[mb_inds],
#                 'timesteps': flat_time[mb_inds]
#             }

#     def clear(self):
#         self.ptr = 0
#         self.fpo_loss_snapshots = []
#         self.fpo_noise = []
#         self.fpo_timesteps = []