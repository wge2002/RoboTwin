"""
FPO Training Script for RoboTwin
Usage:
    python train_fpo.py --config-name=robot_fpo_14
"""

import sys
# use line-buffering for both stdout and stderr
sys.stdout = open(sys.stdout.fileno(), mode='w', buffering=1)
sys.stderr = open(sys.stderr.fileno(), mode='w', buffering=1)

import hydra
from omegaconf import OmegaConf
import pathlib
from easydict import EasyDict
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import copy
import random
import wandb
import os
import numpy as np
from tqdm import tqdm

from diffusion_policy.workspace.base_workspace import BaseWorkspace
from diffusion_policy.policy.flow_fpo_policy import FlowFPOPolicy
from diffusion_policy.common.rollout_buffer import RolloutBuffer
from diffusion_policy.common.pytorch_util import dict_apply, optimizer_to
from diffusion_policy.common.checkpoint_util import TopKCheckpointManager
from diffusion_policy.common.json_logger import JsonLogger
from diffusion_policy.common.env_util import get_env_type, make_env

# 引入你的 Policy 和 Buffer
# 确保 diffusion_policy 已经在 python path 下
# ---------------------------------------------------------

OmegaConf.register_new_resolver("eval", eval, replace=True)

class TrainFPOWorkspace(BaseWorkspace):
    include_keys = ['global_step', 'epoch']

    def __init__(self, cfg: OmegaConf, output_dir=None):
        super().__init__(cfg, output_dir=output_dir)

        # 1. 设置种子
        seed = cfg.training.seed
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)

        # 2. 创建模型 (Policy)
        # 这里会实例化 FlowFPOPolicy
        self.model: FlowFPOPolicy = hydra.utils.instantiate(cfg.policy)
        self.ema_model: FlowFPOPolicy = None
        if cfg.training.use_ema:
            self.ema_model = copy.deepcopy(self.model)

        # 3. 优化器 (Actor 和 Critic 分开优化通常更好，这里为了简单先合用一个，或者分开)
        # FPO 论文通常用同一个 optimizer 更新所有参数
        self.optimizer = hydra.utils.instantiate(
            cfg.optimizer, params=self.model.parameters()
        )

        # 4. 全局计步器
        self.global_step = 0
        self.epoch = 0

    def run(self):
        cfg = self.cfg
        
        # ---------------------------------------------------------
        # 1. 环境初始化 (RL 需要真实的交互环境)
        # ---------------------------------------------------------
        # 假设 cfg.task.env 包含环境配置
        # 如果是 RoboTwin，通常 cfg.task.env_runner 里面有环境信息
        # 这里我们需要手动创建一个用于 Rollout 的 Env
        print("Creating Environment for Rollout...")
        # 注意：这里需要根据你的具体 yaml 结构调整。
        # 假设你有一个可以实例化的 env 配置
        try:
            env = hydra.utils.instantiate(cfg.task.env)
        except Exception as e:
            print(f"Failed to instantiate env from cfg.task.env: {e}")
            print("Trying to fallback to creating env manually or from runner config...")
            # 这是一个回退方案，取决于你的 config 结构
            # 你可能需要手动写 make_env 函数
            raise e
        
        # ---------------------------------------------------------
        # 2. Rollout Buffer 初始化
        # ---------------------------------------------------------
        # 获取环境形状信息
        # 跑一步 reset 拿 shape
        obs = env.reset()
        # 确保 obs 是 dict
        if not isinstance(obs, dict):
            # 如果是数组，封装成 dict
            obs = {'obs': obs}
            
        # 获取 Action Shape
        # 假设 env.action_space 存在，或者通过 step 获取
        dummy_action = env.action_space.sample()
        action_shape = dummy_action.shape
        
        # Obs Shape map
        obs_shape = {k: v.shape for k, v in obs.items()}
        
        # 从配置读取 Buffer 参数
        num_envs = 1 # 暂时假设单环境
        rollout_steps = cfg.training.rollout_steps # e.g. 2048
        
        buffer = RolloutBuffer(
            num_envs=num_envs,
            buffer_size=rollout_steps,
            obs_shape=obs_shape,
            action_shape=action_shape,
            device=self.device
        )
        
        # ---------------------------------------------------------
        # 3. 训练准备
        # ---------------------------------------------------------
        self.model.to(self.device)
        if self.ema_model:
            self.ema_model.to(self.device)
            
        # Checkpoint manager
        checkpoint_manager = TopKCheckpointManager(
            save_dir=os.path.join(self.output_dir, 'checkpoints'),
            monitor_key='train_loss',
            mode='min',
            k=cfg.checkpoint.topk,
            format_str='epoch={epoch:03d}-train_loss={train_loss:.3f}.ckpt'
        )

        # Logging
        wandb_run = wandb.init(
            dir=self.output_dir,
            config=OmegaConf.to_container(cfg, resolve=True),
            project=cfg.logging.project,
            name=cfg.logging.name,
            resume=True
        )
        wandb.config.update({"output_dir": self.output_dir})

        # PPO/FPO Hyperparams
        ppo_epochs = cfg.training.ppo_epochs # e.g. 4
        mini_batch_size = cfg.training.batch_size # e.g. 64
        clip_epsilon = cfg.training.clip_epsilon # 0.2
        entropy_coef = cfg.training.entropy_coef # 0.0
        critic_coef = cfg.training.critic_coef # 0.5
        max_iterations = cfg.training.max_iterations # 总共训练多少轮

        print(f"Start FPO Training Loop for {max_iterations} iterations...")
        
        # ---------------------------------------------------------
        # 4. 主循环 (Iteration -> Rollout -> Update)
        # ---------------------------------------------------------
        for iteration in range(max_iterations):
            self.model.eval()
            buffer.clear()
            
            # === Phase 1: Data Collection (Rollout) ===
            print(f"Iteration {iteration}: Collecting Rollouts...")
            obs = env.reset()
            # 确保 obs 格式正确
            # ...
            
            total_reward = 0
            steps = 0
            
            # 使用 tqdm 显示收集进度
            pbar = tqdm(range(rollout_steps), desc="Rollout")
            for _ in pbar:
                # 1. 整理 Obs
                # obs_dict 需要转成 Tensor 并增加 Batch 维度 (1, ...)
                obs_tensor = dict_apply(obs, lambda x: torch.from_numpy(x).unsqueeze(0).to(self.device))
                
                # 2. 采样动作 & 获取 Info (Loss Snapshot)
                with torch.no_grad():
                    # sample_action_with_info 内部会调用 sample，并计算 snapshot loss
                    # 注意：sample_action_with_info 返回的 action 应该是 unnormalized 的 (如果 policy 内部处理了)
                    # 你的 sample_action_with_info 返回的是 action (unnorm) 和 info
                    action, info = self.model.sample_action_with_info(obs_tensor)
                    
                    # 同时预测 Value
                    value = self.model.predict_value(obs_tensor)

                # 3. 执行动作
                # action 是 (1, horizon, dim) 或 (1, dim)
                # env.step 通常只需要 (dim,)
                action_cpu = action.squeeze(0).cpu().numpy()
                # 如果 horizon > 1，这里需要决定怎么执行（Receding Horizon）
                # 简单起见，假设 Env 能接受这个 action，或者只执行第一步
                # 这里假设 action 维度匹配 env
                
                next_obs, reward, done, env_info = env.step(action_cpu)
                
                # 累加 Reward 用于显示
                total_reward += reward
                
                # 4. 存入 Buffer
                # 注意：Reward 和 Done 需要转成 Tensor
                reward_tensor = torch.tensor([reward], device=self.device, dtype=torch.float32)
                done_tensor = torch.tensor([done], device=self.device, dtype=torch.float32)
                
                buffer.add(
                    obs=obs_tensor, # (1, ...)
                    action=action,  # (1, ...)
                    reward=reward_tensor,
                    done=done_tensor,
                    value=value,    # (1, 1)
                    info=info       # Dict of tensors
                )
                
                obs = next_obs
                steps += 1
                
                if done:
                    obs = env.reset()
            
            # === Phase 2: Compute Advantage (GAE) ===
            with torch.no_grad():
                # 计算最后一个状态的 Value 用于 bootstrap
                last_obs_tensor = dict_apply(obs, lambda x: torch.from_numpy(x).unsqueeze(0).to(self.device))
                next_value = self.model.predict_value(last_obs_tensor)
                buffer.compute_gae(next_value)

            # === Phase 3: Update (PPO/FPO) ===
            self.model.train()
            train_losses = []
            policy_losses = []
            value_losses = []
            ratios = []
            
            print(f"Iteration {iteration}: Updating Policy...")
            for _ in range(ppo_epochs):
                for batch in buffer.get_generator(mini_batch_size):
                    # batch 里的数据已经是 (B, ...) 形状的 Tensor 了
                    
                    # 1. 重新计算 CFM Loss (New Loss)
                    # 需要用 batch['noise'] 和 batch['timesteps'] 来复现
                    new_loss = self.model.compute_cfm_loss(
                        batch, 
                        noise=batch['noise'], 
                        timesteps=batch['timesteps'], 
                        reduction='none'
                    )
                    # new_loss shape: (B,)
                    
                    # 2. 获取 Old Loss Snapshot
                    # old_loss 是 add 进去的时候计算的 (B, N_probes) 的均值或 snapshot
                    # 注意：buffer 里存的 old_loss 是 (B, N_probes)
                    # compute_cfm_loss 返回的是 mean over probes 吗？
                    # 你的 sample_action_with_info 里 info['loss'] 是 (B, N)
                    # 你的 compute_cfm_loss 如果 reduction='none' 返回的是 (B,) 还是 (B, N)?
                    # 看代码：loss = reduce(loss, "b ... -> b", "mean") -> (B,)
                    # 等等！sample_action_with_info 里你是怎么调用的？
                    # loss_tensor = compute_cfm_loss(..., reduction='none')
                    # 如果 input 是 (B*N, ...)，output 是 (B*N,)
                    # 然后 view 成 (B, N)
                    # 所以 old_loss 是 (B, N)
                    
                    # 现在的 new_loss 是基于 batch['obs'] (B, ...) 和 batch['action'] (B, ...) 算的
                    # 这里有一个 mismatch：
                    # Buffer 里的 action 是 rollout 产生的 (B, ...)，noise 是 (B, N, ...), time 是 (B, N)
                    # 我们 update 的时候，实际上是对 buffer 里的 (obs, action) 对进行评估
                    # 但 FPO 的 Loss 是期望形式 $E_{t, eps} [ ||v - (eps-x)|| ]$
                    # 这里的逻辑稍微有点绕：
                    # 我们需要计算的是：在这个 batch 的 (obs, action) 下，
                    # 保持和 old_loss 相同的 noise 和 timesteps，新的 loss 是多少？
                    
                    # 你的 compute_cfm_loss 默认输入 noise 是 (B, ...)，timesteps 是 (B,)
                    # 但 buffer 里存的 noise 是 (B, N_probes, ...)。
                    # 我们需要把 Batch 里的 obs 和 action 复制 N_probes 份，展平，算完 loss 再变回来？
                    # 这是一个计算量巨大的操作。
                    
                    # **简化方案**：
                    # 为了效率，通常 Update 时不再次对 50 个 Probe 采样。
                    # 而是：Rollout 时只存 1 个 Probe 的 Loss？不，那样方差太大。
                    # 或者：Update 时，只用 Buffer 里存的那 N 个 Probe 的 loss 均值？
                    # 
                    # 让我们回看 JAX 代码 `_compute_fpo_loss`:
                    # 它是把 noise (B, N, ...) 传进去了。
                    # 所以你需要修改 compute_cfm_loss 或者在这里手动处理 reshape。
                    
                    # --- 手动处理 Multi-Probe Batch ---
                    B = batch['action'].shape[0]
                    N = batch['noise'].shape[1] # N_probes
                    
                    # 展平 Obs: (B, ...) -> (B*N, ...)
                    obs_rep = dict_apply(batch['obs'], lambda x: x.repeat_interleave(N, dim=0))
                    action_rep = batch['action'].repeat_interleave(N, dim=0)
                    
                    # 展平 Noise/Time: (B, N, ...) -> (B*N, ...)
                    noise_flat = batch['noise'].reshape(-1, *batch['noise'].shape[2:])
                    time_flat = batch['timesteps'].reshape(-1)
                    
                    # 计算 New Loss (Flattened)
                    # new_loss_flat: (B*N, )
                    new_loss_flat = self.model.compute_cfm_loss(
                        {'obs': obs_rep, 'action': action_rep},
                        noise=noise_flat,
                        timesteps=time_flat,
                        reduction='none'
                    )
                    
                    # Reshape 回 (B, N)
                    new_loss = new_loss_flat.view(B, N)
                    old_loss = batch['old_loss'] # (B, N)