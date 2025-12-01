from typing import Dict, Tuple, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, reduce

from diffusion_policy.model.common.normalizer import LinearNormalizer
from diffusion_policy.policy.base_image_policy import BaseImagePolicy
from diffusion_policy.model.diffusion.transformer_for_diffusion import TransformerForDiffusion
from diffusion_policy.model.vision.multi_image_obs_encoder import MultiImageObsEncoder
from diffusion_policy.common.pytorch_util import dict_apply


class FlowMatchingScheduler:
    """
    Standard OT-CFM Scheduler for Horizon=1.
    Time: 0.0 (Noise) -> 1.0 (Data)
    Update: x_{t+1} = x_t + v * dt
    """
    def __init__(self, num_train_timesteps=100):
        # 兼容性参数，实际推理用 set_timesteps
        self.num_train_timesteps = num_train_timesteps

    def set_timesteps(self, num_inference_steps):
        self.num_inference_steps = num_inference_steps
        # 生成 N 个时间点: [0.0, ..., 0.9] (假设 N=10)，最后一步迈向 1.0
        self.timesteps = torch.linspace(0.0, 1.0, num_inference_steps + 1)[:-1]

    def step(self, model_output, timestep, sample):
        """
        Euler Integration Step
        model_output: velocity v(x, t)
        """
        dt = 1.0 / self.num_inference_steps
        # 顺着流场方向积分：从噪声流向数据 (加法)
        prev_sample = sample + dt * model_output
        return prev_sample


class FlowFPOPolicy(BaseImagePolicy):
    def __init__(
        self,
        shape_meta: dict,
        obs_encoder: MultiImageObsEncoder,
        horizon: int, # 这个参数通常来自配置，我们会强制检查它是否为 1
        n_action_steps: int,
        n_obs_steps: int,
        num_inference_steps=None,
        # transformer parameters
        n_layer=12,
        n_head=12,
        n_emb=768,
        p_drop_emb=0.1,
        p_drop_attn=0.1,
        causal_attn=False,
        **kwargs,
    ):
        super().__init__()
        print('Initializing FlowFPOPolicy (Horizon=1 Mode)')

        # =====================================================
        # 1. 强制约束检查：确保真的是 Horizon=1 的 MDP 设定
        # =====================================================
        if n_action_steps != 1:
            raise ValueError(f"FPO Policy requires n_action_steps=1, got {n_action_steps}")
        if n_obs_steps != 1:
            raise ValueError(f"FPO Policy requires n_obs_steps=1, got {n_obs_steps}")
        # 虽然 horizon 参数可能在 yaml 里写了 16，但在这里我们强制把它覆盖为 1
        # 或者报错。为了安全起见，我们在这里强制覆盖内部使用，并打印警告。
        if horizon != 1:
            print(f"Warning: Config horizon is {horizon}, but FlowFPOPolicy enforces horizon=1.")
        
        self.horizon = 1  # 强制为 1
        self.n_action_steps = 1
        self.n_obs_steps = 1

        # 2. 解析形状
        action_shape = shape_meta["action"]["shape"]
        assert len(action_shape) == 1
        action_dim = action_shape[0]
        obs_feature_dim = obs_encoder.output_shape()[0]

        # 3. 构建 DiT 模型 (Sequence Length = 1)
        # 输入: Action (B, 1, D)
        # 条件: Obs (B, 1, D_feat) -> 作为 Global Cond
        
        input_dim = action_dim
        # Global Cond Dim 就是 Obs Feature Dim
        global_cond_dim = obs_feature_dim 

        self.model = TransformerForDiffusion(
            input_dim=input_dim,
            output_dim=input_dim,
            horizon=self.horizon, # 传入 1
            n_obs_steps=self.n_obs_steps, # 传入 1
            cond_dim=global_cond_dim,
            n_layer=n_layer,
            n_head=n_head,
            n_emb=n_emb,
            p_drop_emb=p_drop_emb,
            p_drop_attn=p_drop_attn,
            causal_attn=causal_attn,
            time_as_cond=True, 
            obs_as_cond=True, # 使用 Global Conditioning
        )

        self.obs_encoder = obs_encoder
        self.noise_scheduler = FlowMatchingScheduler()
        self.normalizer = LinearNormalizer()
        
        self.action_dim = action_dim
        self.obs_feature_dim = obs_feature_dim
        self.num_inference_steps = num_inference_steps

    # ==========================================
    # 核心 1: 纯净的 ODE 采样 (Batch-Wise)
    # ==========================================
    def conditional_sample(
        self,
        batch_size,
        global_cond,
        generator=None
    ):
        """
        采样过程：从噪声 -> 动作
        Shape: (B, 1, Action_Dim)
        """
        device = self.device
        dtype = self.dtype
        scheduler = self.noise_scheduler

        # 1. 初始化噪声 (B, 1, D)
        trajectory = torch.randn(
            size=(batch_size, self.horizon, self.action_dim),
            dtype=dtype,
            device=device,
            generator=generator,
        )

        # 2. 设置时间步 0 -> 1
        scheduler.set_timesteps(self.num_inference_steps)

        # 3. 积分循环
        for t in scheduler.timesteps:
            # 时间输入 (B,) 并放大以激活 Time Embedding
            t_input = torch.full((batch_size,), t, device=device) * 1000
            
            # 预测流场 v
            # global_cond: (B, D_feat)
            velocity = self.model(trajectory, t_input, cond=global_cond)
            
            # 欧拉积分
            trajectory = scheduler.step(velocity, t, trajectory)

        return trajectory

    # ==========================================
    # 核心 2: 预测动作 (给 Runner/Eval 用)
    # ==========================================
    def predict_action(self, obs_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        输入: Obs Dict (B, T, ...) 
        输出: Action (B, 1, D)
        注意: 即使输入 Obs 包含历史 T，我们也只取最后一帧 (T=1)
        """
        # 1. 归一化
        nobs = self.normalizer.normalize(obs_dict)
        
        # 2. 提取特征 (ResNet)
        # 输入形状通常是 (B, n_obs_steps, C, H, W)
        # 我们只取这一帧：(B, C, H, W)
        # 假设 obs_dict 的 tensor 是 [B, T, ...]，这里 T=1
        this_nobs = dict_apply(nobs, lambda x: x[:, 0, ...]) 
        
        # ResNet 输出: (B, D_feat)
        nobs_features = self.obs_encoder(this_nobs)
        
        # 3. 采样
        batch_size = nobs_features.shape[0]
        # 返回 (B, 1, D)
        n_action = self.conditional_sample(
            batch_size=batch_size,
            global_cond=nobs_features
        )

        # 4. 反归一化
        action = self.normalizer["action"].unnormalize(n_action)

        # 返回结果
        # action_pred 和 action 是一样的，因为 Horizon=1
        return {
            "action": action, 
            "action_pred": action
        }

    # ==========================================
    # 核心 3: FPO 数据收集 (Rollout)
    # ==========================================
    def sample_action_with_info(self, obs_dict: Dict[str, torch.Tensor], num_probes: int = 50) -> Dict[str, torch.Tensor]:
        """
        FPO 专用接口：生成动作 + 计算 Loss 快照
        """
        # 1. 生成动作
        # action: (B, 1, D)
        res = self.predict_action(obs_dict)
        action = res['action']
        
        # 2. 准备探针 (Probe) - 扩展 Batch
        # 我们要计算：P(action | obs)
        # 用 N 个噪声来估算这个概率密度 (Loss)
        
        device = self.device
        B = action.shape[0] # Batch Size (通常为 1)
        
        # 复制 Obs (B -> B*N)
        obs_rep = dict_apply(obs_dict, lambda x: x.repeat_interleave(num_probes, dim=0))
        # 复制 Action (B -> B*N)
        # 注意：这里用的是 unnormalized 的 action
        # 实际上我们这里直接拿 res['action'] (unnormalized)，然后在 compute_cfm_loss 里归一化
        # 保持逻辑统一
        action_rep = action.repeat_interleave(num_probes, dim=0)
        
        current_bsz = B * num_probes

        # 3. 生成随机探针
        # 噪声形状必须和 (B*N, 1, D) 一致
        probe_noise = torch.randn(action_rep.shape, device=device)
        # 时间形状 (B*N,)
        probe_timesteps = torch.rand((current_bsz,), device=device)

        # 4. 计算 Snapshot Loss
        batch = {
            'obs': obs_rep,
            'action': action_rep
        }
        
        with torch.no_grad():
            # 计算 loss，保留 (B*N,)
            loss_tensor = self.compute_cfm_loss(
                batch, 
                noise=probe_noise, 
                timesteps=probe_timesteps, 
                reduction='none'
            )

        # 5. 打包 Info
        info = {
            # (B, N)
            'loss': loss_tensor.view(B, num_probes),
            # (B, N, 1, D)
            'noise': probe_noise.view(B, num_probes, *probe_noise.shape[1:]),
            # (B, N)
            'timesteps': probe_timesteps.view(B, num_probes),
            # (B, 1, D) - 存下动作本身，防止 Buffer 里存的是切片（虽然这里 horizon=1 不存在切片问题，但保持一致性）
            'action_full': action 
        }

        return action, info

    # ==========================================
    # 核心 4: CFM Loss 计算 (Training)
    # ==========================================
    def compute_cfm_loss(self, batch, noise=None, timesteps=None, reduction='mean'):
        """
        Input:
            batch['obs']: (B, 1, ...)
            batch['action']: (B, 1, D)
        """
        # 1. 归一化
        nobs = self.normalizer.normalize(batch["obs"])
        nactions = self.normalizer["action"].normalize(batch["action"])
        
        # 2. 提取条件 (Global Cond)
        # 取第0帧
        this_nobs = dict_apply(nobs, lambda x: x[:, 0, ...]) 
        global_cond = self.obs_encoder(this_nobs) # (B, D_feat)
        
        # 3. 准备数据 (x1)
        # trajectory: (B, 1, D)
        trajectory = nactions 
        batch_size = trajectory.shape[0]

        # 4. 准备噪声和时间
        if noise is None:
            noise = torch.randn(trajectory.shape, device=trajectory.device)
        if timesteps is None:
            timesteps = torch.rand(batch_size, device=trajectory.device)
        
        # 5. Flow Mixing (x_t)
        # x_t = (1-t) * noise + t * data
        # timesteps: [0, 1]
        t_view = timesteps.unsqueeze(-1).unsqueeze(-1) # (B, 1, 1)
        noisy_trajectory = (1 - t_view) * noise + t_view * trajectory

        # 6. 模型预测
        # input: noisy_trajectory (B, 1, D)
        # time: timesteps * 1000 (B,)
        # cond: global_cond (B, D_feat)
        pred_velocity = self.model(noisy_trajectory, timesteps * 1000, cond=global_cond)

        # 7. 目标与 Loss
        # v = x1 - x0
        target_velocity = trajectory - noise
        
        loss = F.mse_loss(pred_velocity, target_velocity, reduction='none')
        
        # Reduce: mean over (1, D) dimensions -> (B,)
        loss = reduce(loss, "b ... -> b", "mean")

        if reduction == 'mean':
            return loss.mean()
        elif reduction == 'sum':
            return loss.sum()
        elif reduction == 'none':
            return loss
        else:
            raise ValueError(f"Unknown reduction {reduction}")

    def set_normalizer(self, normalizer: LinearNormalizer):
        self.normalizer.load_state_dict(normalizer.state_dict())

    def compute_loss(self, batch):
        """
        标准 BC (Behavior Cloning) 训练接口
        """
        # 1. 强制检查 Horizon=1 (防止 DataLoader 传错)
        assert "valid_mask" not in batch
        
        # 2. 直接调用核心 CFM Loss
        # compute_cfm_loss 内部会自动生成随机 noise 和 random timesteps [0,1]
        loss = self.compute_cfm_loss(
            batch, 
            noise=None, 
            timesteps=None, 
            reduction='mean'
        )
        
        return loss