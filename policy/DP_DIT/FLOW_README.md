# Flow Matching Transformer Policy

这是一个基于Flow Matching的Transformer Policy实现，基于Diffusion Policy的DiT架构。

## 主要特性

- 使用Flow Matching替代DDPM进行生成
- 保持相同的DiT (Diffusion Transformer) 架构
- 连续时间ODE求解，更快的推理速度
- 更稳定的训练过程

## 文件结构

```
DP_DIT/
├── diffusion_policy/
│   ├── policy/
│   │   └── flow_transformer_image_policy.py  # Flow Matching Policy实现
│   └── config/
│       └── robot_flow_14.yaml                 # Flow Matching配置文件
├── train_flow.py                              # 训练脚本
├── train_flow.sh                              # 训练shell脚本
├── deploy_flow_policy.py                      # 部署脚本
└── FLOW_README.md                             # 本文档
```

## 主要修改点

### 1. Policy类 (`flow_transformer_image_policy.py`)

**调度器替换**：
- 从 `DDPMScheduler` 改为自定义的 `FlowMatchingScheduler`

**损失计算**：
- 从预测噪声改为预测流向量
- `target_velocity = trajectory - noise`

**推理采样**：
- 使用ODE求解而不是逐步去噪
- 简化Euler积分实现

### 2. 配置文件 (`robot_flow_14.yaml`)

- Policy target: `FlowTransformerImagePolicy`
- Scheduler: `FlowMatchingScheduler` (自定义实现)
- 移除了DDPM特有的beta参数

## 使用方法

### 测试实现

```bash
# 运行基础测试，验证Flow Matching数学和调度器
python test_flow_simple.py
```

### 训练

```bash
# 使用默认配置
./train_flow.sh

# 或者指定参数
./train_flow.sh robot_flow_14 realsense your_task_name

# 或者直接使用python (注意：train_flow.py已被删除，使用train.py)
python train.py --config-name=robot_flow_14 head_camera_type=realsense task_name=your_task
```

### 部署

```python
from deploy_flow_policy import FlowDP

# 加载训练好的checkpoint
flow_dp = FlowDP("path/to/flow_checkpoint.ckpt", n_obs_steps=3, n_action_steps=6)

# 使用
flow_dp.update_obs(observation)
action = flow_dp.get_action()
```

## 与DDPM版本的区别

| 特性 | DDPM | Flow Matching |
|------|------|---------------|
| 训练稳定性 | 可能不稳定 | 更稳定 |
| 推理速度 | 较慢(多步) | 更快(ODE求解) |
| 采样质量 | 良好 | 通常更好 |
| 时间处理 | 离散 | 连续 |
| 损失函数 | 预测噪声 | 预测流向量 |

## 技术实现说明

### 自定义Flow Matching调度器

由于当前diffusers版本(1.23.5)没有内置的Flow Matching调度器，我们实现了一个简化的自定义调度器：

```python
class FlowMatchingScheduler:
    def __init__(self, num_train_timesteps=100):
        self.num_train_timesteps = num_train_timesteps

    def set_timesteps(self, num_inference_steps):
        self.timesteps = torch.linspace(1.0, 0.0, num_inference_steps)

    def step(self, model_output, timestep, sample):
        dt = 1.0 / self.num_inference_steps
        return sample - dt * model_output  # Euler step
```

**实现特点**：
- 使用连续时间 [0,1] 而非离散时间步
- 简化Euler积分进行ODE求解
- 专注于核心Flow Matching数学

### Flow Matching数学

**训练目标**：
- 输入：`x_t = (1-t) * noise + t * trajectory`
- 目标：预测速度场 `v(x_t, t) = trajectory - noise`
- 损失：`L = ||pred_velocity - target_velocity||²`

**推理过程**：
- 从噪声开始 `x₁ = noise`
- 求解ODE: `dx/dt = v(x,t)` (从t=1到t=0)
- 结果即为生成的trajectory

## 注意事项

1. **超参数调整**：Flow Matching可能需要调整学习率和训练步数
2. **调度器参数**：Flow Matching使用不同的参数配置
3. **时间嵌入**：时间输入从离散变为连续[0,1]
4. **兼容性**：与现有的数据格式和环境完全兼容
5. **简化实现**：当前版本使用简化Euler方法，更高级的ODE求解器可以进一步提升性能

## 实验建议

- 从较小的学习率开始 (1e-4或更小)
- 可以尝试更少的训练步数
- 监控训练稳定性和收敛速度
- 与DDPM版本进行对比实验
