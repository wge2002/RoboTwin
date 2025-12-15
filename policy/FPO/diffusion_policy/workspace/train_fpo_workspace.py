import sys
import os
import hydra
from omegaconf import OmegaConf
import copy
import random
import wandb
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
import yaml
import importlib

from diffusion_policy.workspace.base_workspace import BaseWorkspace
from diffusion_policy.policy.flow_fpo_policy import FlowFPOPolicy
from diffusion_policy.common.rollout_buffer import RolloutBuffer
from diffusion_policy.common.pytorch_util import dict_apply
from diffusion_policy.common.checkpoint_util import TopKCheckpointManager

# 尝试导入 envs，假设运行路径在项目根目录
# try:
#     import envs
#     from envs import CONFIGS_PATH
#     print('1111')
# except ImportError:
#     # 如果找不到，尝试把上级目录加进去 (适配 RoboTwin 结构)
#     # 假设当前在 policy/FPO/
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../.."))
if project_root not in sys.path:
    sys.path.append(project_root)
import envs
from envs import CONFIGS_PATH
# print('2222')

class TrainFPOWorkspace(BaseWorkspace):
    include_keys = ['global_step', 'epoch']

    def __init__(self, cfg: OmegaConf, output_dir=None):
        super().__init__(cfg, output_dir=output_dir)

        # 1. 设置随机种子
        seed = cfg.training.seed
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)

        self.device = torch.device(cfg.training.device)

        # 2. 创建 Policy
        self.model: FlowFPOPolicy = hydra.utils.instantiate(cfg.policy)
        
        self.ema_model: FlowFPOPolicy = None
        if cfg.training.use_ema:
            self.ema_model = copy.deepcopy(self.model)

        # 3. 优化器
        self.optimizer = hydra.utils.instantiate(
            cfg.optimizer, params=self.model.parameters()
        )

        self.global_step = 0
        self.epoch = 0

    # ====================================================
    # 辅助函数：从 eval_policy.py 移植的环境加载逻辑
    # ====================================================
    def _load_env_args(self):
        cfg = self.cfg
        task_name = cfg.task.name
        task_config = cfg.setting  # 对应 eval 中的 task_config (e.g. demo_clean)
        
        print(f"[Env Init] Loading configs for task: {task_name}, setting: {task_config}")

        # 1. 读取 Task Config
        task_config_path = os.path.join(CONFIGS_PATH, f"{task_config}.yml")
        if not os.path.exists(task_config_path):
            # Fallback path logic if needed
            task_config_path = f"./task_config/{task_config}.yml"
            
        with open(task_config_path, "r", encoding="utf-8") as f:
            args = yaml.load(f.read(), Loader=yaml.FullLoader)

        args['task_name'] = task_name
        args["task_config"] = task_config
        # 训练时不需要 ckpt_setting，但也给个默认值防止报错
        args["ckpt_setting"] = "train_rl" 

        # 2. 读取 Embodiment Config
        embodiment_type = args.get("embodiment")
        embodiment_config_path = os.path.join(CONFIGS_PATH, "_embodiment_config.yml")
        
        with open(embodiment_config_path, "r", encoding="utf-8") as f:
            _embodiment_types = yaml.load(f.read(), Loader=yaml.FullLoader)

        def get_embodiment_file(etype):
            return _embodiment_types[etype]["file_path"]

        # 3. 读取 Camera Config
        camera_config_path = os.path.join(CONFIGS_PATH, "_camera_config.yml")
        with open(camera_config_path, "r", encoding="utf-8") as f:
            _camera_config = yaml.load(f.read(), Loader=yaml.FullLoader)

        head_camera_type = args["camera"]["head_camera_type"]
        # 如果有命令行 override，优先使用 cfg 里的
        if "head_camera_type" in cfg and cfg.head_camera_type:
            head_camera_type = cfg.head_camera_type
            args["camera"]["head_camera_type"] = head_camera_type

        args["head_camera_h"] = _camera_config[head_camera_type]["h"]
        args["head_camera_w"] = _camera_config[head_camera_type]["w"]

        # 4. 解析 Embodiment 文件路径
        if len(embodiment_type) == 1:
            args["left_robot_file"] = get_embodiment_file(embodiment_type[0])
            args["right_robot_file"] = get_embodiment_file(embodiment_type[0])
            args["dual_arm_embodied"] = True
        elif len(embodiment_type) == 3:
            args["left_robot_file"] = get_embodiment_file(embodiment_type[0])
            args["right_robot_file"] = get_embodiment_file(embodiment_type[1])
            args["embodiment_dis"] = embodiment_type[2]
            args["dual_arm_embodied"] = False
        
        # 5. 读取具体 Robot Config
        def get_robot_config(robot_file):
            # robot_file 可能是相对路径，需要拼全
            # 假设 robot_file 是相对于项目根目录的 assets/embodiments/...
            # 这里的路径处理可能需要根据实际情况微调
            if not os.path.exists(os.path.join(robot_file, "config.yml")):
                # 尝试加上项目根目录前缀
                project_root = os.path.dirname(CONFIGS_PATH.rstrip('/')) # 粗略估计
                # 更稳妥的方法：假设 robot_file 已经是正确的相对路径
                pass
            
            with open(os.path.join(robot_file, "config.yml"), "r", encoding="utf-8") as f:
                return yaml.load(f.read(), Loader=yaml.FullLoader)

        args["left_embodiment_config"] = get_robot_config(args["left_robot_file"])
        args["right_embodiment_config"] = get_robot_config(args["right_robot_file"])
        
        # 6. 其他训练需要的参数
        args["render_freq"] = 0 # 训练通常不需要渲染到屏幕，除非 debug
        args["eval_video_log"] = False # 暂时关闭视频录制
        
        return args

    def _init_task_env(self, task_name):
        # 动态导入 envs.{task_name}
        try:
            envs_module = importlib.import_module(f"envs.{task_name}")
            env_class = getattr(envs_module, task_name)
            env_instance = env_class()
            return env_instance
        except Exception as e:
            raise RuntimeError(f"Failed to load task env: {task_name}. Error: {e}")
    
    # 在 TrainFPOWorkspace 类中添加或修改这个方法
    def preprocess_obs(self, raw_obs):
        """
        1. 提取嵌套数据：从 obs['observation']['head_camera']['rgb'] 拿数据
        2. 转换维度：(H, W, C) -> (C, H, W)
        3. 归一化/类型转换：(0-255 uint8) -> (0-1 float32) [这一步通常由 dataset 做，但在线 rollout 要自己处理]
           注意：这里 RolloutBuffer 通常存 float32 或者 uint8。
           为了和 Policy 兼容，建议先存 uint8 或 float，但在输入模型前必须是 CHW。
           这里我们只做提取和转置。
        """
        new_obs = {}
        
        # --- A. 处理图像 (head_cam) ---
        # 你的路径是: raw_obs['observation']['head_camera']['rgb']
        try:
            # 1. 提取 (H, W, C)
            head_cam = raw_obs['observation']['head_camera']['rgb'] # (240, 320, 3)
            
            # 2. 维度变换 (H, W, C) -> (C, H, W)
            # numpy 使用 transpose(2, 0, 1)
            head_cam = np.moveaxis(head_cam, -1, 0) # 变成 (3, 240, 320)
            
            # 3. 存入扁平字典 (Key 必须和 Config 里的 shape_meta 对应)
            new_obs['head_cam'] = head_cam
            
        except KeyError:
            print(f"[Warning] 'head_cam' path not found in obs. Available keys: {raw_obs.keys()}")

        # --- B. 处理状态 (agent_pos) ---
        # 提取四个部分
        # 注意：RoboTwin/Aloha 的标准顺序通常是 [左臂, 左爪, 右臂, 右爪]
        # 请确保这里的拼接顺序和你训练数据里的顺序一致！
        ja = raw_obs['joint_action']
        
        parts = [
            ja['left_arm'],     # (6,)
            ja['left_gripper'], # (1,)
            ja['right_arm'],    # (6,)
            ja['right_gripper'] # (1,)
        ]
        
        # 拼接成 (14,)
        parts = [np.atleast_1d(p) for p in parts]
        agent_pos = np.concatenate(parts, axis=-1).astype(np.float32)
        
        new_obs['agent_pos'] = agent_pos

        return new_obs
    

    def env_step(self, env, action):
        # 1. 执行动作 (内部会自动增加 take_action_cnt)
        env.take_action(action)
        
        # 2. 获取原始观测 (后续由 preprocess_obs 处理)
        next_raw_obs = env.get_obs()
        
        # 3. 结算状态
        # 既然没有 shaping reward，逻辑就非常简单：成功即 1.0，否则 0.0
        is_success = env.check_success()
        step_lim = env.step_lim if env.step_lim is not None else float('inf')
        is_timeout = env.take_action_cnt >= step_lim
        #is_timeout = env.take_action_cnt >= env.step_lim
        
        reward = 1.0 if is_success else 0.0
        done = is_success or is_timeout
        
        # 4. 返回
        return next_raw_obs, reward, done, {'success': is_success, 'timeout': is_timeout}


    def run(self):
        cfg = self.cfg
        
        # ---------------------------------------------------------
        # 1. 环境初始化 (Heavy Lift)
        # ---------------------------------------------------------
        print("Creating Environment for Rollout (using RoboTwin logic)...")
        
        # 加载所有复杂的配置参数
        env_args = self._load_env_args()
        
        # 实例化环境类
        env = self._init_task_env(cfg.task.name)

        print("Loading dataset to fit normalizer...")
        # 这里的 cfg.task.dataset 对应 robot_fpo_14.yaml 里的配置
        # 它会自动读取 zarr_path 里的数据
        dataset = hydra.utils.instantiate(cfg.task.dataset)
        
        # 从数据集中获取统计好的 normalizer
        normalizer = dataset.get_normalizer()
        
        # 将 normalizer 的参数加载到 Policy 中
        print(f"Loading normalizer into policy...")
        self.model.set_normalizer(normalizer)
        print("Normalizer loaded successfully.")
        
        # ---------------------------------------------------------
        # 2. Rollout Buffer 初始化
        # ---------------------------------------------------------
        # 为了获取 shape，我们需要先 Setup 一次
        print("Initializing Env to get shapes...")
        # 注意：训练模式 is_test=False (启用随机化)，Setup 会加载机器人
        env.setup_demo(now_ep_num=0, seed=cfg.training.seed, is_test=False, **env_args)
        
        # 获取第一帧 Obs
        raw_obs = env.get_obs() # RoboTwin 的 env.reset() 通常在 setup_demo 内部或之后调用 get_obs

        obs = self.preprocess_obs(raw_obs)

        #action_shape = tuple(cfg.policy.shape_meta.action.shape)
        action_dim_tuple = tuple(cfg.policy.shape_meta.action.shape) # D=14
        horizon = cfg.policy.horizon # 从配置中读取 H=8
        
        # 构造 RolloutBuffer 期望的动作形状: (Horizon, Dimension) -> (8, 14)
        action_shape = (horizon,) + action_dim_tuple
        
        obs_shape = {k: v.shape for k, v in obs.items()}
        
        print(f"Obs Shape: {obs_shape}, Action Shape for Buffer: {action_shape}")
        
        rollout_steps = cfg.training.get('rollout_steps', 2048)
        num_envs = 1 
        num_probes = 50 # 与 FlowFPOPolicy 默认值一致
        
        buffer = RolloutBuffer(
            num_envs=num_envs,
            num_probes=num_probes,
            buffer_size=rollout_steps,
            obs_shape=obs_shape,
            action_shape=action_shape,
            device=self.device,
            horizon=horizon
        )

        
        
        # ---------------------------------------------------------
        # 3. 训练准备
        # ---------------------------------------------------------
        self.model.to(self.device)
        if self.ema_model:
            self.ema_model.to(self.device)
            
        checkpoint_manager = TopKCheckpointManager(
            save_dir=os.path.join(self.output_dir, 'checkpoints'),
            monitor_key='train/reward', 
            mode='max',
            k=cfg.checkpoint.topk.k,
            format_str='epoch={epoch:03d}-reward={train_reward:.3f}.ckpt'
        )

        if cfg.logging.mode != 'disabled':
            wandb.init(
                dir=self.output_dir,
                config=OmegaConf.to_container(cfg, resolve=True),
                project=cfg.logging.project,
                name=cfg.logging.name,
                resume=True
            )

        # 超参数
        ppo_epochs = cfg.training.get('ppo_epochs', 4)
        mini_batch_size = cfg.training.get('batch_size', 64)
        clip_epsilon = cfg.training.get('clip_epsilon', 0.2)
        critic_coef = cfg.training.get('critic_coef', 0.5)
        max_iterations = cfg.training.get('max_iterations', 1000)

        print(f"Start FPO Training Loop for {max_iterations} iterations...")
        
        # ---------------------------------------------------------
        # 4. 主循环
        # ---------------------------------------------------------
        for iteration in range(max_iterations):
            self.model.eval()
            buffer.clear()
            
            # === Phase 1: Data Collection (Rollout) ===
            print(f"Iteration {iteration}: Collecting Rollouts...")
            
            # Reset Env via setup_demo (RoboTwin pattern)
            # 使用 iteration 作为 seed 偏移，保证每次 rollout 环境随机性不同
            current_seed = cfg.training.seed + iteration
            env.setup_demo(now_ep_num=0, seed=current_seed, is_test=False, **env_args)
            
            raw_obs = env.get_obs()
            obs = self.preprocess_obs(raw_obs)
            
            total_reward = 0
            
            # 我们需要手动管理步数，因为 setup_demo 可能会 reset 整个 task
            # 这里的逻辑是：在同一个 task 实例中跑 rollout_steps 步
            # 如果中途 success/fail，需要重新 setup_demo
            
            pbar = tqdm(range(rollout_steps), desc="Rollout")
            steps_collected = 0
            
            while steps_collected < rollout_steps:
                # 处理 Observation
                # obs_tensor = dict_apply(obs, lambda x: torch.from_numpy(x).unsqueeze(0).to(self.device))
                obs_tensor = dict_apply(obs, lambda x: torch.from_numpy(x).unsqueeze(0).unsqueeze(0).to(self.device))
                
                # 采样动作
                with torch.no_grad():
                    action_full, info = self.model.sample_action_with_info(obs_tensor)
                    value = self.model.predict_value(obs_tensor)
                action_to_buffer = action_full[:, 0, :]
                # 执行动作
                action_np = action_to_buffer.squeeze(0).cpu().numpy()
                if len(action_np.shape) > 1 and action_np.shape[0] > 1:
                    env_action = action_np[0] 
                else:
                    env_action = action_np


                try:
                    raw_obs_next, reward, done, info_env = self.env_step(env, env_action)
                except AttributeError:
                    raise RuntimeError("Env missing step() method. Please check RoboTwin env interface.")
                obs_next = self.preprocess_obs(raw_obs_next)
                #if not isinstance(obs_next, dict): obs_next = {'image': obs_next}
                
                # RoboTwin 的 reward 通常是稀疏的 (成功 1，失败 0)，或者是 dense reward
                # done 通常由 max_steps 或 success 决定
                
                total_reward += reward
                
                reward_tensor = torch.tensor([reward], device=self.device, dtype=torch.float32)
                done_tensor = torch.tensor([done], device=self.device, dtype=torch.float32)
                obs_to_buffer = dict_apply(obs_tensor, lambda x: x.squeeze(1)) # 去掉 Time 维度
                
                buffer.add(
                    obs=obs_to_buffer, 
                    action=action_full, 
                    reward=reward_tensor,
                    done=done_tensor,
                    value=value,    
                    info=info      
                )
                
                obs = obs_next
                steps_collected += 1
                pbar.update(1)
                
                if done:
                    # 如果 Done 了，重置环境 (Setup Demo)
                    current_seed += 1 # 换个种子
                    env.setup_demo(now_ep_num=0, seed=current_seed, is_test=False, **env_args)
                    raw_obs = env.get_obs()
                    obs = self.preprocess_obs(raw_obs)
                    #obs = env.get_obs()
                    #if not isinstance(obs, dict): obs = {'image': obs}
            
            pbar.close()
            
            # === Phase 2: Compute GAE ===
            with torch.no_grad():
                # last_obs_tensor = dict_apply(obs, lambda x: torch.from_numpy(x).unsqueeze(0).to(self.device))
                # add time dim
                last_obs_tensor = dict_apply(obs, lambda x: torch.from_numpy(x).unsqueeze(0).unsqueeze(0).to(self.device))
                next_value = self.model.predict_value(last_obs_tensor)
                buffer.compute_gae(next_value)

            # === Phase 3: Update ===
            self.model.train()
            train_losses = []
            policy_losses = []
            value_losses = []
            ratios = []
            
            print(f"Iteration {iteration}: Updating Policy...")
            for _ in range(ppo_epochs):
                for batch in buffer.get_generator(mini_batch_size):
                    B = batch['action'].shape[0]
                    N = batch['noise'].shape[1]
                    
                    obs_rep = dict_apply(batch['obs'], lambda x: x.repeat_interleave(N, dim=0))
                    action_rep = batch['action'].repeat_interleave(N, dim=0)
                    noise_flat = batch['noise'].reshape(-1, *batch['noise'].shape[2:])
                    time_flat = batch['timesteps'].reshape(-1)
                    
                    new_loss_flat = self.model.compute_cfm_loss(
                        {'obs': obs_rep, 'action': action_rep},
                        noise=noise_flat,
                        timesteps=time_flat,
                        reduction='none'
                    ) 
                    
                    new_loss = new_loss_flat.view(B, N)
                    old_loss = batch['old_loss']
                    
                    loss_diff = old_loss.mean(dim=1) - new_loss.mean(dim=1)
                    ratio = torch.exp(loss_diff)
                    
                    adv = batch['advantage']
                    adv = (adv - adv.mean()) / (adv.std() + 1e-8)
                    
                    surr1 = ratio * adv
                    surr2 = torch.clamp(ratio, 1.0 - clip_epsilon, 1.0 + clip_epsilon) * adv
                    policy_loss = -torch.min(surr1, surr2).mean()
                    
                    current_value = self.model.predict_value(batch['obs']).squeeze(-1)
                    value_loss = F.mse_loss(current_value, batch['return'])
                    
                    total_loss = policy_loss + critic_coef * value_loss
                    
                    self.optimizer.zero_grad()
                    total_loss.backward()
                    nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                    self.optimizer.step()
                    
                    train_losses.append(total_loss.item())
                    policy_losses.append(policy_loss.item())
                    value_losses.append(value_loss.item())
                    ratios.append(ratio.mean().item())

            # === Phase 4: Log ===
            mean_loss = np.mean(train_losses)
            log_data = {
                "train/loss": mean_loss,
                "train/policy_loss": np.mean(policy_losses),
                "train/value_loss": np.mean(value_losses),
                "train/ratio": np.mean(ratios),
                "train/reward": total_reward,
                "train/epoch": iteration
            }
            if cfg.logging.mode != 'disabled':
                wandb.log(log_data, step=self.global_step)
            print(f"Iter {iteration} | Reward: {total_reward:.2f} | Loss: {mean_loss:.4f} | Ratio: {np.mean(ratios):.3f}")
            
            self.global_step += 1
            
            if iteration % cfg.checkpoint.save_interval == 0:
                 checkpoint_manager.save_checkpoint(
                    epoch=iteration,
                    train_reward=total_reward,
                    model=self.model,
                    optimizer=self.optimizer
                )