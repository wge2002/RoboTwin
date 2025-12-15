"""
FPO Training Launcher (RL Mode)
Usage:
    python train_fpo.py --config-name=robot_fpo_14
"""

import sys
import os
import pathlib
import yaml
import hydra
from omegaconf import OmegaConf

# 使用行缓冲，确保日志实时输出到文件
sys.stdout = open(sys.stdout.fileno(), mode='w', buffering=1)
sys.stderr = open(sys.stderr.fileno(), mode='w', buffering=1)

# 导入我们的 RL Workspace
# 注意：确保 train_fpo_workspace.py 已经放在 diffusion_policy/workspace/ 下
from diffusion_policy.workspace.train_fpo_workspace import TrainFPOWorkspace

# 获取当前文件路径，用于定位 task_config
current_file_path = os.path.abspath(__file__)
parent_directory = os.path.dirname(current_file_path)


def get_camera_config(camera_type):
    # RoboTwin 的相机配置文件在 ../../task_config/_camera_config.yml
    camera_config_path = os.path.join(parent_directory, "../../task_config/_camera_config.yml")

    assert os.path.isfile(camera_config_path), f"task config file is missing at {camera_config_path}"

    with open(camera_config_path, "r", encoding="utf-8") as f:
        args = yaml.load(f.read(), Loader=yaml.FullLoader)

    assert camera_type in args, f"camera {camera_type} is not defined"
    return args[camera_type]

# 注册 eval 解析器，允许在 yaml 中执行 python 代码
OmegaConf.register_new_resolver("eval", eval, replace=True)

@hydra.main(
    version_base=None,
    config_path=str(pathlib.Path(__file__).parent.joinpath("diffusion_policy", "config")),
)
def main(cfg: OmegaConf):
    try:
        # 尝试获取 head_camera_type (通常由 bash 脚本传入)
        if "head_camera_type" in cfg:
            head_camera_type = cfg.head_camera_type
            head_camera_cfg = get_camera_config(head_camera_type)
            
            # 显式更新 task 中的图像形状
            # [Channel, Height, Width]
            new_shape = [3, head_camera_cfg["h"], head_camera_cfg["w"]]
            cfg.task.image_shape = new_shape
            
            # 更新 shape_meta (Policy 初始化主要看这个)
            if "shape_meta" in cfg.task and "obs" in cfg.task.shape_meta:
                if "head_cam" in cfg.task.shape_meta.obs:
                    cfg.task.shape_meta.obs.head_cam.shape = new_shape
                    
            print(f"[INFO] Updated camera config for {head_camera_type}: {new_shape}")
    except Exception as e:
        print(f"[WARNING] Failed to update camera config dynamically: {e}")
        print("[WARNING] Using default config from yaml.")

    # 解析所有引用 (Resolving ${...})
    OmegaConf.resolve(cfg)
    
    print(f"[INFO] Initializing Workspace: TrainFPOWorkspace")
    workspace = TrainFPOWorkspace(cfg)
    
    print(f"[INFO] Starting Training Loop...")
    workspace.run()

if __name__ == "__main__":
    main()