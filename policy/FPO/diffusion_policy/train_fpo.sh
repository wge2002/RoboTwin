#!/bin/bash

# ================= 配置区域 =================
task_name=stack_blocks_two
task_config=demo_clean
expert_data_num=50
seed=0
action_dim=14
gpu_id=${1}      # 从命令行参数获取 GPU ID
model_name=fpo   # 必须是 fpo
head_camera_type=D435

# RL 特有参数 (可以在这里调整)
max_iterations=1000
rollout_steps=2048
ppo_epochs=4

DEBUG=False
save_ckpt=True

# 构造 Config 名称: robot_fpo_14
alg_name=robot_${model_name}_${action_dim}
config_name=${alg_name}

# 实验名称 (加上 RL 标识)
addition_info=train_rl
exp_name=${task_name}-robot_${model_name}-${addition_info}
run_dir="data/outputs/${exp_name}_seed${seed}"

echo -e "\033[33mgpu id (to use): ${gpu_id}\033[0m"

# ================= 环境设置 =================
if [ "$DEBUG" = "True" ]; then
    wandb_mode=offline
    echo -e "\033[33mDebug mode!\033[0m"
else
    wandb_mode=online
    echo -e "\033[33mTrain mode\033[0m"
fi

export HYDRA_FULL_ERROR=1 
export CUDA_VISIBLE_DEVICES=${gpu_id}

# ================= 数据检查 =================
# 注意：RL 仍然需要这份数据来初始化 Normalizer！
if [ ! -d "./data/${task_name}-${task_config}-${expert_data_num}.zarr" ]; then
    echo "Processing data for Normalizer initialization..."
    bash process_data.sh ${task_name} ${task_config} ${expert_data_num}
fi

# ================= 启动 FPO 训练 =================
# 关键修改：调用 train_fpo.py
python train_fpo.py --config-name=${config_name}.yaml \
                            task.name=${task_name} \
                            task.model_name=${model_name} \
                            task.dataset.zarr_path="data/${task_name}-${task_config}-${expert_data_num}.zarr" \
                            training.debug=$DEBUG \
                            training.seed=${seed} \
                            training.device="cuda:0" \
                            exp_name=${exp_name} \
                            logging.mode=${wandb_mode} \
                            setting=${task_config} \
                            expert_data_num=${expert_data_num} \
                            head_camera_type=$head_camera_type \
                            \
                            training.max_iterations=${max_iterations} \
                            training.rollout_steps=${rollout_steps} \
                            training.ppo_epochs=${ppo_epochs}