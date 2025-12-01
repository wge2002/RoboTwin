#!/bin/bash

# == keep unchanged ==
# policy_name=DP_DIT
task_name=stack_blocks_two
task_config=demo_clean
ckpt_setting=demo_clean
expert_data_num=50
seed=0
gpu_id=${1}
model_name=Flow
DEBUG=False

export CUDA_VISIBLE_DEVICES=${gpu_id}
echo -e "\033[33mgpu id (to use): ${gpu_id}\033[0m"

cd ../..

PYTHONWARNINGS=ignore::UserWarning \
python script/eval_policy.py --config policy/DP_DIT/deploy_policy.yml \
    --overrides \
    --task_name ${task_name} \
    --task_config ${task_config} \
    --ckpt_setting ${ckpt_setting} \
    --expert_data_num ${expert_data_num} \
    --seed ${seed} \
    --model_name ${model_name}