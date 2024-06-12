#!/bin/bash
NUM_GPUS=2  # 使用的GPU数量
nnodes=1  # 使用的机器数量

# 设置MASTER_ADDR和MASTER_PORT
export MASTER_ADDR=localhost
export MASTER_PORT=12355

# 启动分布式训练
python -m torch.distributed.launch --nproc_per_node=$NUM_GPUS --nnodes=$nnodes liguanqun/案例大赛/code/分布式训练/bert-base-chinese/train.py
