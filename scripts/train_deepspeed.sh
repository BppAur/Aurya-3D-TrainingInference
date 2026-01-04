#!/bin/bash

# DeepSpeed multi-node training launcher
# Usage: bash scripts/train_deepspeed.sh <node_num> <node_rank> <num_gpu_per_node> <master_ip> <config> <output_dir>

node_num=$1
node_rank=$2
num_gpu_per_node=$3
master_ip=$4
config=$5
output_dir=$6

# Set distributed training environment variables
export MASTER_ADDR=${master_ip:-"localhost"}
export MASTER_PORT=${MASTER_PORT:-29500}
export NODE_RANK=$node_rank
export WORLD_SIZE=$((node_num * num_gpu_per_node))

echo "========================================"
echo "Training Configuration:"
echo "Node Number: $node_num"
echo "Node Rank: $node_rank"
echo "GPUs per Node: $num_gpu_per_node"
echo "Master Address: $MASTER_ADDR"
echo "Master Port: $MASTER_PORT"
echo "World Size: $WORLD_SIZE"
echo "Config: $config"
echo "Output Dir: $output_dir"
echo "========================================"

# Create output directory
mkdir -p $output_dir

# Launch training with DeepSpeed
deepspeed --num_nodes=$node_num \
          --num_gpus=$num_gpu_per_node \
          --master_addr=$MASTER_ADDR \
          --master_port=$MASTER_PORT \
          --node_rank=$node_rank \
          main.py \
          --config $config \
          --output_dir $output_dir
