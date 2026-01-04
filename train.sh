export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export num_gpu_per_node=8

export node_num=1
export node_rank=${1:-0}
export master_ip=${MASTER_IP:-localhost}

# For single-node training, override to use available GPUs
if [ -z "$CUDA_VISIBLE_DEVICES" ]; then
    # Auto-detect available GPUs
    num_gpu_per_node=$(nvidia-smi --list-gpus | wc -l)
    export CUDA_VISIBLE_DEVICES=$(seq -s, 0 $((num_gpu_per_node - 1)))
fi

echo "Using GPUs: $CUDA_VISIBLE_DEVICES"

############## vae ##############
# export config=configs/train_vae_refine.yaml
# export output_dir=outputs/vae_ultrashape/exp1_token8192
# bash scripts/train_deepspeed.sh $node_num $node_rank $num_gpu_per_node $master_ip $config $output_dir

############## dit ##############
export config=configs/train_dit_refine.yaml
export output_dir=outputs/dit_ultrashape/exp1_token8192
bash scripts/train_deepspeed.sh $node_num $node_rank $num_gpu_per_node $master_ip $config $output_dir
