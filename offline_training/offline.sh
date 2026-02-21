#!/bin/bash
#SBATCH --job-name=ez_dynamics
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --mail-type=begin
#SBATCH --mail-type=end
#SBATCH --mail-user=sk3686@princeton.edu
#SBATCH --gres=gpu:1
#SBATCH --constraint=gpu80

# Set environment variables
export OMP_NUM_THREADS=1
export CUDA_VISIBLE_DEVICES=0
export HYDRA_FULL_ERROR=1

# Initialize your personal Miniconda
source ~/miniconda3/bin/activate

# Activate your environment
conda activate ezv2

# Usage: sbatch offline.sh <game> <model_type>
#   game:       pong | asterix
#   model_type: baseline | stu | mamba | attention | residual_only
#               | temporal_stu | temporal_mamba | temporal_attention
#
# Examples:
#   sbatch offline.sh pong baseline
#   sbatch offline.sh asterix stu
#   sbatch offline.sh pong temporal_stu

GAME=${1:-pong}
MODEL_TYPE=${2:-baseline}

echo "Running game=${GAME} model_type=${MODEL_TYPE}"

cd /home/sk3686/hazan/STUZero/offline_training

COMMON_ARGS="--game ${GAME} \
    --epochs 80 \
    --batch_size 256 \
    --lr 1e-3 \
    --rollout_len 50 \
    --use_wandb \
    --wandb_run_name ${GAME}_${MODEL_TYPE}"

case ${MODEL_TYPE} in
    baseline)
        python train_dynamics_offline.py ${COMMON_ARGS} \
            --model_type baseline \
            --num_blocks 2
        ;;
    stu)
        python train_dynamics_offline.py ${COMMON_ARGS} \
            --model_type stu \
            --num_blocks 1 \
            --dynamics_stu_num_filters 2 \
            --dynamics_stu_seq_len 36
        ;;
    mamba)
        python train_dynamics_offline.py ${COMMON_ARGS} \
            --model_type mamba \
            --num_blocks 1 \
            --mamba_d_state 16 \
            --mamba_d_conv 4 \
            --mamba_expand 1
        ;;
    attention)
        python train_dynamics_offline.py ${COMMON_ARGS} \
            --model_type attention \
            --num_blocks 1 \
            --attn_num_heads 4
        ;;
    residual_only)
        python train_dynamics_offline.py ${COMMON_ARGS} \
            --model_type residual_only \
            --num_blocks 1
        ;;
    temporal_baseline)
        python train_dynamics_offline.py ${COMMON_ARGS} \
            --model_type temporal_baseline \
            --num_blocks 2 \
            --buffer_size 10
        ;;
    spatiotemporal_stu)
        python train_dynamics_offline.py ${COMMON_ARGS} \
            --model_type spatiotemporal_stu \
            --num_blocks 1 \
            --buffer_size 10 \
            --dynamics_stu_num_filters 2 \
            --dynamics_stu_seq_len 36
        ;;
    temporal_stu)
        python train_dynamics_offline.py ${COMMON_ARGS} \
            --model_type temporal_stu \
            --num_blocks 1 \
            --buffer_size 10 \
            --dynamics_stu_num_filters 2
        ;;
    temporal_mamba)
        python train_dynamics_offline.py ${COMMON_ARGS} \
            --model_type temporal_mamba \
            --num_blocks 1 \
            --buffer_size 10 \
            --mamba_d_state 16 \
            --mamba_d_conv 4 \
            --mamba_expand 1
        ;;
    temporal_attention)
        python train_dynamics_offline.py ${COMMON_ARGS} \
            --model_type temporal_attention \
            --num_blocks 1 \
            --buffer_size 10 \
            --attn_num_heads 4
        ;;
    *)
        echo "Unknown model_type: ${MODEL_TYPE}"
        echo "Options: baseline | stu | mamba | attention | residual_only | temporal_baseline | spatiotemporal_stu | temporal_stu | temporal_mamba | temporal_attention"
        exit 1
        ;;
esac
