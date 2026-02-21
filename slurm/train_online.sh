#!/bin/bash
#SBATCH --job-name=ez_online
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=30:00:00
#SBATCH --mail-type=begin
#SBATCH --mail-type=end
#SBATCH --mail-user=sk3686@princeton.edu
#SBATCH --gres=gpu:1
#SBATCH --constraint=gpu80

# Set environment variables
export OMP_NUM_THREADS=1
export CUDA_VISIBLE_DEVICES=0
export HYDRA_FULL_ERROR=1

# Initialize conda
source ~/miniconda3/bin/activate
conda activate ezv2

# Usage: sbatch train_online.sh <job_name>
#   job_name: asterix_baseline | pong_rollout10
#   Example: sbatch train_online.sh asterix_baseline

JOB_NAME=${1:-asterix_baseline}
echo "Running job: ${JOB_NAME}"

case ${JOB_NAME} in
    asterix_baseline)
        python ez/train.py exp_config=ez/config/exp/atari.yaml \
            wandb.tag=Asterix
        ;;
    pong_rollout10)
        python ez/train.py exp_config=ez/config/exp/atari2.yaml \
            wandb.tag=rolloutlen10
        ;;
    *)
        echo "Unknown job: ${JOB_NAME}"
        echo "Options: asterix_baseline | pong_rollout10"
        exit 1
        ;;
esac
