#!/bin/bash
#SBATCH --job-name=ez_train
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --mail-type=end
#SBATCH --mail-user=sk3686@princeton.edu
#SBATCH --gres=gpu:1
#SBATCH --constraint="intel&gpu40"

# Check if config file argument is provided
if [ -z "$1" ]; then
    echo "ERROR: Config file not provided!"
    echo "Usage: sbatch run_atari_80.sh <config_file>"
    exit 1
fi

CONFIG_FILE=$1

if [ ! -f "$CONFIG_FILE" ]; then
    echo "ERROR: Config file '$CONFIG_FILE' does not exist!"
    exit 1
fi

# Set environment variables
export OMP_NUM_THREADS=1
export CUDA_VISIBLE_DEVICES=0
export HYDRA_FULL_ERROR=1

# Initialize your personal Miniconda
source ~/miniconda3/bin/activate

# Activate your environment
conda activate ezv2

# Run your training script
python ez/train.py exp_config=$CONFIG_FILE
