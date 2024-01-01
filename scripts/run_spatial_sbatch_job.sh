#!/bin/bash

#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --ntasks-per-node=1
#SBATCH --output="./slurm_out/slurm-%j.out"
#SBATCH --error="./slurm_out/slurm-%j-error.out"
#SBATCH --cpus-per-task=6
#SBATCH --mem=16G
#SBATCH --time=1440

DATASET=$1
NETWORK=$2
FEATURES=$3
SELECTION=$4
COUNT=$5

# good idea to make sure environment is set up
source ~/.bashrc
# or conda activate conda_env_name if your using conda

# run the training script
srun python train_spatial_network.py $DATASET $NETWORK $FEATURES \
    --selection_function $SELECTION \
    --selection_count $COUNT \
    --batch_size 16 --max_epochs 60 --num_workers=6

