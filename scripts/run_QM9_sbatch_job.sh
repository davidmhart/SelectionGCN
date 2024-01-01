#!/bin/bash

#SBATCH --nodes=1
#SBATCH --gpus=4
#SBATCH --ntasks-per-node=4
#SBATCH --output="./slurm_out/slurm-%j.out"
#SBATCH --error="./slurm_out/slurm-%j-error.out"
#SBATCH --cpus-per-task=6
#SBATCH --mem=64G
#SBATCH --time=4200

TARGET_NUM=$1
NETWORK=$2
SCHEDULE=$3
EMA=$4
SELECTION=$5
DIST_COUNT=$6
ANGLE_COUNT=$7


# good idea to make sure environment is set up
source ~/.bashrc
# or conda activate conda_env_name if your using conda

# run the training script
srun python train_QM9_network.py $TARGET_NUM $NETWORK \
    --schedule_type $SCHEDULE --use_ema $EMA\
    --selection_function $SELECTION \
    --selection_count $DIST_COUNT $ANGLE_COUNT \
    --batch_size 32 --num_workers=6 --devices 0 1 2 3

