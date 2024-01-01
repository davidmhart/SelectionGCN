#!/bin/bash


#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --ntasks-per-node=1
#SBATCH --output="./slurm_out/slurm-%j.out"
#SBATCH --error="./slurm_out/slurm-%j-error.out"
#SBATCH --cpus-per-task=6
#SBATCH --mem=16G
#SBATCH --time=4000

DATASET=$1
NETWORK=$2
SPATIAL=$3
USE_EDGE_WEIGHTS=$4
SEL_FUNCTION=$5
SEL_COUNT=$6

# good idea to make sure environment is set up
source ~/.bashrc
# or conda activate conda_env_name if your using conda

# run the training script
python train_traffic_network.py $DATASET $NETWORK \
    --num_spatial $SPATIAL\
    --use_edge_weights $USE_EDGE_WEIGHTS \
    --selection_function $SEL_FUNCTION \
    --selection_count $SEL_COUNT \
    --schedule epochs30 \
    --num_workers=6
