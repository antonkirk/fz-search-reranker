#!/bin/bash

#SBATCH --job-name=train_model
#SBATCH --output=train_model_result-%J.out
#SBATCH --nodes=1
#SBATCH --time=02:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu
#SBATCH --mail-user=
#SBATCH --mail-type=END,FAIL
#SBATCH --partition=titans
#SBATCH --export=ALL

# Load any necessary modules or activate virtual environment
conda activate fz-search-reranker
module load CUDA/12.1

# Change to the directory where your training script is located
# cd /fz_search_reranker/

# Check CUDA version
nvidia-smi

# Run your training script
make data_local

# Deactivate virtual environment if activated
# deactivate
