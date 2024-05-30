#!/bin/bash

#SBATCH --job-name=train_model
#SBATCH --output=train_model_result-%J.out
#SBATCH --nodes=1
#SBATCH --time=00:00:10
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu
#SBATCH --mail-user=
#SBATCH --mail-type=END,FAIL
#SBATCH --partition=titans
#SBATCH --export=ALL

# Load any necessary modules or activate virtual environment
# module load <module_name>
# source <virtual_env_path>/bin/activate

# Change to the directory where your training script is located
cd /fz_search_reranker/

# Run your training script
make train_titan

# Deactivate virtual environment if activated
# deactivate
