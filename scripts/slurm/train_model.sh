#!/bin/bash

#SBATCH --job-name=train_model
#SBATCH --output=train_model_result-%J.out
#SBATCH --nodes=1
#SBATCH --time=16:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:Ampere:4
#SBATCH --partition=titans
#SBATCH --export=ALL

# Load any necessary modules or activate virtual environment
# conda activate fz_search_reranker
module load CUDA/12.1

# Change to the directory where your training script is located
# cd /fz_search_reranker/

# Check CUDA version
nvidia-smi

# Run your training script
# python fz_search_reranker/train_model.py dataset=llama3_openbiollm_synthetic
accelerate launch --multi_gpu fz_search_reranker/train_model.py dataset=combined_synthetic

# Deactivate virtual environment if activated
# deactivate
