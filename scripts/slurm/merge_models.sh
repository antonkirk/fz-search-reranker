#!/bin/bash

#SBATCH --job-name=merge_models
#SBATCH --output=merge_models_result-%J.out
#SBATCH --nodes=1
#SBATCH --time=12:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=64G
#SBATCH --mail-user=s184191@dtu.dk
#SBATCH --mail-type=END,FAIL
#SBATCH --partition=cyclopes
#SBATCH --export=ALL

# Load any necessary modules or activate virtual environment
conda activate fz_search_reranker
# module load CUDA/12.1

# Change to the directory where your training script is located
# cd /fz_search_reranker/

# Check CUDA version
# nvidia-smi

# Run your training script
mergekit-yaml --transformers-cache '/scratch/s184191/cache/' --lazy-unpickle 'fz_search_reranker/configs/merge/merge_70_40.yaml' '/scratch/s184191/models/llama3-openbiollm-8b-dare-ties-70-40' 

# Deactivate virtual environment if activated
# deactivate
