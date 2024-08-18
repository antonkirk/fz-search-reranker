#!/bin/bash

#SBATCH --job-name=eval_model
#SBATCH --output=eval_model_result-%J.out
#SBATCH --nodes=1
#SBATCH --time=03:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --mail-type=END,FAIL
#SBATCH --partition=titans
#SBATCH --export=ALL

# Load any necessary modules or activate virtual environment
conda activate fz_search_reranker
module load CUDA/12.1

# Change to the directory where your training script is located
# cd /fz_search_reranker/

# Check CUDA version
nvidia-smi

# Run your training script
python fz_search_reranker/evaluate_model.py \
    --multirun \
    sys=titan \
    model=mpnet_base \
    eval/dataset=findzebra,ada_dx \

# Deactivate virtual environment if activated
# deactivate
