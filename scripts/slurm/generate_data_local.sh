#!/bin/bash

#SBATCH --job-name=generate_data_%J
#SBATCH --output=generate_data_result-%J.out
#SBATCH --nodes=1
#SBATCH --time=16:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:Ampere:1
#SBATCH --mem=64G
#SBATCH --mail-type=END,FAIL
#SBATCH --partition=titans
#SBATCH --export=ALL
#SBATCH --nodelist=comp-gpu14

# Load any necessary modules or activate virtual environment
conda activate fz_search_reranker
module load CUDA/12.1

# Change to the directory where your training script is located
# cd /fz_search_reranker/

# Check CUDA version
nvidia-smi

# Run data generation script
python fz_search_reranker/data/make_dataset_local_model.py +experiment=merged_model_generate_data sys=titan

# Deactivate virtual environment if activated
# deactivate
