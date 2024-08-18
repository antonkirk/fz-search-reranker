#!/bin/bash

#SBATCH --job-name=eval_whoosh
#SBATCH --output=eval_whoosh_result-%J.out
#SBATCH --nodes=1
#SBATCH --time=01:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=32G
#SBATCH --mail-type=END,FAIL
#SBATCH --partition=cyclopes
#SBATCH --export=ALL

# Load any necessary modules or activate virtual environment
conda activate fz_search_reranker

# Change to the directory where your training script is located
# cd /fz_search_reranker/

# Run your training script
python fz_search_reranker/evaluate_model.py \
    --multirun \
    sys=titan \
    model=mpnet_base \
    eval/dataset=findzebra,ada_dx \
    +experiment=eval_whoosh

# Deactivate virtual environment if activated
# deactivate
