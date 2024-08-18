#!/bin/bash

#SBATCH --job-name=generate_data_openai_api
#SBATCH --output=generate_data_openai_api_result-%J.out
#SBATCH --nodes=1
#SBATCH --time=12:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mail-user=s184191@dtu.dk
#SBATCH --mail-type=END,FAIL
#SBATCH --partition=cyclopes
#SBATCH --export=ALL

# Load any necessary modules or activate virtual environment
conda activate fz_search_reranker

# Run your training script
python fz_search_reranker/data/api_request_parallel_processor.py \
  --requests_filepath /scratch/s184191/data/api_requests.jsonl \
  --save_filepath /scratch/s184191/data/api_requests_results.jsonl \
  --request_url https://api.openai.com/v1/chat/completions \
  --max_requests_per_minute 3000 \
  --max_tokens_per_minute 1000000
