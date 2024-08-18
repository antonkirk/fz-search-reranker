from datasets import load_dataset, concatenate_datasets
from omegaconf import DictConfig
import hydra
import json


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def combine_datasets(cfg: DictConfig):
    # Load the datasets
    dataset1 = load_dataset(
        "json", field="data",
        data_files=f"{cfg.paths.data_dir}/gpt-4o-mini_synthetic_dataset.jsonl"
    )["train"]
    dataset2 = load_dataset(
        "json", field="data",
        data_files=f"{cfg.paths.data_dir}/llama3-openbiollm-8b-dare-ties-70-40_synthetic_dataset.jsonl"
    )["train"]
    dataset3 = load_dataset(
        "json", field="data",
        data_files=f"{cfg.paths.data_dir}/Meta-Llama-3-8B-Instruct_synthetic_dataset.jsonl"
    )["train"]

    # Combine the datasets
    combined_dataset = concatenate_datasets([dataset1, dataset2, dataset3])
    queries = combined_dataset["queries"]
    chunks = combined_dataset["chunk"]
    data = [{"queries": q, "chunk": c} for q, c in zip(queries, chunks)]
    dataset = {"version": "0.0.1", "data": data}

    # print(f"data: {data}")
    # print(f"dataset: {dataset}")
    with open(f"{cfg.paths.data_dir}/combined_synthetic_dataset.jsonl", "w", encoding="utf-8") as f:
        json.dump(dataset, f, indent=4)

if __name__ == "__main__":
    combine_datasets()