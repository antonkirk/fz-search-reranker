import hydra
from datasets import load_dataset
from omegaconf import DictConfig, OmegaConf
import json

@hydra.main(config_path="../configs", config_name="config", version_base=None)
def clean_dataset(cfg: DictConfig):
    print(OmegaConf.to_yaml(cfg))
    dataset = load_dataset("json", field="data", data_files=cfg.paths.dataset_unprocessed)
    chunks = dataset['train']['chunk']
    queries = dataset['train']['queries']
    
    queries_cleaned = [
        [q.replace('"', '') for q in qs[:4]]
        for qs in queries
    ]

    data = [{"queries": q, "chunk": c} for q, c in zip(queries_cleaned, chunks)]
    dataset = {"version": "0.0.1", "data": data}

    #print(f"data: {data}")
    # print(f"dataset: {dataset}")
    with open(cfg.paths.dataset, "w", encoding="utf-8") as f:
        json.dump(dataset, f, indent=4)

if __name__=='__main__':
    clean_dataset()
