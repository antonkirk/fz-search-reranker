import pandas as pd
import matplotlib.pyplot as plt
import glob
import os
import hydra
from omegaconf import DictConfig

def plot_eval(csv_path):
    data = pd.read_csv(csv_path)
    
    # Extract metric categories
    categories = ['hit', 'precision', 'recall', 'mrr']
    
    plt.figure(figsize=(14, 8))
    
    for category in categories:
        subset = data[data['metric'].str.startswith(category)]
        plt.plot(subset['metric'], subset['faiss'], marker='o', label=f'{category} faiss')
        plt.plot(subset['metric'], subset['hybrid'], marker='x', label=f'{category} hybrid')
    
    plt.xlabel('Metrics')
    plt.ylabel('Values')
    plt.title(f'Metrics Comparison for {os.path.basename(csv_path)}')
    plt.legend()
    plt.xticks(rotation=90)
    plt.grid(True)
    plt.tight_layout()

    # Save the plot
    output_path = os.path.join("plots", f"{os.path.basename(csv_path).replace('.csv', '.png')}")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path)
    plt.close()

@hydra.main(config_path="../configs", config_name="config", version_base=None)
def visualize(cfg: DictConfig):
    csv_files = glob.glob(os.path.join(cfg.paths.eval_dir, "*.csv"))
    
    for csv_file in csv_files:
        plot_eval(csv_file)

if __name__ == "__main__":
    visualize()