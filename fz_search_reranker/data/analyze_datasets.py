from datasets import load_dataset
import hydra

def calculate_vocabulary_size(ds):
    # Calculate the number of unique words in each dataset
    vocabulary = set()
    for example in ds:
        for query in example["queries"]:
            vocabulary.update(query.split())

    return len(vocabulary)

def calculate_size(ds):
    # Calculate the number of words in each dataset
    ds_size = sum(len(query.split()) for example in ds for query in example["queries"])

    return ds_size

@hydra.main(version_base=None, config_path="../configs", config_name="config")
def analyze_datasets(cfg):
    # Load the datasets
    dataset_gpt = load_dataset(
        "json", field="data",
        data_files=f"{cfg.paths.data_dir}/gpt-4o-mini_synthetic_dataset.jsonl"
    )["train"]
    dataset_merged = load_dataset(
        "json", field="data",
        data_files=f"{cfg.paths.data_dir}/llama3-openbiollm-8b-dare-ties-70-40_synthetic_dataset.jsonl"
    )["train"]
    dataset_llama3 = load_dataset(
        "json", field="data",
        data_files=f"{cfg.paths.data_dir}/Meta-Llama-3-8B-Instruct_synthetic_dataset.jsonl"
    )["train"]

    gpt_ds_size = calculate_size(dataset_gpt)
    merged_ds_size = calculate_size(dataset_merged)
    llama3_ds_size = calculate_size(dataset_llama3)

    print(f"Size of gpt-4o-mini ds: {gpt_ds_size}")
    print(f"Size of merged ds: {merged_ds_size}")
    print(f"Size of llama3 ds: {llama3_ds_size}")

    gpt_vocab_size = calculate_vocabulary_size(dataset_gpt)
    merged_vocab_size = calculate_vocabulary_size(dataset_merged)
    llama3_vocab_size = calculate_vocabulary_size(dataset_llama3)

    print(f"Vocabulary size of gpt-4o-mini ds: {gpt_vocab_size}")
    print(f"Vocabulary size of merged ds: {merged_vocab_size}")
    print(f"Vocabulary size of llama3 ds: {llama3_vocab_size}")

    # Calculate the average length of the queries in each dataset
    no_of_queries_gpt = len([query for example in dataset_gpt for query in example["queries"]])
    no_of_queries_merged = len([query for example in dataset_merged for query in example["queries"]])
    no_of_queries_llama3 = len([query for example in dataset_llama3 for query in example["queries"]])

    print(f"Number of queries in gpt-4o-mini ds: {no_of_queries_gpt}")
    print(f"Number of queries in merged ds: {no_of_queries_merged}")
    print(f"Number of queries in llama3 ds: {no_of_queries_llama3}")

    gpt_avg_len = gpt_ds_size / no_of_queries_gpt
    merged_avg_len = merged_ds_size / no_of_queries_merged
    llama3_avg_len = llama3_ds_size / no_of_queries_llama3

    print(f"Average length of queries in gpt-4o-mini ds: {gpt_avg_len}")
    print(f"Average length of queries in merged ds: {merged_avg_len}")
    print(f"Average length of queries in llama3 ds: {llama3_avg_len}")

    # Type-Token Ratio (TTR): Calculate the ratio of unique words (types) to the total number of words (tokens) in each dataset.
    gpt_ttr = gpt_vocab_size / gpt_ds_size
    merged_ttr = merged_vocab_size / merged_ds_size
    llama3_ttr = llama3_vocab_size / llama3_ds_size

    print(f"TTR of gpt-4o-mini ds: {gpt_ttr}")
    print(f"TTR of merged ds: {merged_ttr}")
    print(f"TTR of llama3 ds: {llama3_ttr}")

    # Jaccard Similarity: Measure the overlap of words or phrases between datasets using the Jaccard similarity coefficient.
    gpt_word_set = set(word for example in dataset_gpt for query in example["queries"] for word in query.split())
    merged_word_set = set(word for example in dataset_merged for query in example["queries"] for word in query.split())
    llama3_word_set = set(word for example in dataset_llama3 for query in example["queries"] for word in query.split())

    # Print 10 words from each dataset
    print("10 words from gpt-4o-mini dataset:")
    print(list(gpt_word_set)[:10])
    print("10 words from merged dataset:")
    print(list(merged_word_set)[:10])
    print("10 words from llama3 dataset:")
    print(list(llama3_word_set)[:10])

    gpt_merged_jaccard = len(gpt_word_set.intersection(merged_word_set)) / len(gpt_word_set.union(merged_word_set))
    gpt_llama3_jaccard = len(gpt_word_set.intersection(llama3_word_set)) / len(gpt_word_set.union(llama3_word_set))
    merged_llama3_jaccard = len(merged_word_set.intersection(llama3_word_set)) / len(merged_word_set.union(llama3_word_set))
                                                                                     
    print(f"Jaccard similarity between gpt-4o-mini and merged ds: {gpt_merged_jaccard}")
    print(f"Jaccard similarity between gpt-4o-mini and llama3 ds: {gpt_llama3_jaccard}")
    print(f"Jaccard similarity between merged and llama3 ds: {merged_llama3_jaccard}")


if __name__=='__main__':
    analyze_datasets()
    
    
    # dataset = load_dataset('findzebra/corpus')['train']
    # print(dataset)
    
    # # Find the average number of examples per cui in the dataset
    # cui_counts = {}
    # for example in dataset:
    #     cui = example['cui']
    #     cui_counts[cui] = cui_counts.get(cui, 0) + 1
    # print(len(cui_counts))
    # print(sum(cui_counts.values()) / len(cui_counts))
