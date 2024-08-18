from collections import defaultdict
import hydra
from datasets import load_dataset, Dataset
from omegaconf import DictConfig, OmegaConf
from transformers import set_seed
from sentence_transformers import (
    SentenceTransformer,
    evaluation,
    losses,
    SentenceTransformerTrainer,
    util
)

from sentence_transformers.training_args import SentenceTransformerTrainingArguments, BatchSamplers
from sentence_transformers.similarity_functions import SimilarityFunction
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
import wandb
import pickle


@hydra.main(version_base=None, config_path="configs", config_name="config")
def train_model(cfg: DictConfig):
    # Initialize wandb
    wandb.init(project=cfg.training.wandb.project_name)

    print(OmegaConf.to_yaml(cfg))

    set_seed(cfg.training.params.seed)

    # Load dataset
    dataset = load_dataset("json", field="data", data_files=cfg.paths.dataset)
    dataset_train = dataset["train"]

    print(dataset_train)

    # Load chunks
    with open(cfg.paths.chunks, "rb") as f:
        chunks = pickle.load(f)

    # Pick the first chunk from every document by title
    chunks_dict = {}
    for c in chunks:
        if c.metadata["title"] not in chunks_dict:
            chunks_dict[c.metadata["title"]] = c
    chunks_with_cuis = list(chunks_dict.values())

    chunks = dataset_train["chunk"]
    queries = dataset_train["queries"]
    cuis = [c.metadata["cui"] for c in chunks_with_cuis]

    # Flatten queries, chunks, and cuis
    flattened_queries = []
    flattened_chunks = []
    flattened_cuis = []

    for query_list, chunk, cui in zip(queries, chunks, cuis):
        for query in query_list:
            flattened_queries.append(query)
            flattened_chunks.append(chunk)
            flattened_cuis.append(cui)

    # Group data by CUI
    cui_to_data = defaultdict(list)
    for cui, query, chunk in zip(flattened_cuis, flattened_queries, flattened_chunks):
        if cui is not None:
            cui_to_data[cui].append((query, chunk))

    # Split CUIs into train and test sets
    cuis_list = list(cui_to_data.keys())
    train_cuis, test_cuis = train_test_split(cuis_list, test_size=0.1, random_state=cfg.training.params.seed)

    # Extract train and test data based on the split CUIs
    queries_train, chunks_train = [], []
    queries_test, chunks_test = [], []

    for cui in train_cuis:
        for query, chunk in cui_to_data[cui]:
            queries_train.append(query)
            chunks_train.append(chunk)

    for cui in test_cuis:
        for query, chunk in cui_to_data[cui]:
            queries_test.append(query)
            chunks_test.append(chunk)


    # # TF-IDF Vectorization
    # vectorizer = TfidfVectorizer()
    # X = vectorizer.fit_transform(queries)

    # # K-Means Clustering
    # n_clusters = 5  # You can adjust the number of clusters based on your data
    # kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(X)

    # # Assign each query to a cluster
    # labels = kmeans.labels_
    # print(labels)

    # # Group queries and chunks by cluster
    # clustered_data = defaultdict(list)
    # for label, query, chunk in zip(labels, queries, chunks):
    #     clustered_data[label].append((query, chunk))

    # # Split clusters into train and test sets
    # train_clusters, test_clusters = train_test_split(
    #     list(clustered_data.keys()), test_size=0.2, random_state=42
    # )

    # # Extract queries and chunks for train and test sets
    # queries_train, chunks_train = [], []
    # queries_test, chunks_test = [], []

    # for cluster in train_clusters:
    #     for query, chunk in clustered_data[cluster]:
    #         queries_train.append(query)
    #         chunks_train.append(chunk)

    # for cluster in test_clusters:
    #     for query, chunk in clustered_data[cluster]:
    #         queries_test.append(query)
    #         chunks_test.append(chunk)

    train_dataset = Dataset.from_dict({
        "queries": queries_train,
        "chunks": chunks_train,
    })

    test_dataset = Dataset.from_dict({
        "queries": queries_test,
        "chunks": chunks_test,
    })

    corpus = {str(i):c for i,c in enumerate(chunks_test)}
    queries_eval, relevant_docs = {}, {}

    for i, q in enumerate(queries_test):
            queries_eval[i] = q
            relevant_docs[i] = [str(i)]

    model = SentenceTransformer(cfg.training.model_name)
    train_loss = losses.MultipleNegativesRankingLoss(
        model=model,
        similarity_fct=util.dot_score,
        scale=1,
    )

    training_args = SentenceTransformerTrainingArguments(
        # Required parameter:
        output_dir=cfg.training.checkpoint_dir,
        # Optional training parameters:
        num_train_epochs=cfg.training.params.num_epochs,
        learning_rate=2e-5,
        per_device_train_batch_size=32,
        per_device_eval_batch_size=32,
        warmup_ratio=cfg.training.params.warmup_ratio,
        fp16=True,  # Set to False if your GPU can't handle FP16
        bf16=False,  # Set to True if your GPU supports BF16
        batch_sampler=BatchSamplers.NO_DUPLICATES,  # Losses using "in-batch negatives" benefit from no duplicates
        # Optional tracking/debugging parameters:
        eval_strategy="steps",
        eval_steps=500,
        eval_on_start=True,
        save_strategy="steps",
        save_steps=500,
        logging_steps=100,
        run_name=cfg.dataset.name,  # Used in W&B if `wandb` is installed
        report_to=["wandb"],
        load_best_model_at_end=True,
        metric_for_best_model="eval_dot_accuracy@5",
    )

    evaluator = evaluation.InformationRetrievalEvaluator(
        queries=queries_eval,
        corpus=corpus,
        relevant_docs=relevant_docs,
        main_score_function=SimilarityFunction.DOT_PRODUCT,
        show_progress_bar=True,
    )

    trainer = SentenceTransformerTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        loss=train_loss,
        evaluator=evaluator,
    )

    # print(trainer.args)
    trainer.train()
    trainer.evaluate()
    trainer.save_model(output_dir=cfg.training.output_dir)

if __name__ == "__main__":
    train_model()
