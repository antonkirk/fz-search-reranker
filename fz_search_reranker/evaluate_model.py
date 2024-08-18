from whoosh.index import create_in, open_dir
from whoosh.fields import Schema, TEXT, ID
from whoosh.qparser import QueryParser, OrGroup
from datasets import load_dataset, Dataset
from sentence_transformers import SentenceTransformer
from omegaconf import DictConfig, OmegaConf
from collections import defaultdict
from tqdm import tqdm
import csv
import os
import hydra
import pickle
import faiss

def calculate_metrics(retrieved_cuis, cuis, total_relevant, ks):
    total_hitatk = [0 for _ in ks]
    total_precisionatk = [0 for _ in ks]
    total_recallatk = [0 for _ in ks]
    total_mrratk = [0 for _ in ks]

    for i, k in enumerate(ks):
        # Hit@k
        hitatk = 1 if any(cui in retrieved_cuis[:k] for cui in cuis) else 0
        total_hitatk[i] += hitatk

        # Precision@k
        relevant_retrieved = sum(1 for cui in retrieved_cuis[:k] if cui in cuis)
        precisionatk = relevant_retrieved / k
        total_precisionatk[i] += precisionatk

        # Recall@k
        recallatk = relevant_retrieved / total_relevant if total_relevant > 0 else 0
        total_recallatk[i] += recallatk

        # MRR@k
        for rank, cui in enumerate(retrieved_cuis[:k], start=1):
            if cui in cuis:
                total_mrratk[i] += 1 / rank
                break

    return total_hitatk, total_precisionatk, total_recallatk, total_mrratk

def evaluate_whoosh(query, cuis, total_relevant, ix, ks):
    with ix.searcher() as searcher:
        query_parser = QueryParser("content", ix.schema, group=OrGroup)
        whoosh_query = query_parser.parse(query)
        results = searcher.search(whoosh_query, limit=max(ks))
        retrieved_cuis = [result['cui'] for result in results]
        return calculate_metrics(retrieved_cuis, cuis, total_relevant, ks)

def evaluate_faiss(query_embedding, corpus, cuis, total_relevant, ks):
    scores, results = corpus.get_nearest_examples('embeddings', query_embedding, k=max(ks))
    retrieved_cuis = results['cui']
    # print(f"True CUIs: {cuis}")
    # print(f"Retrieved CUIs: {retrieved_cuis}")
    return calculate_metrics(retrieved_cuis, cuis, total_relevant, ks)


def rrf(ranked_lists, k=60):
    combined_scores = defaultdict(float)
    
    for ranked_list in ranked_lists:
        for rank, doc in enumerate(ranked_list):
            combined_scores[doc] += 1 / (rank + 1 + k)
    
    sorted_docs = sorted(combined_scores.items(), key=lambda item: item[1], reverse=True)
    return [doc for doc, score in sorted_docs]

def evaluate_hybrid(query, query_embedding, corpus, cuis, total_relevant, ix, ks):
    # Obtain results from both types of search
    with ix.searcher() as searcher:
        query_parser = QueryParser("content", ix.schema, group=OrGroup)
        whoosh_query = query_parser.parse(query)
        whoosh_results = searcher.search(whoosh_query, limit=max(ks))
        whoosh_cuis = [result['cui'] for result in whoosh_results]
    
    faiss_scores, faiss_results = corpus.get_nearest_examples('embeddings', query_embedding, k=max(ks))
    faiss_cuis = faiss_results['cui']
    
    # Combine results using RRF
    combined_cuis = rrf([whoosh_cuis, faiss_cuis])[:max(ks)]
    # print(f"True CUIs: {cuis}")
    # print(f"Combined CUIs: {combined_cuis}")

    return calculate_metrics(combined_cuis, cuis, total_relevant, ks)
    

def precount_cuis(corpus):
    cui_count = defaultdict(int)
    for row in corpus:
        for cui in row["cui"].split():  # Split the string to get individual CUIs
            cui_count[cui] += 1
    return cui_count

def load_chunks(chunks_path):
    with open(chunks_path, 'rb') as f:
        chunks = pickle.load(f)
    return chunks

def create_dataset(chunks):
    corpus = Dataset.from_dict({
        "text": [c.page_content for c in chunks],
        "title": [c.metadata["title"] for c in chunks],
        "cui": [c.metadata["cui"] for c in chunks],
    })
    return corpus

def create_whoosh_index(chunks, index_dir):
    schema = Schema(title=TEXT(stored=True), content=TEXT(stored=True), cui=ID(stored=True))
    if not os.path.exists(index_dir):
        os.mkdir(index_dir)
    ix = create_in(index_dir, schema)
    writer = ix.writer()

    for chunk in chunks:
        writer.add_document(title=chunk.metadata["title"], content=chunk.page_content, cui=chunk.metadata["cui"])
    writer.commit()

    return open_dir(index_dir)

def create_faiss_index(corpus, embedding_model, index_dir, device):
    embeddings = embedding_model.encode(
        corpus["text"],
        normalize_embeddings=False,
        show_progress_bar=True,
        device=device,
        convert_to_numpy=True,
    )
    print("Finished embeddings!")
    embeddings = embeddings.reshape(-1, embeddings.shape[-1]).tolist()
    print("Adding embeddings to corpus...")
    corpus = corpus.add_column('embeddings', embeddings)
    corpus.add_faiss_index(
        column="embeddings",
        metric_type=faiss.METRIC_INNER_PRODUCT, # the models are trained for dot product
    )
    print("Index created!")
    print("Saving index...")
    corpus.save_faiss_index('embeddings', index_dir)
    print("Index saved!")
    return corpus

@hydra.main(config_path="configs", config_name="config", version_base=None)
def evaluate_model(cfg: DictConfig):
    print(OmegaConf.to_yaml(cfg))
    ks = cfg.eval.ks
    
    if cfg.eval.dataset.name == "findzebra":
        dataset_eval = load_dataset(cfg.eval.dataset.path)["train"]
    if cfg.eval.dataset.name == "ada_dx":
        dataset_eval = load_dataset('json', data_files=cfg.eval.dataset.path)["train"]
        # Only use the first 5 symptoms for each query
        # queries = [','.join(q.split(',')[:5]) for q in dataset_eval["pathological_symptoms"]]
        queries = [q for q in dataset_eval["pathological_symptoms"]]
        # Make each entry in cui a list with one or more CUIs, splitting by comma
        cuis = [c.split(', ') for c in dataset_eval["cui"]]
        dataset_eval = Dataset.from_dict({"query": queries, "cuis": cuis})

    dataset_eval = dataset_eval.filter(lambda x: x["cuis"] is not None) # Filter `None` values
    chunks = load_chunks(cfg.paths.chunks)
    corpus = create_dataset(chunks)

    if cfg.eval.use_whoosh or cfg.eval.use_hybrid:
        if not os.path.exists(cfg.eval.whoosh_index_dir):
            ix = create_whoosh_index(chunks, cfg.eval.whoosh_index_dir)
        else:
            ix = open_dir(cfg.eval.whoosh_index_dir)

    if cfg.eval.use_faiss or cfg.eval.use_hybrid:
        embedding_model = SentenceTransformer(model_name_or_path=cfg.model.model_dir)
        if not os.path.exists(cfg.model.index_dir):
            corpus = create_faiss_index(corpus, embedding_model, cfg.model.index_dir, cfg.sys.device)
        else:
            corpus.load_faiss_index('embeddings', cfg.model.index_dir)

    metrics = defaultdict(lambda: defaultdict(list))

    cui_count = precount_cuis(corpus)

    for example in tqdm(dataset_eval, desc=f"Evaluating model on {cfg.eval.dataset.name}"):
        query = example['query']
        cuis = example['cuis']
        tqdm.write(f"Cuis: {cuis}")
        total_relevant = sum(cui_count[cui] for cui in cuis if cui in cui_count)
        if cfg.eval.use_faiss or cfg.eval.use_hybrid:
            query_embedding = embedding_model.encode(query, convert_to_numpy=True, show_progress_bar=False)

        if cfg.eval.use_whoosh:
            whoosh_metrics = evaluate_whoosh(query, cuis, total_relevant, ix, ks)
            for i, k in enumerate(ks):
                metrics["whoosh"][f"hit@{k}"].append(whoosh_metrics[0][i])
                metrics["whoosh"][f"precision@{k}"].append(whoosh_metrics[1][i])
                metrics["whoosh"][f"recall@{k}"].append(whoosh_metrics[2][i])
                metrics["whoosh"][f"mrr@{k}"].append(whoosh_metrics[3][i])

        if cfg.eval.use_faiss:
            faiss_metrics = evaluate_faiss(query_embedding, corpus, cuis, total_relevant, ks)
            for i, k in enumerate(ks):
                metrics["faiss"][f"hit@{k}"].append(faiss_metrics[0][i])
                metrics["faiss"][f"precision@{k}"].append(faiss_metrics[1][i])
                metrics["faiss"][f"recall@{k}"].append(faiss_metrics[2][i])
                metrics["faiss"][f"mrr@{k}"].append(faiss_metrics[3][i])

        if cfg.eval.use_hybrid:
            hybrid_metrics = evaluate_hybrid(query, query_embedding, corpus, cuis, total_relevant, ix, ks)
            for i, k in enumerate(ks):
                metrics["hybrid"][f"hit@{k}"].append(hybrid_metrics[0][i])
                metrics["hybrid"][f"precision@{k}"].append(hybrid_metrics[1][i])
                metrics["hybrid"][f"recall@{k}"].append(hybrid_metrics[2][i])
                metrics["hybrid"][f"mrr@{k}"].append(hybrid_metrics[3][i])

    # Calculate average metrics for each k
    average_metrics = defaultdict(lambda: defaultdict(float))
    for key in metrics:
        for metric in metrics[key]:
            average_metrics[key][metric] = sum(metrics[key][metric]) / len(metrics[key][metric])

    # Print metrics to console
    for metric in ['hit', 'precision', 'recall', 'mrr']:
        for k in ks:
            metric_name = f"{metric}@{k}"
            print(f"{metric_name}:")
            for key in average_metrics:
                print(f"  {key}: {average_metrics[key][metric_name]}")

    # Save metrics to CSV
    if cfg.eval.save_metrics:
        fieldnames = ['metric']
        if cfg.eval.use_whoosh:
            fieldnames.append('whoosh')
        if cfg.eval.use_faiss:
            fieldnames.append('faiss')
        if cfg.eval.use_hybrid:
            fieldnames.append('hybrid')

        with open(cfg.eval.metrics_output_path, 'w', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            for metric in ['hit', 'precision', 'recall', 'mrr']:
                for k in ks:
                    metric_name = f"{metric}@{k}"
                    row = {'metric': metric_name}
                    if cfg.eval.use_whoosh:
                        row['whoosh'] = average_metrics["whoosh"].get(metric_name, 0)
                    if cfg.eval.use_faiss:
                        row['faiss'] = average_metrics["faiss"].get(metric_name, 0)
                    if cfg.eval.use_hybrid:
                        row['hybrid'] = average_metrics["hybrid"].get(metric_name, 0)
                    writer.writerow(row)

if __name__ == "__main__":
    evaluate_model()
