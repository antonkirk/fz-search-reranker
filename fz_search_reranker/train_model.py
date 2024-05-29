import hydra
from datasets import load_dataset
from omegaconf import DictConfig, OmegaConf
from sentence_transformers import InputExample, SentenceTransformer, evaluation, losses
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader


@hydra.main(version_base=None, config_path="configs", config_name="config")
def train_model(cfg: DictConfig):
    print(OmegaConf.to_yaml(cfg))

    dataset = load_dataset("json", field="data", data_files=cfg.training.paths.data)
    dataset_train = dataset["train"]

    queries = dataset_train["query"]
    chunks = dataset_train["chunk"]

    queries_train, queries_test, chunks_train, chunks_test = train_test_split(
        queries, chunks, test_size=cfg.training.params.test_size
    )

    train_examples = []
    for q, c in zip(queries_train, chunks_train):
        train_examples.append(InputExample(texts=[q, c]))

    corpus, queries_eval, relevant_docs = {}, {}, {}
    for i, _ in enumerate(queries_test):
        corpus[str(i)] = chunks_test[i]
        queries_eval[str(i)] = queries_test[i]
        relevant_docs[str(i)] = [str(i)]

    model = SentenceTransformer(cfg.training.paths.model_name)

    train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=16)
    train_loss = losses.MultipleNegativesRankingLoss(model=model)
    num_epochs = cfg.training.params.num_epochs
    warmup_steps = int(len(train_dataloader) * num_epochs * cfg.training.params.warmup_pct)

    evaluator = evaluation.InformationRetrievalEvaluator(
        queries=queries_eval, corpus=corpus, relevant_docs=relevant_docs, show_progress_bar=True
    )

    model.fit(
        train_objectives=[(train_dataloader, train_loss)],
        evaluator=evaluator,
        epochs=num_epochs,
        warmup_steps=warmup_steps,
        show_progress_bar=True,
        checkpoint_path=cfg.training.paths.checkpoint,
        output_path=cfg.training.paths.output,
    )


if __name__ == "__main__":
    train_model()
