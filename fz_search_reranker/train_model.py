from datasets import load_dataset
from sentence_transformers import (InputExample, SentenceTransformer,
                                   evaluation, losses)
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader


def train_model(data_path="data/processed/dataset.json"):
    dataset = load_dataset("json", field="data", data_files=data_path)
    dataset_train = dataset["train"]

    queries = dataset_train["query"]
    chunks = dataset_train["chunk"]

    queries_train, queries_test, chunks_train, chunks_test = train_test_split(queries, chunks, test_size=0.2)

    train_examples = []
    for q, c in zip(queries_train, chunks_train):
        train_examples.append(InputExample(texts=[q, c]))

    corpus, queries_eval, relevant_docs = {}, {}, {}
    for i, _ in enumerate(queries_test):
        corpus[str(i)] = chunks_test[i]
        queries_eval[str(i)] = queries_test[i]
        relevant_docs[str(i)] = [str(i)]

    model = SentenceTransformer("msmarco-distilbert-base-v4")

    train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=16)
    train_loss = losses.MultipleNegativesRankingLoss(model=model)
    num_epochs = 10
    warmup_steps = int(len(train_dataloader) * num_epochs * 0.1)  # 10% of train data

    evaluator = evaluation.InformationRetrievalEvaluator(
        queries=queries_eval, corpus=corpus, relevant_docs=relevant_docs, show_progress_bar=True
    )

    model.fit(
        train_objectives=[(train_dataloader, train_loss)],
        evaluator=evaluator,
        epochs=num_epochs,
        warmup_steps=warmup_steps,
        show_progress_bar=True,
        checkpoint_path="models/",
        output_path="models/",
    )


if __name__ == "__main__":
    train_model()
