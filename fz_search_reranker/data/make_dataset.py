import json
import pickle
from pathlib import Path
from typing import List

import hydra
import torch
import transformers
from datasets import load_dataset
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from omegaconf import DictConfig


def save_data(chunks: List[Document], examples: List[Document]):
    with open("data/processed/chunks.pkl", "wb") as f:
        pickle.dump(chunks, f)
    with open("data/processed/example_queries.pkl", "wb") as f:
        pickle.dump(examples, f)


def load_data():
    # Load chunks
    with open("data/processed/chunks.pkl", "rb") as f:
        chunks = pickle.load(f)
    # Load example queries
    with open("data/processed/example_queries.pkl", "rb") as f:
        example_queries = pickle.load(f)

    return chunks, example_queries


def make_example_queries(chunks: List[Document]):
    dataset = load_dataset("findzebra/queries")
    dataset_train = dataset["train"]

    examples_queries = []

    for i in range(1, 2):
        query = dataset_train["query"][i]
        relevant_chunk = next((c for c in chunks if c.metadata["cui"] in dataset_train["cuis"][i]))
        examples_queries.append(
            {
                "chunk_title": relevant_chunk.metadata["title"],
                "chunk_content": relevant_chunk.page_content,
                "query": query,
            }
        )

    return examples_queries


def make_chunks(chunk_size: int, chunk_overlap: int, num_examples: int = 0) -> List[Document]:
    dataset = load_dataset("findzebra/corpus")
    dataset_train = dataset["train"]

    if num_examples == 0:
        num_examples = len(dataset_train["text"])

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap, length_function=len
    )

    metadatas = [
        {"title": dataset_train["title"][i], "cui": dataset_train["cui"][i], "source": dataset_train["source"][i]}
        for i in range(num_examples)
    ]

    chunks = text_splitter.create_documents(texts=dataset_train["text"][:num_examples], metadatas=metadatas)

    return chunks


def make_batch_requests(
    chunks: List[Document],
    n_requests: int,
    example_prompt: str,
    system_prompt: str,
    user_prompt_template: str,
    file_path: str = "data/processed/batch_tasks_queries.jsonl",
):
    """"""
    tasks = []
    for index, chunk in enumerate(chunks):
        if index < n_requests:
            description = user_prompt_template.format(
                chunk_title=chunk.metadata["title"], chunk_content=chunk.page_content, user_query=""
            )

            task = {
                "custom_id": f"task-{index}",
                "method": "POST",
                "url": "/v1/chat/completions",
                "body": {
                    # This is what you would have in your Chat Completions API call
                    "model": "gpt-3.5-turbo",
                    "temperature": 0.1,
                    "messages": [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": example_prompt},
                        {"role": "user", "content": description},
                    ],
                },
            }

            tasks.append(task)

    with open(file_path, "w") as f:
        for obj in tasks:
            f.write(json.dumps(obj))
            f.write("\n")


def make_requests(
    chunks: List[Document],
    example_prompt: str,
    system_prompt: str,
    user_prompt_template: str,
    n_requests: int = 0,
    file_path: str = "data/processed/api_requests.jsonl",
):
    if n_requests == 0:
        n_requests = len(chunks)

    requests = []
    for index, chunk in enumerate(chunks):
        if index < n_requests:
            description = user_prompt_template.format(
                chunk_title=chunk.metadata["title"], chunk_content=chunk.page_content, user_query=""
            )

            request = {
                "model": "gpt-3.5-turbo",
                "temperature": 0.1,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": example_prompt},
                    {"role": "user", "content": description},
                ],
                "metadata": {"row_id": index},
            }
            requests.append(request)

    with open(file_path, "w") as f:
        for obj in requests:
            f.write(json.dumps(obj))
            f.write("\n")


def generate_queries_with_local_model(model: str, device: str):
    model_id = "aaditya/OpenBioLLM-Llama3-8B"

    pipeline = transformers.pipeline(
        "text-generation",
        model=model_id,
        model_kwargs={"torch_dtype": torch.bfloat16},
        device="auto",
    )

    messages = [
        {
            "role": "system",
            "content": "You are an expert and experienced from the healthcare and biomedical domain with extensive medical knowledge and practical experience. Your name is OpenBioLLM, and you were developed by Saama AI Labs. who's willing to help answer the user's query with explanation. In your explanation, leverage your deep medical expertise such as relevant anatomical structures, physiological processes, diagnostic criteria, treatment guidelines, or other pertinent medical concepts. Use precise medical terminology while still aiming to make the explanation clear and accessible to a general audience.",
        },
        {"role": "user", "content": "How can i split a 3mg or 4mg waefin pill so i can get a 2.5mg pill?"},
    ]

    prompt = pipeline.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    terminators = [pipeline.tokenizer.eos_token_id, pipeline.tokenizer.convert_tokens_to_ids("<|eot_id|>")]

    outputs = pipeline(
        prompt,
        max_new_tokens=256,
        eos_token_id=terminators,
        do_sample=True,
        temperature=0.0,
        top_p=0.9,
    )

    print(outputs[0]["generated_text"][len(prompt) :])


def extract_responses_from_jsonl(chunks, file_path):
    queries = []
    with open(file_path, "r") as f:
        for line in f:
            response = json.loads(line)
            query = response[1]["choices"][0]["message"]["content"]
            row_id = response[2]["row_id"]
            queries.append((query, row_id))
    queries.sort(key=lambda x: x[1])

    dataset = [{"query": q[0], "chunk": c.page_content} for q, c in zip(queries, chunks)]

    data = {"version": "0.0.2", "data": dataset}

    # Save dataset as json
    with open("data/processed/dataset_v0.0.2.json", "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4)


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def make_dataset(cfg: DictConfig):
    # print(OmegaConf.to_yaml(cfg))

    if cfg.data.load_from_disk:
        chunks, examples = load_data()
    else:
        chunks = make_chunks(chunk_size=1000, chunk_overlap=100)
        examples = make_example_queries(chunks)
        if cfg.data.save_to_disk:
            save_data(chunks, examples)

    # print(f"Number of chunks: {len(chunks)}")

    # Pick the first chunk from every document by title
    chunks_dict = {}
    for c in chunks:
        if c.metadata["title"] not in chunks_dict:
            chunks_dict[c.metadata["title"]] = c
    chunks = list(chunks_dict.values())

    # example_prompt = cfg.data.user_prompt_template.format(
    #     chunk_title=examples[0]["chunk_title"],
    #     chunk_content=examples[0]["chunk_content"],
    #     user_query=examples[0]["query"],
    # )

    # make_requests(
    #     chunks=chunks,
    #     n_requests=cfg.data.n_requests,
    #     example_prompt=example_prompt,
    #     system_prompt=cfg.data.system_prompt,
    #     user_prompt_template=cfg.data.user_prompt_template,
    #     file_path=Path(f"{cfg.sys.work_dir}/{cfg.data.paths.requests}")
    #     )

    extract_responses_from_jsonl(chunks, Path(f"{cfg.sys.work_dir}/{cfg.data.paths.requests_results}"))

    # num_examples = 5000
    # queries = []
    # start_time = time.time()
    # for i in range(num_examples):
    #     response = llm.invoke(prompt.format(title=chunks[i].metadata['title'], chunk=chunks[i].page_content))
    #     print(response.content)
    #     queries.append(response.content)

    #     print(f"Example {i+1} of {num_examples} completed")
    #     print(f"Time taken: {time.time() - start_time}")

    # print(f"Total time taken: {time.time() - start_time}")


if __name__ == "__main__":
    # make_dataset()
    generate_queries_with_local_model("aaditya/OpenBioLLM-Llama3-8B-GGUF", "cuda")

    # dataset = load_dataset('findzebra/corpus')
    # dataset_train = dataset['train']
    # # Determine total length of text in dataset in tokens
    # total_tokens = 0
    # for i in tqdm(range(1000)):
    #     # print(dataset_train['text'][i])
    #     total_tokens += len(dataset_train['text'][i]) // 4
    # cost_per_token = 0.0005
    # total_cost = cost_per_token * (total_tokens / 1000)
    # print(f"Total tokens: {total_tokens}\nTotal cost:{total_cost}")

    # for text in dataset_train['text']:
    #     if len(text)/4 > 16000:
    #         print(len(text)/4)
    #         break
