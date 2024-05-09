import json
import pickle
from typing import List

from datasets import load_dataset
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents import Document


def save_data(chunks, examples):
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


def make_example_queries(chunks):
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
    chunks,
    n_requests,
    example_prompt,
    system_prompt,
    user_prompt_template,
    file_name="data/processed/batch_tasks_queries.jsonl",
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

    with open(file_name, "w") as f:
        for obj in tasks:
            f.write(json.dumps(obj))
            f.write("\n")


def make_requests(
    chunks: List[Document],
    example_prompt: str,
    system_prompt: str,
    user_prompt_template: str,
    n_requests: int = 0,
    file_name: str = "data/processed/api_requests.jsonl",
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
            }
            requests.append(request)

    with open(file_name, "w") as f:
        for obj in requests:
            f.write(json.dumps(obj))
            f.write("\n")


def make_dataset(load_from_disk=False, save_to_disk=False):
    if load_from_disk:
        chunks, examples = load_data()
    else:
        chunks = make_chunks(chunk_size=1000, chunk_overlap=100)
        examples = make_example_queries(chunks)
        if save_to_disk:
            save_data(chunks, examples)

    print(f"Number of chunks: {len(chunks)}")

    # Pick the first chunk from every document by title
    chunks_dict = {}
    for c in chunks:
        if c.metadata["title"] not in chunks_dict:
            chunks_dict[c.metadata["title"]] = c
    chunks = list(chunks_dict.values())

    system_prompt = (
        "You are a medical professional generating queries relevant to a patient's symptoms.\n\n"
        "You are given an example of a query and a relevant chunk of text from a medical document.\n\n"
        "Generate a comma-separated query of patient symptoms that is relevant to the given chunk."
    )

    user_prompt_template = "Title: {chunk_title}\nChunk: {chunk_content}\nUser query: {user_query}"

    example_prompt = user_prompt_template.format(
        chunk_title=examples[0]["chunk_title"],
        chunk_content=examples[0]["chunk_content"],
        user_query=examples[0]["query"],
    )

    make_requests(chunks, example_prompt, system_prompt, user_prompt_template)

    # prompt = FewShotPromptTemplate(
    #     examples=examples,
    #     example_prompt=example_prompt,
    #     prefix=prefix,
    #     suffix="Title:{title}\n\nChunk:{chunk}\n\nUser query:",
    #     input_variables=['title','chunk']
    # )

    # llm = ChatOpenAI(
    #     model_name="gpt-3.5-turbo",
    #     temperature=0.0,
    #     api_key=openai_api_key
    # )

    # # # Print 10 examples of title and chunks
    # # for title, chunks in list(chunks_dict.items())[:10]:
    # #     print(prompt.format(title=title, chunks=chunks))

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

    # dataset = [{'query': q, 'chunk': c.page_content} for c, q in zip(chunks, queries)]
    # data = {
    #     'version': '0.0.1',
    #     'data': dataset
    # }

    # # Save dataset as json
    # with open('data/processed/dataset.json', 'w', encoding='utf-8') as f:
    #     json.dump(data, f, indent=4)


if __name__ == "__main__":
    make_dataset(load_from_disk=True, save_to_disk=False)

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
