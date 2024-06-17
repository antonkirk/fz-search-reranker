import pickle
from typing import List

import hydra
import torch
from langchain_core.documents import Document
from omegaconf import DictConfig
from transformers import pipeline


def generate_queries_with_local_model(
    model_id: str,
    device: str,
    chunks: List[Document],
    n_requests: int,
    example_prompt: str,
    system_prompt: str,
    user_prompt_template: str,
    file_path: str,
) -> None:
    pipe = pipeline(
        "text-generation",
        model=model_id,
        model_kwargs={"torch_dtype": torch.bfloat16},
        device=device,
    )

    if n_requests == 0:
        n_request = len(chunks)

    requests = []

    for index, chunk in enumerate(chunks):
        if index >= n_requests:
            break

        description = user_prompt_template.format(
            chunk_title=chunk.metadata["title"], chunk_content=chunk.page_content, user_query=""
        )

        task = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": example_prompt},
            {"role": "user", "content": description},
        ]

        terminators = [pipe.tokenizer.eos_token_id, pipe.tokenizer.convert_tokens_to_ids("<|eot_id|>")]

        output = pipe(
            task,
            max_new_tokens=256,
            eos_token_id=terminators,
            do_sample=True,
            temperature=0.6,
            top_p=0.9,
        )

        response = output[0]["generated_text"][-1]["content"]
        print(response)
        requests.append(response)

    # TODO: Save dataset


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def make_data_local_model(cfg: DictConfig):
    # Load chunks and example queries
    with open(cfg.data.paths.chunks, "rb") as f:
        chunks = pickle.load(f)
    with open(cfg.data.paths.example_queries, "rb"):
        example_queries = pickle.load(f)

    # Prepare chunks
    # Pick the first chunk from every document by title
    chunks_dict = {}
    for c in chunks:
        if c.metadata["title"] not in chunks_dict:
            chunks_dict[c.metadata["title"]] = c
    chunks = list(chunks_dict.values())

    example_prompt = cfg.data.user_prompt_template.format(
        chunk_title=example_queries[0]["chunk_title"],
        chunk_content=example_queries[0]["chunk_content"],
        user_query=example_queries[0]["query"],
    )

    # TODO: Change prompt to make more queries per chunk
    generate_queries_with_local_model(
        model_id="aaditya/OpenBioLLM-Llama3-8B",
        device=cfg.sys.device,
        chunks=chunks,
        n_requests=cfg.data.params.n_requests,
        example_prompt=example_prompt,
        system_prompt=cfg.data.system_prompt,
        user_prompt_template=cfg.data.user_prompt_template,
        file_path=cfg.data.paths.output_local_model,
    )


if __name__ == "__main__":
    make_data_local_model()
