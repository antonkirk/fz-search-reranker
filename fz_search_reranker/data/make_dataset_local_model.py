import pickle
import json
from typing import List
import sys

import hydra
import torch
from langchain_core.documents import Document
from omegaconf import DictConfig, OmegaConf
from transformers import pipeline, AutoTokenizer, set_seed
from tqdm import tqdm
from datasets import Dataset
from transformers.pipelines.pt_utils import KeyDataset


def generate_queries_with_local_model(
    model_id: str,
    gguf_file: str,
    device: str,
    cache_dir: str,
    seed: int,
    temperature: float,
    chunks: List[Document],
    n_requests: int,
    example_prompts: List[str],
    system_prompt: str,
    user_prompt_template: str,
    chat_template: str,
) -> List[List[str]]:

    set_seed(seed)
    tokenizer = AutoTokenizer.from_pretrained(model_id, padding_side='left')
    pipe = pipeline(
        "text-generation",
        tokenizer=tokenizer,
        model=model_id,
        model_kwargs={"torch_dtype": torch.bfloat16, "cache_dir": cache_dir},
        device=device,
    )

    # Set the tokenizer's pad_token_id to the model's eos_token_id
    pipe.tokenizer.pad_token_id = pipe.model.config.eos_token_id
    terminators = [pipe.tokenizer.eos_token_id, pipe.tokenizer.convert_tokens_to_ids("<|eot_id|>")]
    tokenizer.chat_template = chat_template

    if n_requests == 0:
        n_requests = len(chunks)

    requests = []

    for index, chunk in enumerate(chunks):
        if index >= n_requests:
            break

        description = user_prompt_template.format(
            chunk_title=chunk.metadata["title"], chunk_content=chunk.page_content, user_query=""
        )

        example = user_prompt_template.format(
            chunk_title="MELAS syndrome",
            chunk_content="Mitochondrial myopathy, encephalomyopathy, lactic acidosis, and stroke-like episodes\nBasal ganglia calcification, cerebellar atrophy, increased lactate; a CT image of a person diagnosed with MELAS\nSpecialty: Sex\nFrequency: 1 in 4000\n\nMitochondrial encephalopathy, lactic acidosis, and stroke-like episodes (MELAS) is one of the family of mitochondrial cytopathies, which also include MERRF, and Leber's hereditary optic neuropathy. It was first characterized under this name in 1984. A feature of these diseases is that they are caused by defects in the mitochondrial genome which is inherited purely from the female parent",
            user_query="seizure, confusion, dysphasia, T2 lesions\nMitochondrial myopathy, encephalomyopathy, lactic acidosis, stroke-like episodes\nBasal ganglia calcification, cerebellar atrophy, increased lactate, mitochondrial cytopathies",
        )

        request = [
            {"role": "system", "content": f"{system_prompt}"},
            # {"role": "user", "content": example},
           # {"role": "user", "content": '\n'.join(example_prompts)},
            {"role": "user", "content": description},
        ]

        #print(request)
        tokenizer.apply_chat_template(request, tokenize=False, add_generation_prompt=True)
        requests.append(request)

    dataset = Dataset.from_dict({"requests": requests})
    queries_by_chunk = []

    for out in tqdm(
        pipe(
            KeyDataset(dataset, "requests"),
            max_new_tokens=256,
            batch_size=8,
            eos_token_id=terminators,
            pad_token_id=pipe.tokenizer.eos_token_id,
            do_sample=True,
            temperature=temperature,
            top_p=0.9
        )
    ):
        response = out[0]["generated_text"][-1]["content"]
        print(response)
        queries = response.splitlines()
        queries_by_chunk.append(queries)

    return queries_by_chunk

@hydra.main(version_base=None, config_path="../configs", config_name="config")
def make_data_local_model(cfg: DictConfig):
    print(OmegaConf.to_yaml(cfg))

    # Load chunks and example queries
    chunks, example_queries = [],[]
    with open(cfg.paths.chunks, "rb") as f:
        chunks = pickle.load(f)
    with open(cfg.paths.example_queries, "rb") as f:
        example_queries = pickle.load(f)

    # Prepare chunks
    # Pick the first chunk from every document by title
    chunks_dict = {}
    for c in chunks:
        if c.metadata["title"] not in chunks_dict:
            chunks_dict[c.metadata["title"]] = c
    chunks = list(chunks_dict.values())

    example_prompts = []
    for i in range(1):
        example_prompt = cfg.prompts.user_prompt_template.format(
            chunk_title=example_queries[i]["chunk_title"],
            chunk_content=example_queries[i]["chunk_content"],
            user_query=example_queries[i]["query"],
        )
        example_prompts.append(example_prompt)

    queries_by_chunk = generate_queries_with_local_model(
        model_id=cfg.model.model_dir,
        gguf_file="",
        device=cfg.sys.device,
        cache_dir=cfg.sys.cache_dir,
        chunks=chunks,
        seed=cfg.data.seed,
        temperature=cfg.data.temperature,
        n_requests=cfg.data.n_requests,
        example_prompts=example_prompts,
        system_prompt=cfg.prompts.system_prompt,
        user_prompt_template=cfg.prompts.user_prompt_template,
        chat_template=cfg.model.chat_template,
    )

    # remove the first line and empty line, since Llama does not like to follow instructions
    # remove line numbering (same reason as above) 
    if cfg.data.clean_output:
        queries_by_chunk = [[q[3:] for q in queries[2:]] for queries in queries_by_chunk]

    # print(queries_by_chunk)
    if cfg.data.save_queries:
        data = [{"queries": q, "chunk": c.page_content} for q, c in zip(queries_by_chunk, chunks)]
        dataset = {"version": "0.0.1", "data": data}

        # print(f"data: {data}")
        # print(f"dataset: {dataset}")
        with open(cfg.paths.dataset, "w", encoding="utf-8") as f:
            json.dump(dataset, f, indent=4)


if __name__ == "__main__":
    make_data_local_model()
