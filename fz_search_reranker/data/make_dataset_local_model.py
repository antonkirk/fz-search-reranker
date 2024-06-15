import torch
from transformers import pipeline


def generate_queries_with_local_model(model_id: str, device: str):
    pipe = pipeline(
        "text-generation",
        model=model_id,
        model_kwargs={"torch_dtype": torch.bfloat16},
        device=device,
    )

    messages = [
        {
            "role": "system",
            "content": "You are an expert and experienced from the healthcare and biomedical domain with extensive medical knowledge and practical experience. Your name is OpenBioLLM, and you were developed by Saama AI Labs. who's willing to help answer the user's query with explanation. In your explanation, leverage your deep medical expertise such as relevant anatomical structures, physiological processes, diagnostic criteria, treatment guidelines, or other pertinent medical concepts. Use precise medical terminology while still aiming to make the explanation clear and accessible to a general audience.",
        },
        {"role": "user", "content": "Hi there, can you speak like a pirate?"},
    ]

    terminators = [pipe.tokenizer.eos_token_id, pipe.tokenizer.convert_tokens_to_ids("<|eot_id|>")]

    outputs = pipe(
        messages,
        max_new_tokens=256,
        eos_token_id=terminators,
        do_sample=True,
        temperature=0.6,
        top_p=0.9,
    )

    print(outputs[0]["generated_text"][-1]["content"])


if __name__ == "__main__":
    generate_queries_with_local_model(model_id="aaditya/OpenBioLLM-Llama3-8B", device="cuda")
