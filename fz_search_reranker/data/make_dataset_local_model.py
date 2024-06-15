import torch
import transformers


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
        {"role": "user", "content": "Hi there, can you speak like a pirate?"},
    ]

    terminators = [pipeline.tokenizer.eos_token_id, pipeline.tokenizer.convert_tokens_to_ids("<|eot_id|>")]

    outputs = pipeline(
        messages,
        max_new_tokens=256,
        eos_token_id=terminators,
        do_sample=True,
        temperature=0.0,
        top_p=0.9,
    )

    print(outputs[0]["generated_text"][-1]["content"])


if __name__ == "__main__":
    generate_queries_with_local_model("aaditya/OpenBioLLM-Llama3-8B-GGUF", "cuda")
