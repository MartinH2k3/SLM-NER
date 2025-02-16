import json
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    pipeline
)

def get_base_model():
    with open('config.json', 'r') as file:
        config = json.load(file)
    model_path = config.get("model_path")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16
    )

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        device_map="auto",
        trust_remote_code=True,
        quantization_config=bnb_config
    )
    model.eval()

    return model


def get_base_tokenizer():
    with open('config.json', 'r') as file:
        config = json.load(file)
    model_path = config.get("model_path")

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    tokenizer.pad_token = tokenizer.unk_token

    return tokenizer

def _prepare_for_inference(user_input: str, system_prompt: str, tokenizer):
    prompt_data = []
    if len(system_prompt):
        prompt_data.append({"role": "system", "content": system_prompt})
    prompt_data.append({"role": "user", "content": user_input})
    return tokenizer.apply_chat_template(
        prompt_data, tokenize=False, add_generation_prompt=True
    )

def generate_response(user_input: str, model, tokenizer):
    with open('config.json', 'r') as file:
        config = json.load(file)
    system_prompt_path = config.get("system_prompt_path")
    with open(system_prompt_path, "r") as f:
        system_prompt = f.read()
    prepared_input = _prepare_for_inference(user_input, system_prompt, tokenizer)
    generation_args = {
        "max_new_tokens": 150,
        "return_full_text": False,
    }
    generation_pipeline = pipeline("text-generation", model=model, tokenizer=tokenizer)
    return generation_pipeline(prepared_input, **generation_args)[0]['generated_text']

