import json
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    pipeline
)
from peft import PeftConfig, PeftModel
import openai
from dotenv import load_dotenv
import os
from typing import Optional

with open('config.json', 'r') as file:
    _config = json.load(file)
_system_prompt_path = _config.get("system_prompt_path")
with open(_system_prompt_path, "r") as f:
    _system_prompt = f.read()
_model_path = _config.get("model_path")
_checkpoint_path = _config.get("checkpoint_path")
_gpt_model = "gpt-4o-mini"
_deepseek_model = "deepseek-chat"
_result_separator = _config.get("result_separator")
_litellm_url = "http://147.175.151.44/"
_deepseek_url = "https://api.deepseek.com"

def get_system_prompt():
    return _system_prompt


def get_base_model():
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16
    )

    model = AutoModelForCausalLM.from_pretrained(
        _model_path,
        device_map="auto",
        trust_remote_code=True,
        quantization_config=bnb_config
    )
    model.eval()

    return model


def get_finetuned_model(model_path="/checkpoint-145"):
    peft_config = PeftConfig.from_pretrained(_checkpoint_path + model_path)

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16
    )

    model = AutoModelForCausalLM.from_pretrained(
        peft_config.base_model_name_or_path,
        return_dict=True,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True
    )

    model = PeftModel.from_pretrained(model, _checkpoint_path + model_path)
    model.merge_and_unload()
    return model


def get_base_tokenizer():
    tokenizer = AutoTokenizer.from_pretrained(_model_path)
    tokenizer.pad_token = tokenizer.unk_token
    return tokenizer


def _prepare_for_inference(user_input: str, tokenizer):
    prompt_data = []
    if len(_system_prompt):
        prompt_data.append({"role": "system", "content": _system_prompt})
    prompt_data.append({"role": "user", "content": user_input})
    return tokenizer.apply_chat_template(
        prompt_data, tokenize=False, add_generation_prompt=True
    )


prev_model = None
prev_tokenizer = None
def generate_response(user_input: str, model=None, tokenizer=None):
    global prev_model, prev_tokenizer

    if model is None:
        if prev_model is None:
            prev_model = get_base_model()
        model = prev_model

    if tokenizer is None:
        if prev_tokenizer is None:
            prev_tokenizer = get_base_tokenizer()
        tokenizer = prev_tokenizer

    prepared_input = _prepare_for_inference(user_input, tokenizer)
    generation_args = {
        "max_new_tokens": 250,
        "return_full_text": False,
    }
    generation_pipeline = pipeline("text-generation", model=model, tokenizer=tokenizer)
    return generation_pipeline(prepared_input, **generation_args)[0]['generated_text']

_gpt_client: Optional[openai.OpenAI] = None
def load_client():
    global _gpt_client
    if _gpt_client is not None:
        return _gpt_client
    load_dotenv()
    api_key = os.getenv("OPENAI_API_KEY")
    _gpt_client = openai.OpenAI(api_key=api_key, base_url=_litellm_url)
    return _gpt_client

_deepseek_client: Optional[openai.OpenAI] = None
def load_deepseep_client():
    global _deepseek_client
    if _deepseek_client is not None:
        return _deepseek_client
    load_dotenv()
    api_key = os.getenv("DEEPSEEK_API_KEY")
    _deepseek_client = openai.OpenAI(api_key=api_key, base_url=_deepseek_url)
    return _deepseek_client

def generate_openai(user_input: str, use_deepseek: bool = False):
    client = load_client() if not use_deepseek else load_deepseep_client()
    if client is None:
        raise ValueError("Client not loaded")
    response = client.chat.completions.create(
        model=_gpt_model if not use_deepseek else _deepseek_model,
        messages=[
            {"role": "system", "content": get_system_prompt()},
            {"role": "user",
             "content": user_input},
        ]
    )
    return response.choices[0].message.content


def store_results(results: list[str], file_path: str, do_backup: bool = False, do_append: bool = False):
    with open(file_path, 'w' if not do_append else 'a') as file:
        file.write(_result_separator.join(results))

    if do_backup:
        from datetime import datetime
        with open(file_path + '_' + datetime.now().strftime("%Y%m%d%H%M%S") + '.txt', 'w') as file:
            file.write(_result_separator.join(results))


def load_results(file_path: str):
    with open(file_path, 'r') as file:
        return file.read().split(_result_separator)