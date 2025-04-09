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
from utils.numind_utils import *
from utils.config_loader import load_config


_config = load_config()
_system_prompt_path = _config.get("system_prompt_path")
with open(_system_prompt_path, "r") as f:
    _system_prompt = f.read()
_model_path = _config.get("model_path")
_checkpoint_path = _config.get("checkpoint_path")
_deepseek_model = "deepseek-chat"
_result_separator = _config.get("result_separator")

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
    return tokenizer._apply_chat_template(
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

_openai_client: Optional[openai.OpenAI] = None
def load_openai_client(provider: str = "litellm"):
    global _openai_client
    if _openai_client is not None:
        return _openai_client
    load_dotenv()
    api_key = os.getenv(f"{provider.upper()}_API_KEY")
    _openai_client = openai.OpenAI(api_key=api_key, base_url=_config.get(f"{provider.lower()}_url"))
    return _openai_client


def generate_openai(user_input: str, provider: str = "litellm", model_name: str = "gpt-4o-mini"):
    client = load_openai_client(provider)
    if client is None:
        raise ValueError("Client not loaded")
    response = client.chat.completions.create(
        model= model_name,
        messages=[
            {"role": "system", "content": get_system_prompt()},
            {"role": "user",
             "content": user_input},
        ]
    )
    return response.choices[0].message.content


numind_model, numind_tokenizer = None, None
def get_numind_model(model_name = "numind/NuExtract-2-2B"):
    global numind_model, numind_tokenizer
    if numind_model is None:
        numind_model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True,
                                             torch_dtype=torch.bfloat16,
                                             attn_implementation="flash_attention_2" # we recommend using flash attention
                                            ).to("cuda")
    if numind_tokenizer is None:
        numind_tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True, padding_side='left')
    return numind_model, numind_tokenizer


def generate_numind(user_input: str, model_name = "numind/NuExtract-2-2B"):
    model, tokenizer = get_numind_model(model_name)
    template = """{"Chemical": ["verbatim-string"], "Disease": ["verbatim-string"]}"""
    examples = [
        {
            "input": "Naloxone reverses the antihypertensive effect of clonidine. In unanesthetized, spontaneously hypertensive rats the decrease in blood pressure and heart rate produced by intravenous clonidine, 5 to 20 micrograms/kg, was inhibited or reversed by nalozone, 0.2 to 2 mg/kg.",
            "output": """[Chemicals: ["Naloxone", "clonidine", clonidine, nalozone], Diseases: [hypertensive]]"""
        }
    ]
    input_messages = [construct_message(user_input, template, examples)]

    input_content = prepare_inputs(
        messages=input_messages,
        image_paths=[],
        tokenizer=tokenizer,
    )

    generation_config = {"do_sample": False, "num_beams": 1, "max_new_tokens": 2048}

    with torch.no_grad():
        result = nuextract_generate(
            model=model,
            tokenizer=tokenizer,
            prompts=input_content['prompts'],
            pixel_values_list=input_content['pixel_values_list'],
            num_patches_list=input_content['num_patches_list'],
            generation_config=generation_config
        )
    return result


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