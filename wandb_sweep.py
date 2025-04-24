import random

import wandb
import os
import json
import torch
from datasets import load_dataset, load_from_disk
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from trl import SFTTrainer, SFTConfig
from peft import LoraConfig
from src.utils.output_formatter import *
from src.utils.model_utils import generate_response, fix_seed, get_finetuned_model
from src.utils.config_loader import load_config
from nervaluate import Evaluator
from tqdm import tqdm
wandb.login()

def _apply_chat_template(sample, tokenizer, include_response=True):
    config = load_config()
    system_prompt = wandb.config.get("system_prompt", config.get("system_prompt"))
    message = []
    if len(system_prompt):
        message.append({"role": "system", "content": system_prompt})
    message.append({"role": "user", "content": sample["user"]})
    if include_response:
        message.append({"role": "assistant", "content": sample["assistant"]})

    # Use the tokenizer's chat template to create formatted text
    message = tokenizer.apply_chat_template(
        message, tokenize=False, add_generation_prompt=False
    )
    return tokenizer(message, padding=True, truncation=True)


def objective():
    # load data and model same way as in `model_finetuning.ipynb`
    # ensure same conditions for every iteration
    fix_seed()

    config = load_config()

    # set up model
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16
    )

    model = AutoModelForCausalLM.from_pretrained(
        config.get("model"),
        device_map="auto",
        trust_remote_code=True,
        quantization_config=bnb_config
    )

    tokenizer = AutoTokenizer.from_pretrained(config.get("model"))
    tokenizer.padding_side = "right"
    model.eval()

    ## Load the datasets
    if os.path.exists("data/train"): # save time by not reprocessing
        processed_train = load_from_disk("data/train")
    else:
        raw_train = load_dataset("json", data_files=config.get("train_dataset_path"), download_mode="force_redownload")[
            "train"]
        processed_train = raw_train.map(
            _apply_chat_template,
            fn_kwargs={"tokenizer": tokenizer},
        )
        processed_train.save_to_disk("data/train")

    if os.path.exists("data/dev"): # save time by not reprocessing
        processed_dev = load_from_disk("data/dev")
    else:
        raw_dev = load_dataset("json", data_files=config.get("dev_dataset_path"), download_mode="force_redownload")[
            "train"]
        processed_dev = raw_dev.map(
            _apply_chat_template,
            fn_kwargs={"tokenizer": tokenizer},
        )
        processed_dev.save_to_disk("data/dev")

    peft_config = LoraConfig(
        r=wandb.config.get("r", 16),
        lora_alpha=wandb.config.get("lora_alpha", 16),
        target_modules=['gate_up_proj', 'base_layer', 'down_proj', 'qkv_proj', 'o_proj'],
        lora_dropout=wandb.config.get("lora_dropout", 0.05),
        bias="none",
        task_type="CAUSAL_LM"
    )

    trainer = SFTTrainer(
        model=model,
        args=SFTConfig(
            per_device_train_batch_size=wandb.config.get("batch_size", 4),
            gradient_accumulation_steps=4,
            num_train_epochs=wandb.config.get("num_train_epochs", 2),
            learning_rate=wandb.config.get("learning_rate", 1e-4),
            max_seq_length=4,
            gradient_checkpointing=True,
            bf16=True,
            optim="adamw_8bit",
            lr_scheduler_type="cosine",
            warmup_ratio=wandb.config.get("warmup_ratio", 0.05),
            logging_steps=50,
            save_strategy="epoch",
            output_dir=config.get("model_dir_path") + "/temp_model",
            eval_accumulation_steps=15,
            per_device_eval_batch_size=1,
        ),
        train_dataset=processed_train,
        eval_dataset=processed_dev,
        peft_config=peft_config,
        tokenizer=tokenizer
    )
    trainer.train()

    trainer.save_model()

    torch.cuda.empty_cache()
    result = trainer.evaluate()
    assert "eval_loss" in result, "'eval_loss' not found in evaluation result"
    wandb.log({"eval/loss": result["eval_loss"]})

    torch.cuda.empty_cache()

    # Returning NER metrics as final objective
    test_sentences = []
    test_true = []
    with open(config.get("test_dataset_path"), 'r') as file:
        testing_data = json.load(file)

    for sample in random.sample(testing_data, 10):
        test_sentences.append(sample["user"])
        test_true.append(transform_to_prodigy(sample["user"], sample["assistant"]))

    model = get_finetuned_model("temp_model", model_dir_path=config.get("model_output_path"))

    test_generated = []
    for sentence in tqdm(test_sentences):
        test_generated.append(generate_response(sentence, model=model, tokenizer=tokenizer, system_prompt=wandb.config.get("system_prompt", config.get("system_prompt"))))

    predicted = []
    for i in range(len(test_generated)):
        predicted_entities = []
        try:
            if config.get("use_nuextract"):
                test_generated[i] = numind_to_default(test_generated[i])
            predicted_entities = transform_to_prodigy(test_sentences[i], test_generated[i])
        except (json.JSONDecodeError, AttributeError, KeyError) as err:
            print(f"Error in transforming generated response: {err}")
        predicted.append(predicted_entities)

    print("\nPredicted:\n\n", predicted)
    print("\nGenerated:\n\n", test_generated)
    evaluator = Evaluator(test_true, predicted, tags=['Disease', 'Chemical'])
    test_results = evaluator.evaluate()[0]
    # calculate geometric mean of f1 scores across evaluation schemas
    output = 1
    for schema in ['ent_type', 'partial', 'strict', 'exact']:
        output *= test_results.get(schema).get('f1')
    output **= (1 / 4)
    wandb.log({"eval/f1": output})
    return output


def main():
    wandb.init(project="finetuning_for_ner")
    eval_loss = objective()
    wandb.log({"eval_loss": eval_loss})


if __name__ == "__main__":
    main()