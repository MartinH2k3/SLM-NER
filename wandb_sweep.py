import wandb
import os
import json
import torch
from datasets import load_dataset, load_from_disk
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from trl import SFTTrainer, SFTConfig
from peft import LoraConfig

wandb.login()

def _apply_chat_template(sample, tokenizer, include_response=True):
    with open('config.json', 'r') as file:
        config = json.load(file)
    with open(config.get("system_prompt_path"), "r") as f:
        system_prompt = f.read()
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



# set up the sweep based on https://docs.wandb.ai/guides/sweeps/walkthrough/
_sweep_config = {
    "name": "initial_sweep",
    "method": "bayes",
    "metric": {
        "name": "eval/loss",
        "goal": "minimize"
    },
    "parameters": {
        "learning_rate": {
            "min": 1e-5,
            "max": 1e-3,
        },
        "batch_size": {
            "values": [1, 2, 4],
        },
        "num_train_epochs": {
            "values": [1, 2, 3, 4],
        },
        "warmup_ratio": {
            "min": 0.1,
            "max": 0.9,
        },
        "r": {
            "values": [2, 4, 8, 16],
        },
        "lora_alpha": {
            "values": [16, 32, 64],
        },
        "lora_dropout": {
            "min": 0.1,
            "max": 0.5,
        },
    }
}

def objective(sweep_config):
    # load data and model same way as in `model_finetuning.ipynb`

    with open('config.json', 'r') as file:
        config = json.load(file)

    ## set up model
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16
    )

    model = AutoModelForCausalLM.from_pretrained(
        config.get("model_path"),
        device_map="auto",
        trust_remote_code=True,
        quantization_config=bnb_config
    )

    tokenizer = AutoTokenizer.from_pretrained(config.get("model_path"))
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
        r=sweep_config.r,
        lora_alpha=sweep_config.lora_alpha,
        target_modules=['gate_up_proj', 'base_layer', 'down_proj', 'qkv_proj', 'o_proj'],
        lora_dropout=sweep_config.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM"
    )

    trainer = SFTTrainer(
        model=model,
        args=SFTConfig(
            per_device_train_batch_size=sweep_config.batch_size,
            gradient_accumulation_steps=4,
            num_train_epochs=sweep_config.num_train_epochs,
            learning_rate=sweep_config.learning_rate,
            max_seq_length=4,
            gradient_checkpointing=True,
            bf16=True,
            optim="adamw_8bit",
            lr_scheduler_type="cosine",
            warmup_ratio=sweep_config.warmup_ratio,
            logging_steps=50,
            save_strategy="epoch",
            output_dir="./temp_checkpoint_dir"
        ),
        train_dataset=processed_train,
        eval_dataset=processed_dev,
        peft_config=peft_config,
        tokenizer=tokenizer
    )
    training_results = trainer.train()

    result = trainer.evaluate()
    print(result)
    # if eval loss isn't there, something is wrong with the trainer and no reason to go on
    assert "eval_loss" in result, "'eval_loss' not found in evaluation result"

    wandb.log({"loss": result["eval_loss"]})

    return result["eval_loss"]


def main():
    wandb.init(project="finetuning_for_ner")
    eval_loss = objective(wandb.config)
    wandb.log({"eval_loss": eval_loss})


if __name__ == "__main__":
    main()