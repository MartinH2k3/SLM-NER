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
    with open(config.get("system_prompt"), "r") as f:
        system_prompt = f.read()
    wandb.config.get("system_prompt", system_prompt)
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
    eval_loss = objective()
    wandb.log({"eval_loss": eval_loss})


if __name__ == "__main__":
    main()