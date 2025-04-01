from flask import Flask, request, jsonify
import json
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    pipeline
)

app = Flask(__name__)

# Load Configurations
with open('config.json', 'r') as file:
    config = json.load(file)
model_path = config.get("model_path")
checkpoint_path = config.get("checkpoint_path")
system_prompt_path = config.get("system_prompt_path")

# Load Model and Tokenizer
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

tokenizer = AutoTokenizer.from_pretrained(model_path)
tokenizer.pad_token = tokenizer.unk_token

# Set up formatting for inference
with open(system_prompt_path, "r") as f:
    system_prompt = f.read()

def prepare_for_inference(user_input: str, system_prompt: str = system_prompt):
    prompt_data = []
    if len(system_prompt):
        prompt_data.append({"role": "system", "content": system_prompt})
    prompt_data.append({"role": "user", "content": user_input})
    return tokenizer._apply_chat_template(
        prompt_data, tokenize=False, add_generation_prompt=True
    )

# Set up pipeline
generation_args = {
    "max_new_tokens": 150,
    "return_full_text": False,
}
peft_pipeline = pipeline("text-generation", model=model, tokenizer=tokenizer)

@app.route("/generate", methods=["POST"])
def generate():
    data = request.get_json()
    prompt = data.get("prompt")
    if prompt is None:
        return jsonify({"error": "Prompt is required"}), 400
    formatted_input = prepare_for_inference(prompt)
    response = peft_pipeline(formatted_input, **generation_args)
    return jsonify({"response": response[0]["generated_text"]})

if __name__ == "__main__":
    print("Starting server...")
    app.run(port=5000, debug=True)