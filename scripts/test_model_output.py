import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import yaml

# Load config
with open("config/train_config.yaml", "r") as f:
    config = yaml.safe_load(f)

# Use checkpoint-31 for both model and tokenizer
checkpoint_path = "models/raadha-lora/checkpoint-31/"
lora_path = "models/raadha-lora/"

# Load tokenizer and base model from checkpoint-31
print("Loading tokenizer and model from:", checkpoint_path)
tokenizer = AutoTokenizer.from_pretrained(checkpoint_path)
model = AutoModelForCausalLM.from_pretrained(checkpoint_path, torch_dtype=torch.float32, device_map="cpu", low_cpu_mem_usage=True)

# Try to load LoRA adapter
try:
    model = PeftModel.from_pretrained(model, lora_path, is_trainable=False)
    print("Loaded LoRA adapter.")
except Exception as e:
    print(f"Could not load LoRA adapter: {e}")
    print("Continuing with base model only.")

model.eval()

# Simple prompt (no ChatML)
prompt = "Hello! How are you?"
inputs = tokenizer(prompt, return_tensors="pt")
with torch.inference_mode():
    outputs = model.generate(
        inputs.input_ids,
        max_length=128,
        temperature=0.85,
        do_sample=True,
        top_p=0.95,
        top_k=50,
        repetition_penalty=1.1,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
        num_return_sequences=1,
        min_length=10,
        early_stopping=True
    )
response = tokenizer.decode(outputs[0], skip_special_tokens=True)
print("=== RAW MODEL OUTPUT ===")
print(response)
