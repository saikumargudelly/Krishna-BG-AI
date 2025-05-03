import os
import yaml
import json
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel, PeftConfig

def load_config():
    with open("config/train_config.yaml", "r") as f:
        return yaml.safe_load(f)

def load_model_and_tokenizer(config):
    # Load base model and tokenizer
    base_model = AutoModelForCausalLM.from_pretrained(
        config["model"]["base_model"],
        torch_dtype=torch.float16,
        device_map="auto"
    )
    tokenizer = AutoTokenizer.from_pretrained(
        config["model"]["base_model"],
        padding_side="right",
        use_fast=True
    )
    
    # Load LoRA adapter
    model = PeftModel.from_pretrained(
        base_model,
        config["output"]["output_dir"],
        torch_dtype=torch.float16,
        device_map="auto"
    )
    
    return model, tokenizer

def format_prompt(messages):
    formatted_text = ""
    for msg in messages:
        role = msg["role"]
        content = msg["content"]
        formatted_text += f"<|{role}|>\n{content}\n"
    return formatted_text

def generate_response(model, tokenizer, messages, max_length=4096):
    # Format the conversation
    prompt = format_prompt(messages)
    
    # Tokenize input
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    # Generate response with optimized parameters for better quality
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_length=max_length,
            num_return_sequences=1,
            temperature=0.8,  # Slightly increased for more natural responses
            top_p=0.92,
            top_k=50,
            repetition_penalty=1.15,  # Increased to reduce repetition
            do_sample=True,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
            min_length=20,  # Ensure minimum response length
            max_new_tokens=1024,  # Increased from 200 to 1024 for longer responses
            length_penalty=1.2,  # Favor longer responses
            no_repeat_ngram_size=3,  # Prevent repetitive text
            early_stopping=True
        )
    
    # Decode and return response
    response = tokenizer.decode(outputs[0], skip_special_tokens=False)
    
    # Clean up the response to extract only the assistant's part
    if "<|assistant|>" in response:
        response = response.split("<|assistant|>")[-1].strip()
    
    # Remove any remaining special tokens
    special_tokens = ["<|user|>", "<|system|>", "<|endoftext|>", "<|startoftext|>"]
    for token in special_tokens:
        response = response.replace(token, "")
    
    # Check if response is too short
    if len(response) < 10:
        # Try again with different parameters if response is too short
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_length=max_length,
                num_return_sequences=1,
                temperature=0.9,
                top_p=0.95,
                top_k=60,
                repetition_penalty=1.1,
                do_sample=True,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
                min_length=30,
                max_new_tokens=2048,  # Increased from 250 to 2048 for longer responses
                length_penalty=1.5,
                no_repeat_ngram_size=2,
                early_stopping=True
            )
        response = tokenizer.decode(outputs[0], skip_special_tokens=False)
        
        # Clean up the response again
        if "<|assistant|>" in response:
            response = response.split("<|assistant|>")[-1].strip()
        
        for token in special_tokens:
            response = response.replace(token, "")
    
    return response

def main():
    # Load configuration
    config = load_config()
    
    # Load model and tokenizer
    model, tokenizer = load_model_and_tokenizer(config)
    
    print("Raadhe AI is ready to chat! Type 'quit' to exit.")
    print("----------------------------------------")
    
    # Initialize conversation
    messages = [
        {
            "role": "system",
            "content": "You are Raadhe, a warm, kind, and emotionally intelligent AI friend. You're empathetic, supportive, and always ready to listen. You use emojis naturally and speak in a friendly, conversational way."
        }
    ]
    
    while True:
        # Get user input
        user_input = input("\nYou: ").strip()
        
        if user_input.lower() == "quit":
            print("\nGoodbye! Take care! 💛")
            break
        
        # Add user message
        messages.append({"role": "user", "content": user_input})
        
        # Generate response
        response = generate_response(model, tokenizer, messages)
        
        # Add assistant response to messages
        messages.append({"role": "assistant", "content": response})
        
        # Print response
        print(f"\nRaadhe: {response}")

if __name__ == "__main__":
    main() 