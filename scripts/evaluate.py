import os
import yaml
import json
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from datasets import load_dataset
from tqdm import tqdm
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

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

def generate_response(model, tokenizer, messages, max_length=2048):
    # Format the conversation
    prompt = format_prompt(messages)
    
    # Tokenize input
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    # Generate response with optimized parameters for more natural and friendly responses
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_length=max_length,
            num_return_sequences=1,
            temperature=0.85,  # Increased for more natural, varied responses
            top_p=0.92,
            top_k=50,
            repetition_penalty=1.2,  # Increased to reduce repetition
            do_sample=True,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
            min_length=30,  # Ensure minimum response length
            max_new_tokens=250,  # Increased for more complete responses
            length_penalty=1.3,  # Increased to favor longer responses
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
    if len(response) < 15:
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
                min_length=40,
                max_new_tokens=300,
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

def evaluate_model(model, tokenizer, eval_dataset):
    results = {
        "responses": [],
        "references": [],
        "metrics": {}
    }
    
    for example in tqdm(eval_dataset, desc="Evaluating"):
        # Get conversation history
        messages = example["messages"][:-1]  # Exclude the last message (reference)
        reference = example["messages"][-1]["content"]
        
        # Generate response
        response = generate_response(model, tokenizer, messages)
        
        # Store results
        results["responses"].append(response)
        results["references"].append(reference)
    
    # Calculate metrics
    # Note: This is a simple implementation. You might want to use more sophisticated metrics
    # like BLEU, ROUGE, or semantic similarity scores
    results["metrics"]["response_length"] = {
        "mean": np.mean([len(r) for r in results["responses"]]),
        "std": np.std([len(r) for r in results["responses"]])
    }
    
    return results

def save_results(results, output_file):
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)

def main():
    # Load configuration
    config = load_config()
    
    # Load model and tokenizer
    model, tokenizer = load_model_and_tokenizer(config)
    
    # Load evaluation dataset
    eval_dataset = load_dataset("json", data_files=config["data"]["eval_file"])["train"]
    
    # Evaluate model
    results = evaluate_model(model, tokenizer, eval_dataset)
    
    # Save results
    output_file = os.path.join(config["output"]["output_dir"], "evaluation_results.json")
    save_results(results, output_file)
    
    # Print summary
    print("\nEvaluation Results:")
    print("------------------")
    print(f"Number of examples evaluated: {len(results['responses'])}")
    print(f"Average response length: {results['metrics']['response_length']['mean']:.2f} characters")
    print(f"Response length std dev: {results['metrics']['response_length']['std']:.2f} characters")
    print(f"\nResults saved to: {output_file}")

if __name__ == "__main__":
    main() 