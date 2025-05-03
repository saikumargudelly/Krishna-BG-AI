import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import yaml
import json
import sys
import logging

logger = logging.getLogger(__name__)

# Initialize global inference manager
_inference_manager = None

def get_inference_manager():
    """Get or create the inference manager singleton."""
    global _inference_manager
    if _inference_manager is None:
        model, tokenizer = load_model_and_tokenizer()
        _inference_manager = InferenceManager(model, "cuda" if torch.cuda.is_available() else "cpu")
    return _inference_manager

def generate_response(prompt, history=None, max_length=4096, temperature=0.7,
                     top_p=0.95, top_k=50, repetition_penalty=1.1, min_length=20,
                     max_new_tokens=1024):
    """Standalone function to generate responses using the inference manager."""
    manager = get_inference_manager()
    return manager.generate_response(
        prompt, history, max_length, temperature, top_p, top_k,
        repetition_penalty, min_length, max_new_tokens
    )

def load_config(config_path: str = "config/train_config.yaml") -> dict:
    """Load training configuration."""
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config

def load_lora_config(config_path: str = "config/lora_config.json") -> dict:
    """Load LoRA configuration."""
    with open(config_path, "r") as f:
        config = json.load(f)
    return config

def load_model_and_tokenizer():
    """Load the base model, tokenizer, and LoRA weights."""
    config = load_config()
    lora_config = load_lora_config()
    
    model = AutoModelForCausalLM.from_pretrained(
        config["model"]["base_model"],
        torch_dtype=torch.float32,
        device_map="cpu",
        low_cpu_mem_usage=True
    )
    
    tokenizer = AutoTokenizer.from_pretrained(
        config["model"]["base_model"],
        padding_side="right",
        use_fast=True
    )
    
    # Add special tokens
    special_tokens = {
        "pad_token": "<|pad|>",
        "eos_token": "<|endoftext|>",
        "bos_token": "<|startoftext|>"
    }
    tokenizer.add_special_tokens(special_tokens)
    model.resize_token_embeddings(len(tokenizer))
    
    try:
        model = PeftModel.from_pretrained(model, config["output"]["output_dir"])
    except Exception as e:
        print(f"Error loading LoRA weights: {e}")
    
    return model, tokenizer

def format_chat_prompt(prompt: str, system_prompt: str = None) -> str:
    """Format the chat prompt with system message if provided."""
    if system_prompt:
        return f"<|system|>\n{system_prompt}\n<|user|>\n{prompt}\n<|assistant|>\n"
    return f"<|user|>\n{prompt}\n<|assistant|>\n"

def clean_response(response: str) -> str:
    """Clean up the model's response."""
    # Remove any system or user messages that might be in the response
    if "<|system|>" in response:
        response = response.split("<|system|>")[-1]
    if "<|user|>" in response:
        response = response.split("<|user|>")[-1]
    
    # Extract assistant's response
    if "<|assistant|>" in response:
        response = response.split("<|assistant|>")[-1]
    
    # Clean up the response
    response = response.strip()
    response = response.replace("<|endoftext|>", "").strip()
    
    # Remove any remaining special tokens
    response = response.replace("<|user|>", "").replace("<|assistant|>", "")
    response = response.replace("<||system||>", "").replace("<|user|}{assistant}|", "")
    response = response.replace("<||assistant---", "").replace("|>", "")
    
    # Remove URLs and technical content - but be less aggressive
    # Only remove URLs if they appear to be incomplete or malformed
    if "http://" in response and not response.endswith("http://"):
        response = response.split("http://")[0].strip()
    if "https://" in response and not response.endswith("https://"):
        response = response.split("https://")[0].strip()
    
    return response

class InferenceManager:
    def __init__(self, model_path, device="cuda" if torch.cuda.is_available() else "cpu"):
        self.device = device
        self.model = AutoModelForCausalLM.from_pretrained(model_path).to(device)
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        
    def generate_response(self, prompt, history=None, max_length=4096, temperature=0.7,
                         top_p=0.95, top_k=50, repetition_penalty=1.1, min_length=20,
                         max_new_tokens=1024):
        try:
            # Prepare input
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
            
            # Generate response
            outputs = self.model.generate(
                **inputs,
                max_length=max_length,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                repetition_penalty=repetition_penalty,
                min_length=min_length,
                max_new_tokens=max_new_tokens,
                pad_token_id=self.tokenizer.eos_token_id,
                do_sample=True
            )
            
            # Decode response
            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Remove prompt from response
            response = response[len(prompt):].strip()
            
            # Check if response is too short
            if len(response) < 10 and len(prompt) > 20:
                logger.warning("Generated response too short, retrying with adjusted parameters")
                return self.generate_response(
                    prompt,
                    history,
                    max_length=max_length,
                    temperature=temperature * 1.2,
                    top_p=top_p,
                    top_k=top_k,
                    repetition_penalty=repetition_penalty * 1.1,
                    min_length=min_length,
                    max_new_tokens=max_new_tokens * 2
                )
            
            return response
            
        except Exception as e:
            logger.error(f"Error generating response: {str(e)}")
            return "I apologize, but I encountered an error generating a response. Please try again."
            
    def __call__(self, prompt, history=None):
        return self.generate_response(prompt, history)

def main():
    """Main function to run the model."""
    try:
        # Load model and tokenizer
        model, tokenizer = load_model_and_tokenizer()
        print("\nModel loaded successfully! You can now start chatting.")
        print("Type 'quit' to exit.\n")
        
        # Interactive chat loop
        while True:
            user_input = input("You: ")
            if user_input.lower() == 'quit':
                break
                
            response = InferenceManager(model, "cuda" if torch.cuda.is_available() else "cpu")(user_input)
            print(f"Assistant: {response}\n")
            
    except Exception as e:
        print(f"Error: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main() 