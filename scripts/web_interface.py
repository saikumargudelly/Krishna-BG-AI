import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import yaml
import json
import gradio as gr
import re
import os
import sys
from typing import Optional, Tuple

def load_config(config_path: str = "config/train_config.yaml") -> dict:
    """Load training configuration."""
    try:
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)
        return config
    except Exception as e:
        print(f"Error loading config: {e}")
        sys.exit(1)

def load_system_prompt(config_path: str = "config/system_prompt.yaml") -> str:
    """Load system prompt from config."""
    try:
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)
        return config.get("system_prompt", "")
    except Exception as e:
        print(f"Error loading system prompt: {e}")
        return ""

def load_lora_config(config_path: str = "config/lora_config.json") -> dict:
    """Load LoRA configuration."""
    try:
        with open(config_path, "r") as f:
            config = json.load(f)
        return config
    except Exception as e:
        print(f"Error loading LoRA config: {e}")
        return {}

def load_model_and_tokenizer() -> Tuple[AutoModelForCausalLM, AutoTokenizer]:
    """Load the base model, tokenizer, and LoRA weights."""
    try:
        config = load_config()
        lora_config = load_lora_config()
        
        # Load base model
        model = AutoModelForCausalLM.from_pretrained(
            config["model"]["base_model"],
            torch_dtype=torch.float32,
            device_map="cpu",
            low_cpu_mem_usage=True
        )
        
        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(
            config["model"]["base_model"],
            padding_side="right",
            use_fast=True
        )
        
        # Add special tokens
        special_tokens = {
            "pad_token": " ",
            "eos_token": " ",
            "bos_token": ""
        }
        tokenizer.add_special_tokens(special_tokens)
        model.resize_token_embeddings(len(tokenizer))
        
        # Try to load LoRA weights with error handling
        try:
            model = PeftModel.from_pretrained(
                model, 
                config["output"]["output_dir"],
                is_trainable=False  # Set to False for inference
            )
            print("Successfully loaded LoRA weights")
        except Exception as e:
            print(f"Warning: Could not load LoRA weights: {e}")
            print("Continuing with base model only")
        
        return model, tokenizer
    except Exception as e:
        print(f"Error loading model and tokenizer: {e}")
        sys.exit(1)

def format_chat_prompt(prompt: str, system_prompt: Optional[str] = None) -> str:
    """Format the chat prompt with system message if provided."""
    if system_prompt:
        return f"<|system|>\n{system_prompt}\n\n{prompt}\n<|assistant|>\n"
    return f"\n{prompt}\n<|assistant|>\n"

def clean_response(response: str) -> str:
    """Clean up the model's response to ensure only the assistant's reply is returned."""
    try:
        # Always extract only the assistant's reply
        if "<|assistant|>" in response:
            response = response.split("<|assistant|>")[-1]
        
        # Remove everything after a user prompt token (hard stop)
        user_prompt_patterns = [
            r"<\|user\|>", r"<user>", r"<USER>", r"<\\?\\?user\\?\\?>",
            r"User:", r"user:", r"</s>", r"<\|user\|"
        ]
        for pattern in user_prompt_patterns:
            match = re.search(pattern, response)
            if match:
                response = response[:match.start()]
        
        # Remove all special tokens and formatting
        special_tokens = [
            " ", "<|system|>", " ", "",
            "<||system||>", "<|user|}{assistant|>", "<||assistant---", "|>",
            "<||user|>", "<||assistant|>", "---", "||", "</s>", "<s>",
            " ", " ", "<|bos|>", "<|eos|>"
        ]
        for token in special_tokens:
            response = response.replace(token, "")
        
        # Remove any remaining angle-bracketed tokens
        response = re.sub(r"<\\|.*?\\|>", "", response)
        response = re.sub(r"<.*?>", "", response)
        
        # Remove URLs and technical content - but be less aggressive
        if "http://" in response and not response.endswith("http://"):
            response = response.split("http://")[0].strip()
        if "https://" in response and not response.endswith("https://"):
            response = response.split("https://")[0].strip()
        
        # Remove any remaining technical artifacts
        response = re.sub(r"\s+", " ", response)  # Normalize whitespace
        response = re.sub(r"\.{3,}", "...", response)  # Normalize ellipsis
        
        # Remove any responses that seem to be system prompts or technical content
        if re.match(r"(?i)as (an )?(ai|assistant|language model)", response.strip()):
            return "Bestie... don't worry about all that techy stuff . Just tell me what's on your heart "
        
        # Final strip
        response = response.strip()
        return response
    except Exception as e:
        print(f"Error cleaning response: {e}")
        return "Hey bestie! I'm having a little trouble right now. Could you try asking me again? "

def generate_response(model, tokenizer, prompt: str, max_length: int = 2048) -> str:
    """Generate a response for the given prompt."""
    try:
        # Load system prompt
        system_prompt = load_system_prompt()
        
        # Format the prompt
        formatted_prompt = format_chat_prompt(prompt, system_prompt)
        
        # Tokenize the prompt
        inputs = tokenizer(formatted_prompt, return_tensors="pt")
        
        # Generate response with optimized parameters for better quality and length
        with torch.no_grad():
            outputs = model.generate(
                inputs.input_ids,
                max_length=min(max_length, 2048),  # Increased for longer responses
                temperature=0.85,  # Slightly increased for more creative responses
                do_sample=True,
                top_p=0.95,  # Increased for more diverse responses
                top_k=100,   # Increased for better quality
                repetition_penalty=1.2,  # Increased to avoid repetition
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
                num_return_sequences=1,
                max_time=15.0,  # Increased for better quality
                no_repeat_ngram_size=4,  # Increased to avoid repetition
                min_length=50,  # Increased minimum length
                max_new_tokens=1024,  # Increased for longer responses
                length_penalty=1.5,  # Increased to encourage longer responses
                early_stopping=True,
                use_cache=True,
                num_beams=1
            )
        
        # Decode and clean up the response
        response = tokenizer.decode(outputs[0], skip_special_tokens=False)
        response = clean_response(response)
        
        # Additional validation checks
        if not response or all(c in "<|>{}" for c in response):
            return "Hey bestie! I'm not sure I understood that completely. Could you tell me again in a different way? "
        
        # Check if response is too short
        if len(response) < 20:
            return "Hey sweetie! I'd love to hear more about that. Could you tell me a bit more? "
        
        # Check if response is too long and needs trimming
        if len(response) > 1000:
            # Find a good stopping point near 1000 characters
            last_period = response[:1000].rfind('.')
            if last_period > 800:  # Only trim if we can find a good stopping point
                response = response[:last_period + 1]
            else:
                response = response[:1000] + "..."
        
        # Check for any remaining special tokens or formatting
        if any(token in response for token in ["<|", "|>", "<user>", "<system>", "<assistant>"]):
            return "Hey bestie! I'm having a little trouble with that. Could you try asking me again? "
        
        return response
    except Exception as e:
        print(f"Error generating response: {e}")
        return "Oh no! I'm having a little trouble right now. Could you try asking me again? "

def create_interface():
    """Create and launch the Gradio interface with split screen for augmentation."""
    try:
        model, tokenizer = load_model_and_tokenizer()

        # Import augmentation utilities
        try:
            from scripts.utils import detect_emotion, final_response
        except ImportError:
            import sys, os
            sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
            from scripts.utils import detect_emotion, final_response

        def chat(message: str, history: list) -> str:
            """Chat function for the Gradio interface (returns only model response)."""
            if not message or not message.strip():
                return "Please enter a message to chat with me! "
            response = generate_response(model, tokenizer, message)
            return response

        def augment(message: str, model_response: str) -> str:
            """Compute emotion+sloka augmentation from user message and model response."""
            emotion_tag = detect_emotion(message)
            return final_response(model_response, emotion_tag)

        # Gradio UI components
        with gr.Blocks(theme="soft") as interface:
            gr.Markdown("# Chat with Krish \nHi Arjuna! I'm Krish, your supportive and wise best friend. Let's chat about anything! ")
            with gr.Row():
                with gr.Column(scale=2):
                    chatbox = gr.Chatbot(label="Krish's Response")
                    user_input = gr.Textbox(placeholder="Type your message here, Arjuna...", label="Your Message (Arjuna)")
                    send_btn = gr.Button("Send")
                with gr.Column(scale=1):
                    gr.Markdown("### Emotion + Sloka Augmentation")
                    aug_output = gr.Markdown("", elem_id="sloka-augmentation")

            def on_send(message, history):
                # Defensive: Ensure history is a list of tuples for chatbox
                if not isinstance(history, list):
                    history = []
                # Get the model response
                response = chat(message, history)
                # Update chat history (append user and bot turns)
                new_history = history + [(message, response)]
                # Augmentation output
                aug = augment(message, response)
                # Clear user input after send
                return new_history, "", aug

            def on_augment(message, model_response):
                return augment(message, model_response)

            # When send is clicked, update chat and augmentation
            send_btn.click(
                fn=on_send,
                inputs=[user_input, chatbox],
                outputs=[chatbox, user_input, aug_output],
                queue=False
            )
            # Also update augmentation when chat updates
            user_input.submit(
                fn=on_send,
                inputs=[user_input, chatbox],
                outputs=[chatbox, user_input, aug_output],
                queue=False
            )

        return interface
    except Exception as e:
        print(f"Error creating interface: {e}")
        sys.exit(1)

if __name__ == "__main__":
    try:
        interface = create_interface()
        
        # Try multiple ports
        ports = range(3000, 3010)  # Try ports 3000-3009
        for port in ports:
            try:
                interface.launch(
                    server_name="127.0.0.1",
                    server_port=port,
                    share=True,
                    show_error=True
                )
                break  # If successful, break the loop
            except OSError as e:
                if port == ports[-1]:  # If this was the last port to try
                    print(f"Error: Could not find an available port in range {ports[0]}-{ports[-1]}")
                    print("Please try one of these solutions:")
                    print("1. Close other applications using these ports")
                    print("2. Set a different port using the GRADIO_SERVER_PORT environment variable")
                    print("3. Specify a different port in the launch() parameters")
                    sys.exit(1)
                continue  # Try next port
    except Exception as e:
        print(f"Fatal error: {e}")
        sys.exit(1) 