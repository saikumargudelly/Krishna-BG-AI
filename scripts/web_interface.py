import sys
import os
import logging
from functools import lru_cache
from typing import Optional, Tuple, Dict, Any, List, TYPE_CHECKING
import json
import random

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import yaml
import gradio as gr
import re
from sentence_transformers import SentenceTransformer, util

# Configure logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

# Type checking imports
if TYPE_CHECKING:
    from scripts.rag_manager import RAGManager

# RAG Integration
try:
    from scripts.rag_manager import RAGManager
    rag_enabled = True
    rag_manager = RAGManager()
except ImportError as e:
    logging.warning(f"RAG integration failed: {e}")
    rag_enabled = False
    rag_manager = None
except Exception as e:
    logging.error(f"Unexpected error during RAG initialization: {e}")
    rag_enabled = False
    rag_manager = None

# --- Device selection ---
device = (
    "mps" if torch.backends.mps.is_available() else
    "cuda" if torch.cuda.is_available() else
    "cpu"
)

# --- Load Gita Slokas for Context-Aware Augmentation ---
SLOKA_LIST = []
SLOKA_PATH = os.path.join("data", "gita_slokas.json")
if os.path.exists(SLOKA_PATH):
    with open(SLOKA_PATH, "r", encoding="utf-8") as f:
        SLOKA_LIST = json.load(f)
else:
    print(f"Warning: {SLOKA_PATH} not found. Sloka augmentation will not work.")

# --- Improved Emotion/Tag Detection ---
def detect_emotion(message: str) -> str:
    msg = message.lower()
    if any(word in msg for word in ["sad", "down", "depressed", "unhappy"]):
        return "sad"
    if any(word in msg for word in ["angry", "mad", "frustrated"]):
        return "angry"
    if any(word in msg for word in ["happy", "joy", "excited"]):
        return "happy"
    if any(word in msg for word in ["stress", "anxiety", "worried"]):
        return "stress"
    if any(word in msg for word in ["motivate", "inspire", "encourage"]):
        return "motivation"
    if any(word in msg for word in ["work", "duty", "action"]):
        return "duty"
    # Add more as needed
    return "neutral"

# --- Context-Aware Sloka Selection ---
def get_relevant_sloka(emotion: str) -> dict:
    # Only consider dicts
    matches = [sloka for sloka in SLOKA_LIST if isinstance(sloka, dict) and emotion in sloka.get("tags", [])]
    if matches:
        return random.choice(matches)
    fallback = [sloka for sloka in SLOKA_LIST if isinstance(sloka, dict)]
    return random.choice(fallback) if fallback else None

def final_response(message: str, model_response: str) -> str:
    # Dummy: just returns the model response
    return model_response

class ModelManager:
    _instance = None
    _initialized = False

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(ModelManager, cls).__new__(cls)
        return cls._instance

    def __init__(self):
        if not self._initialized:
            self.model = None
            self.tokenizer = None
            self.encoder = None
            self.config = None
            self.lora_config = None
            self.system_prompt = None
            self._initialized = True

    def load_configs(self):
        try:
            with open("config/train_config.yaml", "r") as f:
                self.config = yaml.safe_load(f)
            with open("config/lora_config.json", "r") as f:
                self.lora_config = json.load(f)
            with open("config/system_prompt.yaml", "r") as f:
                self.system_prompt = yaml.safe_load(f).get("system_prompt", "")
        except Exception as e:
            logging.error(f"Failed to load configurations: {e}")
            raise

    def initialize_model(self):
        if self.model is not None:
            return

        if not self.config:
            self.load_configs()

        logging.info(f"Loading base model: {self.config['model']['base_model']}")
        
        try:
            self.model = AutoModelForCausalLM.from_pretrained(
                self.config["model"]["base_model"],
                torch_dtype=torch.float16 if device != "cpu" else torch.float32,
                device_map=device,
                low_cpu_mem_usage=True
            )
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.config["model"]["base_model"],
                padding_side="right",
                use_fast=True
            )
            self.tokenizer.add_special_tokens({"pad_token": "[PAD]", "eos_token": "</s>", "bos_token": "<s>"})
            self.model.resize_token_embeddings(len(self.tokenizer))

            try:
                lora_path = self.config["output"]["output_dir"]
                logging.info(f"Loading LoRA adapter from: {lora_path}")
                self.model = PeftModel.from_pretrained(self.model, lora_path, is_trainable=False)
                logging.info("Successfully loaded LoRA weights.")
            except Exception as e:
                logging.error(f"Warning: Could not load LoRA weights: {e}")
                logging.info("Continuing with base model only.")

            self.model.eval()
            self.encoder = SentenceTransformer("all-MiniLM-L6-v2")
            self.model.to(device)
        except Exception as e:
            logging.error(f"Failed to initialize model: {e}")
            raise

# Initialize model manager
model_manager = ModelManager()

@lru_cache(maxsize=100)
def semantic_relevance(response: str, user_query: str, threshold: float = 0.4) -> bool:
    try:
        if not model_manager.encoder:
            model_manager.initialize_model()
        emb_query = model_manager.encoder.encode(user_query, convert_to_tensor=True)
        emb_resp = model_manager.encoder.encode(response, convert_to_tensor=True)
        sim = float(util.pytorch_cos_sim(emb_query, emb_resp))
        return sim >= threshold
    except Exception:
        return False

def format_chat_prompt(prompt: str, system_prompt: Optional[str] = None) -> str:
    if system_prompt:
        return f"{system_prompt}\nUser: {prompt}\nAssistant:"
    return f"User: {prompt}\nAssistant:"

def build_prompt(messages: List[Dict[str, str]], system_prompt: str = None, max_turns: int = 3) -> str:
    """Build a prompt from conversation history."""
    prompt = []
    
    # Add system prompt first if provided
    if system_prompt:
        prompt.append(f"<|system|>\n{system_prompt}\n")
    
    # Only keep the last max_turns*2 messages
    messages = messages[-max_turns*2:]
    
    # Add conversation history
    for msg in messages:
        role = msg.get("role", "").lower()
        content = msg.get("content", "").strip()
        if role == "user" and content:
            prompt.append(f"<|user|>\n{content}")
        elif role == "assistant" and content:
            prompt.append(f"<|assistant|>\n{content}")
    
    # Always end with assistant cue for generation
    prompt.append("<|assistant|>\n")
    
    return "\n".join(prompt)

def clean_response(response: str) -> str:
    # First remove any system prompt echoes
    system_patterns = [
        r"You are Krishna.*?CRITICAL INSTRUCTIONS:.*?(?:\d+\.)+",
        r"<\|system\|>.*?CRITICAL INSTRUCTIONS:.*?(?:\d+\.)+",
        r"You are.*?CRITICAL INSTRUCTIONS:.*?(?:\d+\.)+"
    ]
    for pattern in system_patterns:
        response = re.sub(pattern, '', response, flags=re.DOTALL)
    
    # Stop at any special tag
    stop_tokens = ["<|user|>", "<|system|>", "<|assistant|>"]
    for token in stop_tokens:
        if token in response:
            response = response.split(token)[0]
    
    # Remove excessive whitespace
    response = response.strip()
    
    # Only keep the first 2-3 sentences
    sentences = re.split(r'(?<=[.!?]) +', response)
    response = " ".join(sentences[:3])
    
    # Validate response
    if not response or len(response.strip()) < 5:
        return "Hey bestie! I'd love to hear more about that. Could you tell me a bit more?"
    
    # Check for remaining system prompt fragments
    if any(pattern in response.lower() for pattern in ["you are", "critical instructions", "system prompt"]):
        return "Hey sweetie! I'm having a little trouble with that. Could you try asking me again?"
    
    return response

def generate_response(
    message: str,
    history: List[Tuple[str, str]],
    system_prompt: str = None,
    max_new_tokens: int = 512,
    temperature: float = 0.7,
    top_p: float = 0.9,
    top_k: int = 40,
    repetition_penalty: float = 1.1,
    rag_manager: Optional["RAGManager"] = None
) -> str:
    """Generate a response using the model with RAG enhancement."""
    try:
        messages = []
        for user_msg, assistant_msg in history:
            if user_msg.strip():
                messages.append({"role": "user", "content": user_msg})
            if assistant_msg.strip():
                messages.append({"role": "assistant", "content": assistant_msg})
        if message.strip():
            messages.append({"role": "user", "content": message})
        if rag_manager:
            relevant_response = rag_manager.get_relevant_response(message, messages)
            if relevant_response:
                return relevant_response
        prompt = build_prompt(messages, system_prompt)
        if rag_manager:
            enhanced_messages = rag_manager.enhance_prompt(message, messages)
            prompt = build_prompt(enhanced_messages, system_prompt)
        inputs = model_manager.tokenizer(prompt, return_tensors="pt", padding=True)
        input_ids = inputs["input_ids"].to(device)
        attention_mask = inputs["attention_mask"].to(device)
        with torch.no_grad():
            outputs = model_manager.model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                repetition_penalty=repetition_penalty,
                pad_token_id=model_manager.tokenizer.eos_token_id,
                do_sample=True
            )
        response = model_manager.tokenizer.decode(outputs[0][input_ids.shape[1]:], skip_special_tokens=True)
        cleaned_response = clean_response(response)
        if not cleaned_response or not isinstance(cleaned_response, str):
            return "I apologize, but I encountered an error while generating a response. Please try again."
        return cleaned_response
    except Exception as e:
        logging.error(f"Error generating response: {str(e)}")
        return "I apologize, but I encountered an error while generating a response. Please try again."

def format_chatml(messages: List[Dict[str, str]], system_prompt: str = None) -> str:
    formatted = []
    if system_prompt:
        formatted.append(f"<|system|>\n{system_prompt}")
    for msg in messages:
        role = msg.get("role", "").lower()
        content = msg.get("content", "").strip()
        if role == "user":
            formatted.append(f"<|user|>\n{content}")
        elif role == "assistant":
            formatted.append(f"<|assistant|>\n{content}")
    # Always end with assistant cue for generation
    formatted.append("<|assistant|>\n")
    return "\n".join(formatted)

def chat(message: str, history: List[Tuple[str, str]], system_prompt: str = None) -> Tuple[str, str]:
    """Handle chat interaction with RAG enhancement."""
    try:
        response = generate_response(
            message=message,
            history=history,
            system_prompt=system_prompt,
            rag_manager=rag_manager
        )
        # Sloka augmentation
        try:
            emotion_tag = detect_emotion(message)
            sloka = get_relevant_sloka(emotion_tag)
            if isinstance(sloka, dict):
                meaning = sloka.get('meaning', '')
                if len(meaning) > 300:
                    meaning = meaning[:300] + '...'
                sloka_text = f"Sloka {sloka.get('chapter', '?')}.{sloka.get('verse', '?')}\n{sloka.get('sloka', '')}\nMeaning: {meaning}"
            else:
                sloka_text = "(No relevant sloka found for your emotion, but I'm here for you!)"
        except Exception as e:
            logging.error(f"Error in sloka augmentation: {str(e)}")
            sloka_text = "(No sloka available)"
        return response, sloka_text
    except Exception as e:
        logging.error(f"Error in chat: {str(e)}")
        return "I apologize, but I encountered an error. Please try again.", "(No sloka available)"

# --- Custom Gradio Blocks UI ---
with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown("# KRISH AI Chat\nChat with KRISH AI - Your friendly and helpful assistant!")
    with gr.Row():
        with gr.Column(scale=2):
            chatbot = gr.Chatbot(label="Krish's Response")
            user_input = gr.Textbox(placeholder="Type your message here...", label="Your Message")
            send_btn = gr.Button("Submit")
        with gr.Column(scale=1):
            gr.Markdown("### Emotion + Sloka Augmentation")
            sloka_output = gr.Markdown("", elem_id="sloka-augmentation")
    def on_send(message, history):
        if not isinstance(history, list):
            history = []
        response, sloka = chat(message, history)
        new_history = history + [(message, response)]
        return new_history, "", sloka
    send_btn.click(fn=on_send, inputs=[user_input, chatbot], outputs=[chatbot, user_input, sloka_output], queue=False)
    user_input.submit(fn=on_send, inputs=[user_input, chatbot], outputs=[chatbot, user_input, sloka_output], queue=False)

def find_available_port(start_port: int = 3000, end_port: int = 3010) -> int:
    import socket
    for port in range(start_port, end_port + 1):
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.bind(('127.0.0.1', port))
                return port
        except OSError:
            continue
    raise OSError("No available ports found in the specified range")

if __name__ == "__main__":
    try:
        # Initialize model manager first
        model_manager.load_configs()
        model_manager.initialize_model()
        
        # Launch the interface
        demo.launch(share=True)
    except Exception as e:
        logging.error(f"Failed to start interface: {e}")
        sys.exit(1)