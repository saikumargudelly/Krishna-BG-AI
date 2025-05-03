import os
import json
import yaml
import torch
import re
import logging
import functools
from typing import List, Dict, Any, Optional, Union, Tuple
from transformers import PreTrainedTokenizer
from pathlib import Path
import random

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Cache for file loading to avoid repeated disk reads
_file_cache = {}

def load_yaml(file_path: str) -> Dict[str, Any]:
    """Load a YAML configuration file with caching."""
    if file_path in _file_cache:
        return _file_cache[file_path]
    try:
        with open(file_path, "r") as f:
            config = yaml.safe_load(f)
            _file_cache[file_path] = config
            return config
    except Exception as e:
        logger.error(f"YAML Load Error: {file_path} -> {e}")
        raise

def load_json(file_path: str) -> Dict[str, Any]:
    """Load a JSON file with caching."""
    if file_path in _file_cache:
        return _file_cache[file_path]
    try:
        with open(file_path, "r") as f:
            data = json.load(f)
            _file_cache[file_path] = data
            return data
    except Exception as e:
        logger.error(f"JSON Load Error: {file_path} -> {e}")
        raise

def save_json(data: Dict[str, Any], file_path: str, indent: int = 2) -> None:
    """Save data to a JSON file with error handling."""
    try:
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, "w") as f:
            json.dump(data, f, indent=indent)
        _file_cache[file_path] = data
    except Exception as e:
        logger.error(f"Save Error: {file_path} -> {e}")
        raise

def format_chatml(messages: List[Dict[str, str]]) -> str:
    """Format messages in ChatML structure."""
    if not isinstance(messages, list) or not all(isinstance(m, dict) for m in messages):
        raise TypeError("Expected a list of dicts with 'role' and 'content'.")
    
    formatted = []
    for msg in messages:
        role = msg.get("role", "").strip().lower()
        content = msg.get("content", "").strip()
        if not role or not content:
            continue
        formatted.append(f"<|{role}|>\n{content}\n")

    if not formatted or not formatted[-1].endswith("<|assistant|>\n"):
        formatted.append("<|assistant|>\n")

    return "".join(formatted)

def parse_chatml(text: str) -> List[Dict[str, str]]:
    """Parse ChatML formatted text into a list of messages."""
    if not text:
        return []

    messages = []
    current_role = None
    current_content = []

    lines = text.split("\n")
    for line in lines:
        if line.startswith("<|") and line.endswith("|>"):
            if current_role and current_content:
                messages.append({
                    "role": current_role,
                    "content": "\n".join(current_content).strip()
                })
            current_role = line[2:-2]
            current_content = []
        else:
            current_content.append(line)

    if current_role and current_content:
        messages.append({
            "role": current_role,
            "content": "\n".join(current_content).strip()
        })

    return messages

def prepare_input_for_model(
    tokenizer: PreTrainedTokenizer,
    messages: List[Dict[str, str]],
    max_length: Optional[int] = None,
    padding: bool = True,
    truncation: bool = True
) -> Dict[str, torch.Tensor]:
    """Prepare input for the model by tokenizing and formatting messages."""
    if not messages:
        raise ValueError("Messages list cannot be empty")

    formatted_text = format_chatml(messages)

    try:
        return tokenizer(
            formatted_text,
            return_tensors="pt",
            max_length=max_length,
            padding=padding,
            truncation=truncation
        )
    except Exception as e:
        logger.error(f"Tokenization Error: {e}")
        raise

# Compile regex patterns
SPECIAL_TOKENS = [
    "", "<|system|>", "", 
    "<||system||>", "<|user|}{assistant|>", "<||assistant---", "|>", 
    "<||user|>", "<||assistant|>", "---", "||", "</s>", "<s>", 
    "", "<|bos|>", "<|eos|>"
]
STOP_TOKENS = ["", "<user>", "<|system|>", "</s>", ""]

TAG_PATTERN = re.compile(r"<.*?>")
WHITESPACE_PATTERN = re.compile(r"\s+")

def extract_assistant_response(text: str) -> str:
    """Extract assistant's message cleanly from formatted output."""
    if not text:
        return ""

    response = text.split("<|assistant|>")[-1] if "<|assistant|>" in text else text

    for token in STOP_TOKENS:
        idx = response.find(token)
        if idx != -1:
            response = response[:idx]

    response = TAG_PATTERN.sub("", response)
    for token in SPECIAL_TOKENS:
        response = response.replace(token, "")

    if "http://" in response and not response.endswith("http://"):
        response = response.split("http://")[0].strip()
    if "https://" in response and not response.endswith("https://"):
        response = response.split("https://")[0].strip()

    response = WHITESPACE_PATTERN.sub(" ", response)

    if len(response) < 10 and "<|assistant|>" in text:
        parts = text.split("<|assistant|>")
        if len(parts) > 2:
            additional_content = parts[1]
            for token in STOP_TOKENS:
                idx = additional_content.find(token)
                if idx != -1:
                    additional_content = additional_content[:idx]
            additional_content = TAG_PATTERN.sub("", additional_content)
            for token in SPECIAL_TOKENS:
                additional_content = additional_content.replace(token, "")
            response += " " + additional_content.strip()

    return response.strip()

def create_directory_if_not_exists(directory: Union[str, Path]) -> None:
    """Create a directory if it doesn't exist."""
    try:
        Path(directory).mkdir(parents=True, exist_ok=True)
    except Exception as e:
        logger.error(f"Directory Creation Error: {directory} -> {e}")
        raise

@functools.lru_cache(maxsize=1)
def get_device() -> torch.device:
    """Get the appropriate device for model training/inference with caching."""
    device = (
        torch.device("cuda") if torch.cuda.is_available() else
        torch.device("mps") if torch.backends.mps.is_available() else
        torch.device("cpu")
    )
    logger.info(f"Using device: {device}")
    return device

def count_parameters(model: torch.nn.Module) -> Dict[str, int]:
    """Count the number of trainable and non-trainable parameters in a model."""
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    non_trainable_params = sum(p.numel() for p in model.parameters() if not p.requires_grad)
    total_params = trainable_params + non_trainable_params

    return {
        "trainable": trainable_params,
        "non_trainable": non_trainable_params,
        "total": total_params
    }

def clear_cache() -> None:
    """Clear the file cache."""
    global _file_cache
    _file_cache = {}
    logger.info("File cache cleared.")

def get_relevant_sloka(emotion_tag, slokas_path="data/gita_slokas.json"):
    """
    Retrieve a random sloka matching the emotion tag from gita_slokas.json.
    Returns a dict with chapter, verse, sloka, and meaning. Returns None if not found.
    """
    try:
        with open(slokas_path, "r") as f:
            data = json.load(f)
        matching_slokas = []
        for chapter in data.get("slokas", []):
            for verse in chapter.get("verses", []):
                if "tags" in verse and emotion_tag.lower() in [t.lower() for t in verse["tags"]]:
                    matching_slokas.append({
                        "chapter": chapter["chapter"],
                        "verse": verse["verse"],
                        "sloka": verse["sloka"],
                        "meaning": verse["meaning"]
                    })
        if not matching_slokas:
            return None
        return random.choice(matching_slokas)
    except Exception as e:
        logger.error(f"Sloka Retrieval Error: {e}")
        return None

def detect_emotion(text):
    """
    Simple keyword-based emotion detector.
    Returns one of: 'happy', 'sad', 'depressed', 'angry', 'anxious', 'motivated', 'neutral'.
    Defaults to 'neutral' if no strong emotion is detected.
    """
    if not isinstance(text, str) or not text.strip():
        return 'neutral'
    text_lower = text.lower()
    emotion_keywords = {
        'happy': ['happy', 'joy', 'delighted', 'excited', 'grateful', 'thankful', 'pleased', 'content'],
        'sad': ['sad', 'unhappy', 'down', 'cry', 'tears', 'sorrow', 'heartbroken', 'upset', 'grief'],
        'depressed': ['depressed', 'hopeless', 'worthless', 'empty', 'numb', 'helpless', 'tired', 'exhausted'],
        'angry': ['angry', 'mad', 'furious', 'rage', 'annoyed', 'irritated', 'resentful'],
        'anxious': ['anxious', 'worried', 'nervous', 'scared', 'afraid', 'panic', 'tense', 'stressed'],
        'motivated': ['motivated', 'inspired', 'determined', 'driven', 'ambitious', 'confident', 'strong'],
    }
    for emotion, keywords in emotion_keywords.items():
        for kw in keywords:
            if kw in text_lower:
                return emotion
    return 'neutral'

def final_response(model_response, emotion_tag):
    """
    Combine the model's response with a relevant Bhagavad Gita sloka and its meaning.
    """
    if not isinstance(model_response, str):
        model_response = str(model_response)
    if not isinstance(emotion_tag, str) or not emotion_tag.strip():
        emotion_tag = 'neutral'
    sloka = get_relevant_sloka(emotion_tag)
    if sloka:
        sloka_text = (
            f"\n\nHere's a relevant Bhagavad Gita sloka for you:\n"
            f"Sloka {sloka.get('chapter', '?')}.{sloka.get('verse', '?')}:\n"
            f"{sloka.get('sloka', '')}\n"
            f"Meaning: {sloka.get('meaning', '')}"
        )
        return model_response + sloka_text
    else:
        return model_response + "\n\n(I couldn't find a relevant sloka this time, but I'm here to support you!)"
