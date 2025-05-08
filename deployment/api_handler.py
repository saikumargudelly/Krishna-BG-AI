from typing import List, Dict, Any, Optional
import torch
from transformers import PreTrainedModel, PreTrainedTokenizer
from fastapi import HTTPException
import logging
import time
from datetime import datetime
import random

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class RaadheAPIHandler:
    def __init__(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizer,
        device: torch.device,
        max_length: int = 2048,
        default_temperature: float = 0.7,
        default_top_p: float = 0.9
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.max_length = max_length
        self.default_temperature = default_temperature
        self.default_top_p = default_top_p
        self.request_count = 0
        self.start_time = datetime.now()
        
        # Enable model optimizations
        self.model.eval()  # Ensure model is in eval mode
        if hasattr(self.model, 'half'):  # Enable half precision if available
            self.model = self.model.half()
        
        # Enable model optimizations for faster inference
        if hasattr(self.model, 'config'):
            self.model.config.use_cache = True
        
        # Initialize response cache
        self.response_cache = {}
        self.cache_size = 100  # Maximum number of cached responses

    def _get_cache_key(self, messages: List[Dict[str, str]]) -> str:
        """Generate a cache key from messages."""
        return "|".join(f"{msg['role']}:{msg['content']}" for msg in messages)

    def _get_cached_response(self, messages: List[Dict[str, str]]) -> Optional[Dict[str, Any]]:
        """Get cached response if available."""
        cache_key = self._get_cache_key(messages)
        return self.response_cache.get(cache_key)

    def _cache_response(self, messages: List[Dict[str, str]], response: Dict[str, Any]):
        """Cache the response."""
        if len(self.response_cache) >= self.cache_size:
            # Remove oldest entry if cache is full
            self.response_cache.pop(next(iter(self.response_cache)))
        cache_key = self._get_cache_key(messages)
        self.response_cache[cache_key] = response

    def generate_response(
        self,
        messages: List[Dict[str, str]],
        max_length: Optional[int] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None
    ) -> Dict[str, Any]:
        try:
            start_time = time.time()
            self.request_count += 1

            # Check cache first
            cached_response = self._get_cached_response(messages)
            if cached_response:
                logger.info("Using cached response")
                return cached_response

            # Apply defaults with dynamic adjustment based on message length
            avg_msg_length = sum(len(msg["content"]) for msg in messages) / len(messages)
            max_length = min(max_length or self.max_length, 1024)
            temperature = temperature or (0.85 if avg_msg_length > 100 else 0.7)
            top_p = top_p or (0.92 if avg_msg_length > 100 else 0.9)

            # Format messages in ChatML with enhanced context
            formatted_text = ""
            for msg in messages:
                role = msg["role"].strip().lower()
                content = msg["content"].strip()
                formatted_text += f"<|{role}|>\n{content}\n"
            if not formatted_text.strip().endswith("<|assistant|>"):
                formatted_text += "<|assistant|>\n"

            # Tokenize with enhanced padding and batching
            inputs = self.tokenizer(
                formatted_text,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=max_length
            ).to(self.device)

            # Optimize generation parameters for speed
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=512,
                    temperature=temperature,
                    top_p=top_p,
                    do_sample=True,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                    repetition_penalty=1.2,
                    no_repeat_ngram_size=3,
                    length_penalty=1.3,
                    early_stopping=True,
                    use_cache=True,
                    num_beams=2,  # Reduced from 4 for faster generation
                    min_length=40,
                    typical_p=0.95,
                    encoder_repetition_penalty=1.1,
                    diversity_penalty=0.1,
                    num_return_sequences=1,  # Ensure only one sequence is generated
                    output_scores=False,  # Disable score computation for speed
                    return_dict_in_generate=False  # Disable dictionary return for speed
                )

            # Decode and extract with enhanced cleaning
            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            assistant_response = response.split("<|assistant|>")[-1].strip()

            # Clean special tokens and normalize
            special_tokens = ["<|user|>", "<|system|>", "<|endoftext|>", "<|startoftext|>"]
            for token in special_tokens:
                assistant_response = assistant_response.replace(token, "")

            # Enhanced fallbacks with emotion awareness
            if not assistant_response or len(assistant_response) < 10:
                assistant_response = "Hey sweetie! I'd love to hear more about that. Could you tell me a bit more? 💖"
            elif len(assistant_response) > 400:
                sentences = assistant_response[:400].rsplit(".", 1)[0] + "..."
                assistant_response = sentences

            if any(t in assistant_response for t in ["<|", "|>"]):
                assistant_response = "Oops! Something glitched. Mind asking that again, lovely? 💫"

            # Add natural language patterns
            if not any(marker in assistant_response.lower() for marker in ["you know", "actually", "well", "i think"]):
                markers = ["You know,", "Actually,", "Well,", "I think,"]
                assistant_response = f"{random.choice(markers)} {assistant_response}"

            # Prepare response
            response_data = {
                "response": assistant_response,
                "metadata": {
                    "generation_time": time.time() - start_time,
                    "response_length": len(assistant_response),
                    "request_count": self.request_count,
                    "uptime": str(datetime.now() - self.start_time),
                    "generation_params": {
                        "temperature": temperature,
                        "top_p": top_p,
                        "max_length": max_length
                    }
                }
            }

            # Cache the response
            self._cache_response(messages, response_data)

            return response_data

        except Exception as e:
            logger.error(f"Error generating response: {str(e)}")
            raise HTTPException(status_code=500, detail=str(e))

    def get_stats(self) -> Dict[str, Any]:
        """Return usage statistics."""
        return {
            "total_requests": self.request_count,
            "uptime": str(datetime.now() - self.start_time),
            "model_info": {
                "max_length": self.max_length,
                "default_temperature": self.default_temperature,
                "default_top_p": self.default_top_p
            }
        }

    def reset_stats(self) -> None:
        """Reset API usage stats."""
        self.request_count = 0
        self.start_time = datetime.now()
        logger.info("API statistics reset")
