from typing import List, Dict, Any, Optional
import torch
from transformers import PreTrainedModel, PreTrainedTokenizer
from fastapi import HTTPException
import logging
import time
from datetime import datetime

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

            # Apply defaults
            max_length = min(max_length or self.max_length, 1024)
            temperature = temperature or self.default_temperature
            top_p = top_p or self.default_top_p

            # Format messages in ChatML
            formatted_text = ""
            for msg in messages:
                role = msg["role"].strip().lower()
                content = msg["content"].strip()
                formatted_text += f"<|{role}|>\n{content}\n"
            if not formatted_text.strip().endswith("<|assistant|>"):
                formatted_text += "<|assistant|>\n"

            # Tokenize
            inputs = self.tokenizer(
                formatted_text,
                return_tensors="pt",
                padding=True,
                truncation=True
            ).to(self.device)

            # Generate
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=300,
                    temperature=temperature,
                    top_p=top_p,
                    do_sample=True,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                    repetition_penalty=1.1,
                    no_repeat_ngram_size=3,
                    length_penalty=1.0,
                    early_stopping=True,
                    use_cache=True
                )

            # Decode and extract
            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            assistant_response = response.split("<|assistant|>")[-1].strip()

            # Clean special tokens
            special_tokens = ["<|user|>", "<|system|>", "<|endoftext|>", "<|startoftext|>"]
            for token in special_tokens:
                assistant_response = assistant_response.replace(token, "")

            # Fallbacks
            if not assistant_response or len(assistant_response) < 10:
                assistant_response = "Hey sweetie! I'd love to hear more about that. Could you tell me a bit more? 💖"
            elif len(assistant_response) > 400:
                assistant_response = assistant_response[:400].rsplit(" ", 1)[0] + "..."

            if any(t in assistant_response for t in ["<|", "|>"]):
                assistant_response = "Oops! Something glitched. Mind asking that again, lovely? 💫"

            # Log
            generation_time = time.time() - start_time
            logger.info(
                f"Request #{self.request_count} - Time: {generation_time:.2f}s - Length: {len(assistant_response)} chars"
            )

            return {
                "response": assistant_response,
                "metadata": {
                    "generation_time": generation_time,
                    "response_length": len(assistant_response),
                    "request_count": self.request_count,
                    "uptime": str(datetime.now() - self.start_time)
                }
            }

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
