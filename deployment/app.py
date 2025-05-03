from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Optional
import uvicorn
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import yaml
import os

# Import utility functions
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scripts.utils import (
    load_yaml,
    format_chatml,
    extract_assistant_response,
    get_device
)

# Initialize FastAPI app
app = FastAPI(
    title="Raadhe AI API",
    description="API for interacting with the Raadhe AI emotional intelligence model",
    version="1.0.0"
)

# Load configuration
config = load_yaml("config/train_config.yaml")

# Initialize model and tokenizer
device = get_device()
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

# Define request and response models
class Message(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    messages: List[Message]
    max_length: Optional[int] = 2048
    temperature: Optional[float] = 0.7
    top_p: Optional[float] = 0.9

class ChatResponse(BaseModel):
    response: str
    model_info: Dict[str, str]

@app.get("/")
async def root():
    """Root endpoint with basic information."""
    return {
        "name": "Raadhe AI",
        "version": "1.0.0",
        "description": "An emotionally intelligent AI companion"
    }

@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """Generate a response from Raadhe AI."""
    try:
        # Format messages
        messages = [msg.dict() for msg in request.messages]
        formatted_text = format_chatml(messages)
        
        # Tokenize input
        inputs = tokenizer(
            formatted_text,
            return_tensors="pt",
            max_length=request.max_length,
            padding=True,
            truncation=True
        ).to(device)
        
        # Generate response
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=120,
                #max_length=request.max_length,
                num_return_sequences=1,
                temperature=request.temperature or 0.85,
                top_p=request.top_p or 0.95,
                top_k=50,
                repetition_penalty=1.05,
                no_repeat_ngram_size=3,
                do_sample=True,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id
            )
        
        # Decode response
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        assistant_response = extract_assistant_response(response)
        
        return ChatResponse(
            response=assistant_response,
            model_info={
                "model": config["model"]["base_model"],
                "version": "1.0.0"
            }
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy"}

if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True) 