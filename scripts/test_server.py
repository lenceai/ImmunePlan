#!/usr/bin/env python3
"""
Simple OpenAI-compatible API server for DeepSeek-R1-Distill-Qwen-1.5B
Optimized for low-VRAM devices (Steam Deck, laptops, etc.)

Usage: python3 scripts/test_server.py
Version: 1.0.0
"""

import os
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import uvicorn

try:
    from transformers import AutoModelForCausalLM, AutoTokenizer
    import torch
except ImportError:
    print("ERROR: transformers and torch are required.")
    print("Install with: pip install transformers torch accelerate bitsandbytes")
    sys.exit(1)

# Configuration
MODEL_NAME = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
MAX_CONTEXT = 2048
HOST = "0.0.0.0"
PORT = 8000

# Global model and tokenizer
model = None
tokenizer = None

app = FastAPI(title="DeepSeek-R1-Distill-Qwen-1.5B Test Server")

# Request/Response models (OpenAI-compatible)
class Message(BaseModel):
    role: str
    content: str

class ChatCompletionRequest(BaseModel):
    model: str
    messages: List[Message]
    max_tokens: Optional[int] = 512
    temperature: Optional[float] = 0.7
    top_p: Optional[float] = 0.9

class ChatCompletionChoice(BaseModel):
    index: int
    message: Message
    finish_reason: str

class ChatCompletionResponse(BaseModel):
    id: str
    object: str = "chat.completion"
    created: int
    model: str
    choices: List[ChatCompletionChoice]

def load_model():
    """Load the model with 4-bit quantization for low VRAM."""
    global model, tokenizer
    
    if model is not None:
        return
    
    print(f"Loading model: {MODEL_NAME}...")
    print("Using 4-bit quantization for low VRAM usage...")
    
    # Check if we can use 4-bit quantization
    try:
        from transformers import BitsAndBytesConfig
        
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4"
        )
        
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME,
            quantization_config=quantization_config,
            device_map="auto",
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
        )
    except Exception as e:
        print(f"4-bit quantization failed: {e}")
        print("Falling back to FP16 (will use more VRAM)...")
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME,
            device_map="auto",
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
        )
    
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    print("✓ Model loaded successfully!")

@app.on_event("startup")
async def startup_event():
    """Load model on server startup."""
    load_model()

@app.get("/")
async def root():
    return {"message": "DeepSeek-R1-Distill-Qwen-1.5B Test Server", "status": "running"}

@app.get("/v1/models")
async def list_models():
    """List available models (OpenAI-compatible endpoint)."""
    return {
        "object": "list",
        "data": [{
            "id": MODEL_NAME,
            "object": "model",
            "created": 0,
            "owned_by": "deepseek-ai"
        }]
    }

@app.post("/v1/chat/completions")
async def chat_completions(request: ChatCompletionRequest):
    """Handle chat completion requests (OpenAI-compatible)."""
    global model, tokenizer
    
    if model is None or tokenizer is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    # Format messages using the model's chat template
    try:
        # Convert messages to format expected by tokenizer
        formatted_messages = [{"role": msg.role, "content": msg.content} for msg in request.messages]
        prompt = tokenizer.apply_chat_template(
            formatted_messages,
            tokenize=False,
            add_generation_prompt=True
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error formatting messages: {e}")
    
    # Tokenize
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=MAX_CONTEXT)
    if torch.cuda.is_available():
        inputs = {k: v.cuda() for k, v in inputs.items()}
    
    # Generate
    try:
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=request.max_tokens,
                temperature=request.temperature,
                top_p=request.top_p,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id,
            )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Generation error: {e}")
    
    # Decode response
    generated_text = tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
    
    # Return OpenAI-compatible response
    import time
    return {
        "id": f"chatcmpl-{int(time.time())}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": MODEL_NAME,
        "choices": [{
            "index": 0,
            "message": {
                "role": "assistant",
                "content": generated_text
            },
            "finish_reason": "stop"
        }],
        "usage": {
            "prompt_tokens": inputs['input_ids'].shape[1],
            "completion_tokens": len(outputs[0]) - inputs['input_ids'].shape[1],
            "total_tokens": len(outputs[0])
        }
    }

if __name__ == "__main__":
    print("=" * 64)
    print("DEEPSEEK-R1-DISTILL-QWEN-1.5B TEST SERVER")
    print("=" * 64)
    print(f"Model: {MODEL_NAME}")
    print(f"API: http://{HOST}:{PORT}/v1")
    print(f"Docs: http://{HOST}:{PORT}/docs")
    print("=" * 64)
    
    uvicorn.run(app, host=HOST, port=PORT)

