#!/usr/bin/env python3
"""
Script 1: Download and Test Model
Version: 1.0.0

Purpose: Download DeepSeek-R1-Distill-Qwen-8B model and run basic inference test.

Usage:
    python scripts/1_download_and_test.py

Output:
    - Downloads model to Hugging Face cache
    - Creates model_info.txt with model details
    - Prints inference test results to console
"""

import os
import sys
import time
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

MODEL_NAME = os.getenv("MODEL_NAME", "deepseek-ai/DeepSeek-R1-Distill-Qwen-8B")
MAX_SEQ_LENGTH = int(os.getenv("MAX_SEQ_LENGTH", "2048"))


def check_cuda():
    """Check CUDA availability and print GPU information."""
    if not torch.cuda.is_available():
        print("WARNING: CUDA is not available. Model will run on CPU (very slow).")
        return False
    
    print(f"✓ CUDA Available: {torch.cuda.is_available()}")
    print(f"✓ CUDA Version: {torch.version.cuda}")
    print(f"✓ GPU: {torch.cuda.get_device_name(0)}")
    print(f"✓ VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
    return True


def download_and_load_model():
    """
    Download and load the model with 4-bit quantization.
    
    Returns:
        Tuple[AutoModelForCausalLM, AutoTokenizer]: Model and tokenizer
    """
    print(f"\n{'='*60}")
    print(f"Downloading model: {MODEL_NAME}")
    print(f"{'='*60}\n")
    
    # Load tokenizer
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME,
        trust_remote_code=True
    )
    
    # Load model with 4-bit quantization
    print("Loading model with 4-bit quantization...")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
        load_in_4bit=True,
        max_length=MAX_SEQ_LENGTH
    )
    
    # Print model info
    if torch.cuda.is_available():
        vram_used = torch.cuda.memory_allocated(0) / 1024**3
        print(f"\n✓ Model loaded successfully")
        print(f"✓ VRAM Used: {vram_used:.2f} GB")
    
    return model, tokenizer


def test_inference(model, tokenizer, prompt: str):
    """
    Run inference test on the model.
    
    Args:
        model: The loaded model
        tokenizer: The tokenizer
        prompt: Input prompt text
    
    Returns:
        str: Generated response
    """
    print(f"\n{'='*60}")
    print("Running Inference Test")
    print(f"{'='*60}\n")
    print(f"Prompt: {prompt[:100]}...\n")
    
    # Tokenize input
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    # Generate response
    start_time = time.time()
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=512,
            temperature=0.7,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id
        )
    generation_time = time.time() - start_time
    
    # Decode response
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # Calculate metrics
    input_tokens = inputs['input_ids'].shape[1]
    output_tokens = outputs.shape[1] - input_tokens
    tokens_per_second = output_tokens / generation_time if generation_time > 0 else 0
    
    print(f"Response:\n{response[len(prompt):]}\n")
    print(f"Generation Time: {generation_time:.2f} seconds")
    print(f"Input Tokens: {input_tokens}")
    print(f"Output Tokens: {output_tokens}")
    print(f"Tokens/Second: {tokens_per_second:.2f}")
    
    return response, generation_time, tokens_per_second


def save_model_info(model, tokenizer):
    """Save model information to file."""
    info_file = "model_info.txt"
    with open(info_file, "w") as f:
        f.write(f"Model: {MODEL_NAME}\n")
        f.write(f"Max Sequence Length: {MAX_SEQ_LENGTH}\n")
        f.write(f"CUDA Available: {torch.cuda.is_available()}\n")
        if torch.cuda.is_available():
            f.write(f"GPU: {torch.cuda.get_device_name(0)}\n")
            f.write(f"VRAM Used: {torch.cuda.memory_allocated(0) / 1024**3:.2f} GB\n")
        f.write(f"Model Type: {type(model).__name__}\n")
        f.write(f"Tokenizer Type: {type(tokenizer).__name__}\n")
        f.write(f"Vocab Size: {len(tokenizer)}\n")
    
    print(f"\n✓ Model info saved to {info_file}")


def main():
    """Main execution function."""
    print("\n" + "="*60)
    print("SCRIPT 1: DOWNLOAD AND TEST MODEL")
    print("="*60 + "\n")
    
    # Check CUDA
    cuda_available = check_cuda()
    if not cuda_available:
        response = input("\nContinue without CUDA? (y/n): ")
        if response.lower() != 'y':
            sys.exit(1)
    
    try:
        # Download and load model
        model, tokenizer = download_and_load_model()
        
        # Test inference with medical prompt
        test_prompt = """<｜begin▁of▁sentence｜>User: What are the key diagnostic criteria for systemic lupus erythematosus (SLE)? Please explain the ACR classification criteria.RetryRMfinish"""
        
        response, gen_time, tokens_per_sec = test_inference(model, tokenizer, test_prompt)
        
        # Save model info
        save_model_info(model, tokenizer)
        
        print("\n" + "="*60)
        print("✓ Script 1 completed successfully!")
        print("="*60 + "\n")
        
    except Exception as e:
        print(f"\n✗ Error: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()

