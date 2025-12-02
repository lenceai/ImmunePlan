#!/usr/bin/env python3
"""
Script 1: Download and Test Model
Version: 1.1.0

Purpose: Download DeepSeek-R1-Distill-Qwen-7B model and run basic inference test.

Usage:
    python scripts/1_download_and_test.py

Output:
    - Downloads model to Hugging Face cache
    - Creates model_info.txt with model details
    - Prints inference test results to console
"""

import sys
from pathlib import Path

# Add scripts directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from common import (
    Config,
    setup_logging,
    check_cuda,
    print_cuda_info,
    load_model_and_tokenizer,
    format_prompt,
    generate_response,
    get_vram_usage,
    print_header,
    confirm_continue,
)


def save_model_info(model, tokenizer, test_results: dict):
    """Save model information to file."""
    import torch
    
    info_file = Path("model_info.txt")
    
    with open(info_file, "w") as f:
        f.write("=" * 60 + "\n")
        f.write("AUTOIMMUNE LLM - MODEL INFORMATION\n")
        f.write("=" * 60 + "\n\n")
        
        f.write(f"Model: {Config.MODEL_NAME}\n")
        f.write(f"Max Sequence Length: {Config.MAX_SEQ_LENGTH}\n")
        f.write(f"Quantization: 4-bit NF4\n")
        f.write(f"Dtype: bfloat16\n\n")
        
        f.write("HARDWARE\n")
        f.write("-" * 30 + "\n")
        f.write(f"CUDA Available: {torch.cuda.is_available()}\n")
        if torch.cuda.is_available():
            f.write(f"GPU: {torch.cuda.get_device_name(0)}\n")
            f.write(f"CUDA Version: {torch.version.cuda}\n")
            vram = get_vram_usage()
            f.write(f"VRAM Allocated: {vram['allocated']:.2f} GB\n")
            f.write(f"VRAM Total: {vram['total']:.2f} GB\n")
        f.write("\n")
        
        f.write("MODEL INFO\n")
        f.write("-" * 30 + "\n")
        f.write(f"Model Type: {type(model).__name__}\n")
        f.write(f"Tokenizer Type: {type(tokenizer).__name__}\n")
        f.write(f"Vocab Size: {len(tokenizer)}\n\n")
        
        f.write("INFERENCE TEST\n")
        f.write("-" * 30 + "\n")
        f.write(f"Generation Time: {test_results['time']:.2f}s\n")
        f.write(f"Input Tokens: {test_results['tokens']['input_tokens']}\n")
        f.write(f"Output Tokens: {test_results['tokens']['output_tokens']}\n")
        f.write(f"Tokens/Second: {test_results['tokens_per_second']:.2f}\n")
    
    print(f"✓ Model info saved to {info_file}")


def main():
    """Main execution function."""
    print_header("SCRIPT 1: DOWNLOAD AND TEST MODEL")
    
    # Setup logging
    logger = setup_logging("1_download_and_test")
    logger.info(f"Starting script with model: {Config.MODEL_NAME}")
    
    # Check CUDA
    cuda_info = print_cuda_info(logger)
    
    if not cuda_info["available"]:
        logger.warning("CUDA not available - model will run on CPU (very slow)")
        if not confirm_continue("Continue without CUDA?"):
            sys.exit(1)
    
    try:
        # Download and load model
        print_header(f"Downloading model: {Config.MODEL_NAME}", char="-")
        
        print("Loading tokenizer...")
        logger.info("Loading tokenizer...")
        
        model, tokenizer = load_model_and_tokenizer(quantize=True)
        
        vram = get_vram_usage()
        print(f"\n✓ Model loaded successfully")
        print(f"✓ VRAM Used: {vram['allocated']:.2f} GB")
        logger.info(f"Model loaded, VRAM: {vram['allocated']:.2f} GB")
        
        # Test inference
        print_header("Running Inference Test", char="-")
        
        test_question = "What are the key diagnostic criteria for systemic lupus erythematosus (SLE)? Please explain the ACR/EULAR classification criteria briefly."
        prompt = format_prompt(test_question)
        
        print(f"Prompt: {test_question[:80]}...\n")
        
        response, gen_time, tokens = generate_response(
            model, tokenizer, prompt,
            max_new_tokens=512,
            temperature=0.7
        )
        
        tokens_per_second = tokens['output_tokens'] / gen_time if gen_time > 0 else 0
        
        print(f"\nResponse:\n{'-' * 40}")
        print(response[:500] + "..." if len(response) > 500 else response)
        print(f"{'-' * 40}\n")
        
        print(f"Generation Time: {gen_time:.2f}s")
        print(f"Input Tokens: {tokens['input_tokens']}")
        print(f"Output Tokens: {tokens['output_tokens']}")
        print(f"Tokens/Second: {tokens_per_second:.2f}")
        
        logger.info(f"Inference test completed: {gen_time:.2f}s, {tokens_per_second:.2f} tok/s")
        
        # Save model info
        test_results = {
            "time": gen_time,
            "tokens": tokens,
            "tokens_per_second": tokens_per_second
        }
        save_model_info(model, tokenizer, test_results)
        
        print_header("✓ Script 1 completed successfully!")
        logger.info("Script completed successfully")
        
    except Exception as e:
        logger.error(f"Error: {str(e)}", exc_info=True)
        print(f"\n✗ Error: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
