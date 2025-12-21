#!/usr/bin/env python3
"""
Script 7: Test Full Fine-Tuned Model
Version: 1.1.0

Purpose: Test the fully fine-tuned model on benchmark questions.

Usage:
    python scripts/7_test_full_model.py

Input:
    - models/full_finetuned_model/: Full model from script 6

Output:
    - results/full_model_results.json: Full model benchmark results
"""

import sys
from pathlib import Path
from datetime import datetime

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Add scripts directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from common import (
    Config,
    setup_logging,
    format_prompt,
    generate_response,
    save_json,
    print_header,
    print_section,
    get_vram_usage,
    AUTOIMMUNE_QUESTIONS,
)


def load_full_model(logger):
    """Load the fully fine-tuned model."""
    model_path = Config.MODELS_DIR / "full_finetuned_model"
    
    if not model_path.exists():
        logger.error(f"Full model not found: {model_path}")
        print(f"✗ ERROR: Full model not found at {model_path}")
        print("Please run script 6 first to train the full model.")
        sys.exit(1)
    
    print(f"Loading full model from {model_path}...")
    logger.info(f"Loading full model from {model_path}")
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        trust_remote_code=True
    )
    
    # Load model
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    
    vram = get_vram_usage()
    print(f"✓ Model loaded, VRAM: {vram['allocated']:.2f} GB")
    logger.info(f"Model loaded, VRAM: {vram['allocated']:.2f} GB")
    
    return model, tokenizer


def run_benchmark(model, tokenizer, logger) -> list:
    """Run benchmark on all questions."""
    results = []
    total_questions = len(AUTOIMMUNE_QUESTIONS)
    
    print_section(f"Running Full Model Benchmark: {total_questions} Questions")
    
    for i, question_data in enumerate(AUTOIMMUNE_QUESTIONS, 1):
        print(f"\n[{i}/{total_questions}] {question_data['id']}: {question_data['category']}")
        print(f"Difficulty: {question_data['difficulty']}")
        print(f"Question: {question_data['question'][:80]}...")
        
        logger.info(f"Processing {question_data['id']}: {question_data['category']}")
        
        try:
            prompt = format_prompt(question_data['question'], tokenizer=tokenizer)
            response, gen_time, tokens = generate_response(
                model, tokenizer, prompt,
                max_new_tokens=1024,
                temperature=0.7
            )
            
            result = {
                "id": question_data['id'],
                "category": question_data['category'],
                "difficulty": question_data['difficulty'],
                "question": question_data['question'],
                "response": response,
                "time_seconds": round(gen_time, 2),
                "input_tokens": tokens['input_tokens'],
                "output_tokens": tokens['output_tokens'],
                "word_count": len(response.split()),
                "timestamp": datetime.now().isoformat(),
                "model_type": "full_finetuned"
            }
            
            results.append(result)
            print(f"✓ Completed in {gen_time:.2f}s ({tokens['output_tokens']} tokens)")
            logger.info(f"Completed {question_data['id']} in {gen_time:.2f}s")
            
        except Exception as e:
            logger.error(f"Error on {question_data['id']}: {str(e)}")
            print(f"✗ Error: {str(e)}")
            
            result = {
                "id": question_data['id'],
                "category": question_data['category'],
                "difficulty": question_data['difficulty'],
                "question": question_data['question'],
                "response": f"ERROR: {str(e)}",
                "time_seconds": 0.0,
                "input_tokens": 0,
                "output_tokens": 0,
                "word_count": 0,
                "timestamp": datetime.now().isoformat(),
                "model_type": "full_finetuned",
                "error": True
            }
            results.append(result)
    
    return results


def print_summary(results: list):
    """Print summary statistics."""
    print_section("FULL MODEL BENCHMARK SUMMARY")
    
    successful = [r for r in results if not r.get('error')]
    
    total_time = sum(r['time_seconds'] for r in successful)
    avg_time = total_time / len(successful) if successful else 0
    avg_words = sum(r['word_count'] for r in successful) / len(successful) if successful else 0
    
    print(f"Total Questions: {len(results)}")
    print(f"Successful: {len(successful)}")
    print(f"Total Time: {total_time:.2f}s")
    print(f"Average Time: {avg_time:.2f}s")
    print(f"Average Word Count: {avg_words:.0f}")


def main():
    """Main execution function."""
    print_header("SCRIPT 7: TEST FULL FINE-TUNED MODEL")
    
    # Setup
    logger = setup_logging("7_test_full_model")
    Config.ensure_directories()
    
    try:
        # Load full model
        model, tokenizer = load_full_model(logger)
        
        # Run benchmark
        results = run_benchmark(model, tokenizer, logger)
        
        # Save results
        output_file = Config.RESULTS_DIR / "full_model_results.json"
        save_json(results, output_file)
        print(f"\n✓ Results saved to {output_file}")
        logger.info(f"Results saved to {output_file}")
        
        # Print summary
        print_summary(results)
        
        print_header("✓ Script 7 completed successfully!")
        logger.info("Script completed successfully")
        
    except Exception as e:
        logger.error(f"Error: {str(e)}", exc_info=True)
        print(f"\n✗ Error: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
