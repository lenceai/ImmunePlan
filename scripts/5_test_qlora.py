#!/usr/bin/env python3
"""
Script 5: Test Fine-Tuned Model
Version: 2.0.0

Purpose: Test the fine-tuned model (DoRA or QLoRA) on benchmark questions.

Usage:
    python scripts/5_test_qlora.py

Input:
    - models/dora_model/: DoRA model from script 4 (preferred)
    - models/qlora_model/: QLoRA model from script 4 (legacy)

Output:
    - results/finetuned_results.json: Fine-tuned model benchmark results
"""

import sys
from pathlib import Path
from datetime import datetime

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

# Add scripts directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from common import (
    Config,
    setup_logging,
    get_quantization_config,
    format_prompt,
    generate_response,
    save_json,
    print_header,
    print_section,
    get_vram_usage,
    AUTOIMMUNE_QUESTIONS,
)


def load_finetuned_model(logger):
    """Load base model and fine-tuned adapters (DoRA or QLoRA)."""
    # Check for DoRA model first (newer), then QLoRA (legacy)
    dora_path = Config.MODELS_DIR / "dora_model"
    qlora_path = Config.MODELS_DIR / "qlora_model"
    
    adapter_path = None
    adapter_type = None
    
    if dora_path.exists():
        adapter_path = dora_path
        adapter_type = "DoRA"
    elif qlora_path.exists():
        adapter_path = qlora_path
        adapter_type = "QLoRA"
    else:
        logger.error(f"No fine-tuned model found")
        print(f"✗ ERROR: No fine-tuned model found")
        print("  Checked for:")
        print(f"    - {dora_path}")
        print(f"    - {qlora_path}")
        print("Please run script 4 first to train the model.")
        sys.exit(1)
    
    print(f"Loading base model: {Config.MODEL_NAME}...")
    logger.info(f"Loading base model: {Config.MODEL_NAME}")
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        Config.MODEL_NAME,
        trust_remote_code=True
    )
    
    # Load base model with quantization
    base_model = AutoModelForCausalLM.from_pretrained(
        Config.MODEL_NAME,
        quantization_config=get_quantization_config(),
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    
    # Load adapters
    print(f"Loading {adapter_type} adapters from {adapter_path}...")
    logger.info(f"Loading {adapter_type} adapters from {adapter_path}")
    
    model = PeftModel.from_pretrained(base_model, adapter_path)
    
    vram = get_vram_usage()
    print(f"✓ Model loaded ({adapter_type}), VRAM: {vram['allocated']:.2f} GB")
    logger.info(f"Model loaded ({adapter_type}), VRAM: {vram['allocated']:.2f} GB")
    
    return model, tokenizer, adapter_type


def run_benchmark(model, tokenizer, logger, adapter_type: str = "finetuned") -> list:
    """Run benchmark on all questions."""
    results = []
    total_questions = len(AUTOIMMUNE_QUESTIONS)
    
    print_section(f"Running Fine-Tuned Model Benchmark: {total_questions} Questions")
    
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
                "model_type": adapter_type.lower() if 'adapter_type' in locals() else "finetuned"
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
                "model_type": adapter_type.lower(),
                "error": True
            }
            results.append(result)
    
    return results


def print_summary(results: list):
    """Print summary statistics."""
    print_section("FINE-TUNED MODEL BENCHMARK SUMMARY")
    
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
    print_header("SCRIPT 5: TEST FINE-TUNED MODEL")
    
    # Setup
    logger = setup_logging("5_test_finetuned")
    Config.ensure_directories()
    
    try:
        # Load fine-tuned model (DoRA or QLoRA)
        model, tokenizer, adapter_type = load_finetuned_model(logger)
        
        # Run benchmark
        results = run_benchmark(model, tokenizer, logger, adapter_type)
        
        # Save results
        output_file = Config.RESULTS_DIR / "finetuned_results.json"
        save_json(results, output_file)
        print(f"\n✓ Results saved to {output_file}")
        logger.info(f"Results saved to {output_file}")
        
        # Print summary
        print_summary(results)
        
        print_header("✓ Script 5 completed successfully!")
        logger.info("Script completed successfully")
        
    except Exception as e:
        logger.error(f"Error: {str(e)}", exc_info=True)
        print(f"\n✗ Error: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
