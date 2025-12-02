#!/usr/bin/env python3
"""
Script 2: Autoimmune Benchmark Questions
Version: 1.1.0

Purpose: Test model with comprehensive autoimmune disease benchmark questions.

Usage:
    python scripts/2_autoimmune_questions.py

Output:
    - results/baseline_results.json: Benchmark results for baseline model
"""

import sys
from pathlib import Path
from datetime import datetime

# Add scripts directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from common import (
    Config,
    setup_logging,
    load_model_and_tokenizer,
    format_prompt,
    generate_response,
    save_json,
    print_header,
    print_section,
    AUTOIMMUNE_QUESTIONS,
)


def run_benchmark(model, tokenizer, logger) -> list:
    """
    Run benchmark on all questions.
    
    Args:
        model: The model
        tokenizer: The tokenizer
        logger: Logger instance
    
    Returns:
        List of result dictionaries
    """
    results = []
    total_questions = len(AUTOIMMUNE_QUESTIONS)
    
    print_section(f"Running Benchmark: {total_questions} Questions")
    
    for i, question_data in enumerate(AUTOIMMUNE_QUESTIONS, 1):
        print(f"\n[{i}/{total_questions}] {question_data['id']}: {question_data['category']}")
        print(f"Difficulty: {question_data['difficulty']}")
        print(f"Question: {question_data['question'][:80]}...")
        
        logger.info(f"Processing {question_data['id']}: {question_data['category']}")
        
        try:
            prompt = format_prompt(question_data['question'])
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
                "timestamp": datetime.now().isoformat()
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
                "error": True
            }
            results.append(result)
    
    return results


def print_summary(results: list):
    """Print summary statistics."""
    print_section("BENCHMARK SUMMARY")
    
    successful = [r for r in results if not r.get('error')]
    
    total_time = sum(r['time_seconds'] for r in successful)
    avg_time = total_time / len(successful) if successful else 0
    avg_words = sum(r['word_count'] for r in successful) / len(successful) if successful else 0
    
    print(f"Total Questions: {len(results)}")
    print(f"Successful: {len(successful)}")
    print(f"Total Time: {total_time:.2f}s")
    print(f"Average Time: {avg_time:.2f}s")
    print(f"Average Word Count: {avg_words:.0f}")
    
    # By difficulty
    print("\nBy Difficulty:")
    difficulties = {}
    for r in successful:
        diff = r['difficulty']
        if diff not in difficulties:
            difficulties[diff] = []
        difficulties[diff].append(r['time_seconds'])
    
    for diff, times in sorted(difficulties.items()):
        avg = sum(times) / len(times)
        print(f"  {diff}: {len(times)} questions, avg {avg:.2f}s")
    
    # By category
    print("\nBy Category:")
    categories = {}
    for r in successful:
        cat = r['category']
        if cat not in categories:
            categories[cat] = 0
        categories[cat] += 1
    
    for cat, count in sorted(categories.items()):
        print(f"  {cat}: {count}")


def main():
    """Main execution function."""
    print_header("SCRIPT 2: AUTOIMMUNE BENCHMARK QUESTIONS")
    
    # Setup
    logger = setup_logging("2_autoimmune_questions")
    Config.ensure_directories()
    
    logger.info(f"Starting benchmark with model: {Config.MODEL_NAME}")
    
    try:
        # Load model
        print(f"Loading model: {Config.MODEL_NAME}...")
        model, tokenizer = load_model_and_tokenizer(quantize=True)
        logger.info("Model loaded successfully")
        
        # Run benchmark
        results = run_benchmark(model, tokenizer, logger)
        
        # Save results
        output_file = Config.RESULTS_DIR / "baseline_results.json"
        save_json(results, output_file)
        print(f"\n✓ Results saved to {output_file}")
        logger.info(f"Results saved to {output_file}")
        
        # Print summary
        print_summary(results)
        
        print_header("✓ Script 2 completed successfully!")
        logger.info("Script completed successfully")
        
    except Exception as e:
        logger.error(f"Error: {str(e)}", exc_info=True)
        print(f"\n✗ Error: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
