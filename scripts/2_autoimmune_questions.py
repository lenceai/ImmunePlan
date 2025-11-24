#!/usr/bin/env python3
"""
Script 2: Autoimmune Benchmark Questions
Version: 1.0.0

Purpose: Test model with comprehensive autoimmune disease benchmark questions.

Usage:
    python scripts/2_autoimmune_questions.py

Output:
    - results/baseline_results.json: Benchmark results for baseline model
"""

import os
import sys
import json
import time
from datetime import datetime
from pathlib import Path
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

MODEL_NAME = os.getenv("MODEL_NAME", "deepseek-ai/DeepSeek-R1-Distill-Qwen-8B")
RESULTS_DIR = os.getenv("RESULTS_DIR", "./results")
MAX_SEQ_LENGTH = int(os.getenv("MAX_SEQ_LENGTH", "2048"))

# Ensure results directory exists
Path(RESULTS_DIR).mkdir(parents=True, exist_ok=True)


# Comprehensive autoimmune disease questions
AUTOIMMUNE_QUESTIONS = [
    {
        "id": "Q001",
        "category": "Systemic Lupus Erythematosus",
        "difficulty": "medium",
        "question": "A 28-year-old woman presents with malar rash, photosensitivity, oral ulcers, and joint pain. Her ANA is positive at 1:640 with a speckled pattern, and anti-dsDNA antibodies are elevated. What is the most likely diagnosis, and what are the key diagnostic criteria you would use?"
    },
    {
        "id": "Q002",
        "category": "Rheumatoid Arthritis",
        "difficulty": "medium",
        "question": "A 45-year-old patient has symmetric polyarthritis affecting wrists, MCPs, and PIPs for 6 months, with morning stiffness lasting over 1 hour. RF and anti-CCP antibodies are positive. Explain the diagnostic criteria for rheumatoid arthritis and the significance of these serological markers."
    },
    {
        "id": "Q003",
        "category": "Sjogren's Syndrome",
        "difficulty": "hard",
        "question": "A 52-year-old woman complains of dry eyes and dry mouth for 2 years. Schirmer's test is abnormal, and she has positive anti-SSA/Ro and anti-SSB/La antibodies. What diagnostic criteria would you use to confirm Sjogren's syndrome, and what are the potential systemic complications?"
    },
    {
        "id": "Q004",
        "category": "Mixed Connective Tissue Disease",
        "difficulty": "very_hard",
        "question": "A patient has features of SLE, scleroderma, and polymyositis. Anti-RNP antibodies are strongly positive. What is the diagnosis, and how does this condition differ from overlap syndromes?"
    },
    {
        "id": "Q005",
        "category": "Polymyositis",
        "difficulty": "hard",
        "question": "A 40-year-old presents with progressive proximal muscle weakness, elevated CK, and positive anti-Jo-1 antibodies. What is the diagnosis, and what are the key features distinguishing polymyositis from dermatomyositis?"
    },
    {
        "id": "Q006",
        "category": "Systemic Sclerosis",
        "difficulty": "hard",
        "question": "A patient has Raynaud's phenomenon, skin thickening, and positive anti-Scl-70 antibodies. Explain the classification criteria for systemic sclerosis and the difference between limited and diffuse forms."
    },
    {
        "id": "Q007",
        "category": "Vasculitis",
        "difficulty": "very_hard",
        "question": "A patient presents with sinusitis, pulmonary nodules, and rapidly progressive glomerulonephritis. c-ANCA and PR3 antibodies are positive. What is the diagnosis, and what is the typical treatment approach?"
    },
    {
        "id": "Q008",
        "category": "Differential Diagnosis",
        "difficulty": "very_hard",
        "question": "A 35-year-old woman has fatigue, joint pain, positive ANA, and low complement levels. However, she also has hyperthyroidism and vitiligo. How would you differentiate between multiple autoimmune conditions versus a single systemic autoimmune disease?"
    },
    {
        "id": "Q009",
        "category": "Behçet's Disease",
        "difficulty": "very_hard",
        "question": "A patient has recurrent oral ulcers, genital ulcers, and uveitis. What are the diagnostic criteria for Behçet's disease, and what are the key features that distinguish it from other causes of oral and genital ulcers?"
    },
    {
        "id": "Q010",
        "category": "Drug-Induced Lupus",
        "difficulty": "medium",
        "question": "A patient on hydralazine for hypertension develops positive ANA and symptoms resembling SLE. What are the key differences between drug-induced lupus and idiopathic SLE, and how would you manage this case?"
    }
]


def load_model():
    """
    Load the model and tokenizer.
    
    Returns:
        Tuple[Model, Tokenizer]: Model and tokenizer
    """
    print(f"Loading model: {MODEL_NAME}...")
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME,
        trust_remote_code=True
    )
    
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
        load_in_4bit=True,
        max_length=MAX_SEQ_LENGTH
    )
    
    return model, tokenizer


def get_model_response(model, tokenizer, question: str):
    """
    Get response from model for a given question.
    
    Args:
        model: The model
        tokenizer: The tokenizer
        question: Question text
    
    Returns:
        Tuple[str, float]: Response text and generation time in seconds
    """
    # Format prompt
    prompt = f"<｜begin▁of▁sentence｜>User: {question}RetryRMfinish"
    
    # Tokenize
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    # Generate
    start_time = time.time()
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=1024,
            temperature=0.7,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id
        )
    generation_time = time.time() - start_time
    
    # Decode
    full_response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    response = full_response[len(prompt):].strip()
    
    return response, generation_time


def run_benchmark(model, tokenizer):
    """
    Run benchmark on all questions.
    
    Args:
        model: The model
        tokenizer: The tokenizer
    
    Returns:
        List[Dict]: Results for each question
    """
    results = []
    
    print(f"\n{'='*60}")
    print(f"Running Benchmark: {len(AUTOIMMUNE_QUESTIONS)} Questions")
    print(f"{'='*60}\n")
    
    for i, question_data in enumerate(AUTOIMMUNE_QUESTIONS, 1):
        print(f"[{i}/{len(AUTOIMMUNE_QUESTIONS)}] {question_data['id']}: {question_data['category']}")
        print(f"Difficulty: {question_data['difficulty']}")
        print(f"Question: {question_data['question'][:100]}...")
        
        try:
            response, gen_time = get_model_response(
                model, tokenizer, question_data['question']
            )
            
            result = {
                "id": question_data['id'],
                "category": question_data['category'],
                "difficulty": question_data['difficulty'],
                "question": question_data['question'],
                "response": response,
                "time_seconds": round(gen_time, 2),
                "timestamp": datetime.now().isoformat()
            }
            
            results.append(result)
            print(f"✓ Completed in {gen_time:.2f}s\n")
            
        except Exception as e:
            print(f"✗ Error: {str(e)}\n")
            result = {
                "id": question_data['id'],
                "category": question_data['category'],
                "difficulty": question_data['difficulty'],
                "question": question_data['question'],
                "response": f"ERROR: {str(e)}",
                "time_seconds": 0.0,
                "timestamp": datetime.now().isoformat()
            }
            results.append(result)
    
    return results


def save_results(results, filename):
    """
    Save results to JSON file.
    
    Args:
        results: List of result dictionaries
        filename: Output filename
    """
    filepath = os.path.join(RESULTS_DIR, filename)
    with open(filepath, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"✓ Results saved to {filepath}")


def print_summary(results):
    """Print summary statistics."""
    print(f"\n{'='*60}")
    print("BENCHMARK SUMMARY")
    print(f"{'='*60}\n")
    
    total_time = sum(r['time_seconds'] for r in results)
    avg_time = total_time / len(results) if results else 0
    
    print(f"Total Questions: {len(results)}")
    print(f"Total Time: {total_time:.2f} seconds")
    print(f"Average Time per Question: {avg_time:.2f} seconds")
    
    # Count by difficulty
    difficulty_counts = {}
    for r in results:
        diff = r['difficulty']
        difficulty_counts[diff] = difficulty_counts.get(diff, 0) + 1
    
    print(f"\nBy Difficulty:")
    for diff, count in sorted(difficulty_counts.items()):
        print(f"  {diff}: {count}")
    
    # Count by category
    category_counts = {}
    for r in results:
        cat = r['category']
        category_counts[cat] = category_counts.get(cat, 0) + 1
    
    print(f"\nBy Category:")
    for cat, count in sorted(category_counts.items()):
        print(f"  {cat}: {count}")


def main():
    """Main execution function."""
    print("\n" + "="*60)
    print("SCRIPT 2: AUTOIMMUNE BENCHMARK QUESTIONS")
    print("="*60 + "\n")
    
    try:
        # Load model
        model, tokenizer = load_model()
        
        # Run benchmark
        results = run_benchmark(model, tokenizer)
        
        # Save results
        save_results(results, "baseline_results.json")
        
        # Print summary
        print_summary(results)
        
        print("\n" + "="*60)
        print("✓ Script 2 completed successfully!")
        print("="*60 + "\n")
        
    except Exception as e:
        print(f"\n✗ Error: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()

