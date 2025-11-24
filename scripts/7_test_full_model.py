#!/usr/bin/env python3
"""
Script 7: Test Full Fine-Tuned Model
Version: 1.0.0

Purpose: Test the fully fine-tuned model on benchmark questions.

Usage:
    python scripts/7_test_full_model.py

Input:
    - models/full_finetuned_model/: Full model from script 6

Output:
    - results/full_model_results.json: Full model benchmark results
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

MODELS_DIR = os.getenv("MODELS_DIR", "./models")
RESULTS_DIR = os.getenv("RESULTS_DIR", "./results")
MAX_SEQ_LENGTH = int(os.getenv("MAX_SEQ_LENGTH", "2048"))

Path(RESULTS_DIR).mkdir(parents=True, exist_ok=True)

FULL_MODEL_PATH = os.path.join(MODELS_DIR, "full_finetuned_model")

# Define questions (same as script 2)
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


def load_full_model():
    """Load the fully fine-tuned model."""
    if not os.path.exists(FULL_MODEL_PATH):
        print(f"ERROR: Full model not found at {FULL_MODEL_PATH}")
        print("Please run script 6 first to train the full model.")
        sys.exit(1)
    
    print(f"Loading full model from {FULL_MODEL_PATH}...")
    
    tokenizer = AutoTokenizer.from_pretrained(
        FULL_MODEL_PATH,
        trust_remote_code=True
    )
    
    model = AutoModelForCausalLM.from_pretrained(
        FULL_MODEL_PATH,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
        max_length=MAX_SEQ_LENGTH
    )
    
    return model, tokenizer


def get_model_response(model, tokenizer, question: str):
    """Get response from full model."""
    prompt = f"User: {question}RetryRMfinish"
    
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
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
    
    full_response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    response = full_response[len(prompt):].strip()
    
    return response, generation_time


def run_benchmark(model, tokenizer):
    """Run benchmark on all questions."""
    results = []
    
    print(f"\n{'='*60}")
    print(f"Running Full Model Benchmark: {len(AUTOIMMUNE_QUESTIONS)} Questions")
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
    """Save results to JSON file."""
    filepath = os.path.join(RESULTS_DIR, filename)
    with open(filepath, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"✓ Results saved to {filepath}")


def print_summary(results):
    """Print summary statistics."""
    print(f"\n{'='*60}")
    print("FULL MODEL BENCHMARK SUMMARY")
    print(f"{'='*60}\n")
    
    total_time = sum(r['time_seconds'] for r in results)
    avg_time = total_time / len(results) if results else 0
    
    print(f"Total Questions: {len(results)}")
    print(f"Total Time: {total_time:.2f} seconds")
    print(f"Average Time per Question: {avg_time:.2f} seconds")


def main():
    """Main execution function."""
    print("\n" + "="*60)
    print("SCRIPT 7: TEST FULL FINE-TUNED MODEL")
    print("="*60 + "\n")
    
    try:
        # Load full model
        model, tokenizer = load_full_model()
        
        # Run benchmark
        results = run_benchmark(model, tokenizer)
        
        # Save results
        save_results(results, "full_model_results.json")
        
        # Print summary
        print_summary(results)
        
        print("\n" + "="*60)
        print("✓ Script 7 completed successfully!")
        print("="*60 + "\n")
        
    except Exception as e:
        print(f"\n✗ Error: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()

