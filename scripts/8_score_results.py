#!/usr/bin/env python3
"""
Script 8: Score and Compare Results
Version: 1.0.0

Purpose: Compare baseline, QLoRA, and full model results and generate reports.

Usage:
    python scripts/8_score_results.py

Input:
    - results/baseline_results.json
    - results/qlora_results.json
    - results/full_model_results.json (optional)

Output:
    - results/model_comparison.csv: Comparison table
    - results/final_report.txt: Executive summary
"""

import os
import sys
import json
import re
from pathlib import Path
from datetime import datetime
import pandas as pd
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

RESULTS_DIR = os.getenv("RESULTS_DIR", "./results")

Path(RESULTS_DIR).mkdir(parents=True, exist_ok=True)


def load_results(filename):
    """Load results from JSON file."""
    filepath = os.path.join(RESULTS_DIR, filename)
    
    if not os.path.exists(filepath):
        return None
    
    with open(filepath, 'r') as f:
        return json.load(f)


def calculate_metrics(results):
    """
    Calculate metrics for a set of results.
    
    Args:
        results: List of result dictionaries
    
    Returns:
        Dictionary of metrics
    """
    if not results:
        return None
    
    metrics = {
        "total_questions": len(results),
        "avg_time": sum(r['time_seconds'] for r in results) / len(results),
        "total_time": sum(r['time_seconds'] for r in results),
        "avg_word_count": 0,
        "avg_sentence_count": 0,
        "has_reasoning_tags": 0,
        "medical_terms_count": 0,
        "structured_output": 0,
        "confidence_mentioned": 0,
        "next_steps_mentioned": 0
    }
    
    word_counts = []
    sentence_counts = []
    
    medical_terms = [
        "diagnosis", "diagnostic", "criteria", "antibody", "antibodies",
        "autoimmune", "lupus", "rheumatoid", "arthritis", "syndrome",
        "sclerosis", "vasculitis", "polymyositis", "sjogren", "behçet",
        "ANA", "RF", "anti-CCP", "anti-dsDNA", "complement", "biopsy",
        "pathophysiology", "etiology", "prognosis", "treatment", "therapy"
    ]
    
    for result in results:
        response = result.get('response', '')
        
        # Word count
        words = len(response.split())
        word_counts.append(words)
        
        # Sentence count
        sentences = len(re.split(r'[.!?]+', response))
        sentence_counts.append(sentences)
        
        # Check for reasoning tags (DeepSeek-R1 format)
        if "<think>" in response.lower() or "<reasoning>" in response.lower():
            metrics["has_reasoning_tags"] += 1
        
        # Medical terminology
        response_lower = response.lower()
        term_count = sum(1 for term in medical_terms if term.lower() in response_lower)
        metrics["medical_terms_count"] += term_count
        
        # Structured output (check for lists, numbered items, etc.)
        if re.search(r'\d+\.|•|[-*]', response):
            metrics["structured_output"] += 1
        
        # Confidence mentioned
        if any(word in response_lower for word in ["confidence", "likely", "probable", "certain", "uncertain"]):
            metrics["confidence_mentioned"] += 1
        
        # Next steps mentioned
        if any(phrase in response_lower for phrase in ["next steps", "further", "additional", "recommend", "suggest"]):
            metrics["next_steps_mentioned"] += 1
    
    metrics["avg_word_count"] = sum(word_counts) / len(word_counts) if word_counts else 0
    metrics["avg_sentence_count"] = sum(sentence_counts) / len(sentence_counts) if sentence_counts else 0
    
    # Convert counts to percentages
    metrics["has_reasoning_tags_pct"] = (metrics["has_reasoning_tags"] / metrics["total_questions"]) * 100
    metrics["structured_output_pct"] = (metrics["structured_output"] / metrics["total_questions"]) * 100
    metrics["confidence_mentioned_pct"] = (metrics["confidence_mentioned"] / metrics["total_questions"]) * 100
    metrics["next_steps_mentioned_pct"] = (metrics["next_steps_mentioned"] / metrics["total_questions"]) * 100
    
    return metrics


def create_comparison_table(baseline_metrics, qlora_metrics, full_metrics):
    """Create comparison table."""
    data = []
    
    if baseline_metrics:
        data.append({
            "Model": "Baseline",
            "Avg Time (s)": f"{baseline_metrics['avg_time']:.2f}",
            "Avg Words": f"{baseline_metrics['avg_word_count']:.1f}",
            "Reasoning Tags %": f"{baseline_metrics['has_reasoning_tags_pct']:.1f}",
            "Medical Terms": f"{baseline_metrics['medical_terms_count']:.1f}",
            "Structured %": f"{baseline_metrics['structured_output_pct']:.1f}",
            "Confidence %": f"{baseline_metrics['confidence_mentioned_pct']:.1f}",
            "Next Steps %": f"{baseline_metrics['next_steps_mentioned_pct']:.1f}"
        })
    
    if qlora_metrics:
        data.append({
            "Model": "QLoRA",
            "Avg Time (s)": f"{qlora_metrics['avg_time']:.2f}",
            "Avg Words": f"{qlora_metrics['avg_word_count']:.1f}",
            "Reasoning Tags %": f"{qlora_metrics['has_reasoning_tags_pct']:.1f}",
            "Medical Terms": f"{qlora_metrics['medical_terms_count']:.1f}",
            "Structured %": f"{qlora_metrics['structured_output_pct']:.1f}",
            "Confidence %": f"{qlora_metrics['confidence_mentioned_pct']:.1f}",
            "Next Steps %": f"{qlora_metrics['next_steps_mentioned_pct']:.1f}"
        })
    
    if full_metrics:
        data.append({
            "Model": "Full Fine-Tuned",
            "Avg Time (s)": f"{full_metrics['avg_time']:.2f}",
            "Avg Words": f"{full_metrics['avg_word_count']:.1f}",
            "Reasoning Tags %": f"{full_metrics['has_reasoning_tags_pct']:.1f}",
            "Medical Terms": f"{full_metrics['medical_terms_count']:.1f}",
            "Structured %": f"{full_metrics['structured_output_pct']:.1f}",
            "Confidence %": f"{full_metrics['confidence_mentioned_pct']:.1f}",
            "Next Steps %": f"{full_metrics['next_steps_mentioned_pct']:.1f}"
        })
    
    df = pd.DataFrame(data)
    return df


def generate_report(baseline_metrics, qlora_metrics, full_metrics):
    """Generate final report."""
    report = []
    report.append("="*60)
    report.append("AUTOIMMUNE LLM FINE-TUNING - FINAL REPORT")
    report.append("="*60)
    report.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report.append("")
    
    report.append("EXECUTIVE SUMMARY")
    report.append("-"*60)
    
    if baseline_metrics:
        report.append(f"\nBaseline Model:")
        report.append(f"  - Average Response Time: {baseline_metrics['avg_time']:.2f}s")
        report.append(f"  - Average Word Count: {baseline_metrics['avg_word_count']:.1f}")
        report.append(f"  - Reasoning Tags Present: {baseline_metrics['has_reasoning_tags_pct']:.1f}%")
        report.append(f"  - Medical Terminology Score: {baseline_metrics['medical_terms_count']:.1f}")
    
    if qlora_metrics:
        report.append(f"\nQLoRA Fine-Tuned Model:")
        report.append(f"  - Average Response Time: {qlora_metrics['avg_time']:.2f}s")
        report.append(f"  - Average Word Count: {qlora_metrics['avg_word_count']:.1f}")
        report.append(f"  - Reasoning Tags Present: {qlora_metrics['has_reasoning_tags_pct']:.1f}%")
        report.append(f"  - Medical Terminology Score: {qlora_metrics['medical_terms_count']:.1f}")
        
        if baseline_metrics:
            time_improvement = ((baseline_metrics['avg_time'] - qlora_metrics['avg_time']) / baseline_metrics['avg_time']) * 100
            word_improvement = ((qlora_metrics['avg_word_count'] - baseline_metrics['avg_word_count']) / baseline_metrics['avg_word_count']) * 100
            report.append(f"\n  Improvement vs Baseline:")
            report.append(f"    - Time: {time_improvement:+.1f}%")
            report.append(f"    - Word Count: {word_improvement:+.1f}%")
    
    if full_metrics:
        report.append(f"\nFull Fine-Tuned Model:")
        report.append(f"  - Average Response Time: {full_metrics['avg_time']:.2f}s")
        report.append(f"  - Average Word Count: {full_metrics['avg_word_count']:.1f}")
        report.append(f"  - Reasoning Tags Present: {full_metrics['has_reasoning_tags_pct']:.1f}%")
        report.append(f"  - Medical Terminology Score: {full_metrics['medical_terms_count']:.1f}")
    
    report.append("\n" + "="*60)
    report.append("RECOMMENDATIONS")
    report.append("-"*60)
    
    if qlora_metrics and baseline_metrics:
        if qlora_metrics['medical_terms_count'] > baseline_metrics['medical_terms_count']:
            report.append("✓ QLoRA model shows improved medical terminology usage.")
        if qlora_metrics['has_reasoning_tags_pct'] > baseline_metrics['has_reasoning_tags_pct']:
            report.append("✓ QLoRA model shows improved reasoning capabilities.")
    
    report.append("\nFor detailed comparison, see: results/model_comparison.csv")
    
    return "\n".join(report)


def main():
    """Main execution function."""
    print("\n" + "="*60)
    print("SCRIPT 8: SCORE AND COMPARE RESULTS")
    print("="*60 + "\n")
    
    try:
        # Load results
        baseline_results = load_results("baseline_results.json")
        qlora_results = load_results("qlora_results.json")
        full_results = load_results("full_model_results.json")
        
        if not baseline_results:
            print("WARNING: Baseline results not found. Run script 2 first.")
        
        if not qlora_results:
            print("WARNING: QLoRA results not found. Run script 5 first.")
        
        if not full_results:
            print("INFO: Full model results not found. This is optional.")
        
        # Calculate metrics
        baseline_metrics = calculate_metrics(baseline_results) if baseline_results else None
        qlora_metrics = calculate_metrics(qlora_results) if qlora_results else None
        full_metrics = calculate_metrics(full_results) if full_results else None
        
        # Create comparison table
        df = create_comparison_table(baseline_metrics, qlora_metrics, full_metrics)
        
        if not df.empty:
            csv_path = os.path.join(RESULTS_DIR, "model_comparison.csv")
            df.to_csv(csv_path, index=False)
            print(f"✓ Comparison table saved to {csv_path}")
            print("\nComparison Table:")
            print(df.to_string(index=False))
        else:
            print("WARNING: No results to compare.")
        
        # Generate report
        report = generate_report(baseline_metrics, qlora_metrics, full_metrics)
        
        report_path = os.path.join(RESULTS_DIR, "final_report.txt")
        with open(report_path, 'w') as f:
            f.write(report)
        print(f"\n✓ Final report saved to {report_path}")
        print("\n" + report)
        
        print("\n" + "="*60)
        print("✓ Script 8 completed successfully!")
        print("="*60 + "\n")
        
    except Exception as e:
        print(f"\n✗ Error: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()

