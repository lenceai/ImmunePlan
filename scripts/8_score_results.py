#!/usr/bin/env python3
"""
Script 8: Score and Compare Results
Version: 1.1.0

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

import sys
import re
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional

import pandas as pd

# Add scripts directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from common import (
    Config,
    setup_logging,
    load_json,
    print_header,
    print_section,
)


# =============================================================================
# METRICS CALCULATION
# =============================================================================

# Medical terminology for scoring
MEDICAL_TERMS = [
    "diagnosis", "diagnostic", "criteria", "antibody", "antibodies",
    "autoimmune", "lupus", "rheumatoid", "arthritis", "syndrome",
    "sclerosis", "vasculitis", "polymyositis", "sjogren", "behçet",
    "ANA", "RF", "anti-CCP", "anti-dsDNA", "complement", "biopsy",
    "pathophysiology", "etiology", "prognosis", "treatment", "therapy",
    "inflammation", "immunosuppressive", "corticosteroid", "DMARD",
    "cytokine", "autoantibody", "serological", "clinical"
]


def calculate_metrics(results: List[Dict], logger=None) -> Optional[Dict]:
    """
    Calculate comprehensive metrics for a set of results.
    
    Args:
        results: List of result dictionaries
        logger: Optional logger
    
    Returns:
        Dictionary of metrics
    """
    if not results:
        return None
    
    successful = [r for r in results if not r.get('error')]
    
    if not successful:
        return {"error": "No successful results"}
    
    # Basic metrics
    metrics = {
        "total_questions": len(results),
        "successful_responses": len(successful),
        "success_rate": len(successful) / len(results) * 100,
        "avg_time": sum(r['time_seconds'] for r in successful) / len(successful),
        "total_time": sum(r['time_seconds'] for r in successful),
    }
    
    # Word and token counts
    word_counts = [r.get('word_count', len(r.get('response', '').split())) for r in successful]
    metrics["avg_word_count"] = sum(word_counts) / len(word_counts)
    metrics["min_word_count"] = min(word_counts)
    metrics["max_word_count"] = max(word_counts)
    
    # Quality metrics
    reasoning_count = 0
    medical_terms_total = 0
    structured_count = 0
    confidence_count = 0
    next_steps_count = 0
    
    for result in successful:
        response = result.get('response', '').lower()
        
        # Check for reasoning tags
        if "<think>" in response or "<reasoning>" in response or "let me" in response:
            reasoning_count += 1
        
        # Medical terminology count
        term_count = sum(1 for term in MEDICAL_TERMS if term.lower() in response)
        medical_terms_total += term_count
        
        # Structured output (lists, numbered items)
        if re.search(r'\d+\.|•|[-*]\s', response):
            structured_count += 1
        
        # Confidence mentioned
        confidence_words = ["confidence", "likely", "probable", "certain", "uncertain", "suggest"]
        if any(word in response for word in confidence_words):
            confidence_count += 1
        
        # Next steps mentioned
        next_step_phrases = ["next step", "further", "additional", "recommend", "suggest", "follow-up"]
        if any(phrase in response for phrase in next_step_phrases):
            next_steps_count += 1
    
    # Calculate percentages
    n = len(successful)
    metrics["reasoning_rate"] = (reasoning_count / n) * 100
    metrics["avg_medical_terms"] = medical_terms_total / n
    metrics["structured_rate"] = (structured_count / n) * 100
    metrics["confidence_rate"] = (confidence_count / n) * 100
    metrics["next_steps_rate"] = (next_steps_count / n) * 100
    
    # By difficulty
    metrics["by_difficulty"] = {}
    for result in successful:
        diff = result['difficulty']
        if diff not in metrics["by_difficulty"]:
            metrics["by_difficulty"][diff] = {"count": 0, "total_time": 0}
        metrics["by_difficulty"][diff]["count"] += 1
        metrics["by_difficulty"][diff]["total_time"] += result['time_seconds']
    
    for diff in metrics["by_difficulty"]:
        count = metrics["by_difficulty"][diff]["count"]
        total = metrics["by_difficulty"][diff]["total_time"]
        metrics["by_difficulty"][diff]["avg_time"] = total / count if count > 0 else 0
    
    return metrics


# =============================================================================
# COMPARISON & REPORTING
# =============================================================================

def create_comparison_table(all_metrics: Dict[str, Dict]) -> pd.DataFrame:
    """Create comparison table from metrics."""
    data = []
    
    for model_name, metrics in all_metrics.items():
        if metrics and "error" not in metrics:
            data.append({
                "Model": model_name,
                "Success Rate %": f"{metrics['success_rate']:.1f}",
                "Avg Time (s)": f"{metrics['avg_time']:.2f}",
                "Avg Words": f"{metrics['avg_word_count']:.0f}",
                "Medical Terms": f"{metrics['avg_medical_terms']:.1f}",
                "Reasoning %": f"{metrics['reasoning_rate']:.0f}",
                "Structured %": f"{metrics['structured_rate']:.0f}",
                "Confidence %": f"{metrics['confidence_rate']:.0f}",
                "Next Steps %": f"{metrics['next_steps_rate']:.0f}"
            })
    
    return pd.DataFrame(data)


def generate_report(all_metrics: Dict[str, Dict], logger) -> str:
    """Generate comprehensive final report."""
    lines = []
    
    lines.append("=" * 70)
    lines.append("AUTOIMMUNE LLM FINE-TUNING - FINAL REPORT")
    lines.append("=" * 70)
    lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append("")
    
    # Executive Summary
    lines.append("EXECUTIVE SUMMARY")
    lines.append("-" * 70)
    
    baseline = all_metrics.get("Baseline")
    qlora = all_metrics.get("QLoRA")
    full = all_metrics.get("Full Fine-Tuned")
    
    if baseline:
        lines.append(f"\nBaseline Model:")
        lines.append(f"  • Average Response Time: {baseline['avg_time']:.2f}s")
        lines.append(f"  • Average Word Count: {baseline['avg_word_count']:.0f}")
        lines.append(f"  • Medical Terminology Score: {baseline['avg_medical_terms']:.1f}")
        lines.append(f"  • Structured Responses: {baseline['structured_rate']:.0f}%")
    
    if qlora:
        lines.append(f"\nQLoRA Fine-Tuned Model:")
        lines.append(f"  • Average Response Time: {qlora['avg_time']:.2f}s")
        lines.append(f"  • Average Word Count: {qlora['avg_word_count']:.0f}")
        lines.append(f"  • Medical Terminology Score: {qlora['avg_medical_terms']:.1f}")
        lines.append(f"  • Structured Responses: {qlora['structured_rate']:.0f}%")
        
        if baseline:
            lines.append(f"\n  Improvement vs Baseline:")
            time_diff = baseline['avg_time'] - qlora['avg_time']
            word_diff = qlora['avg_word_count'] - baseline['avg_word_count']
            med_diff = qlora['avg_medical_terms'] - baseline['avg_medical_terms']
            lines.append(f"    • Time: {time_diff:+.2f}s ({-time_diff/baseline['avg_time']*100:+.1f}%)")
            lines.append(f"    • Word Count: {word_diff:+.0f} ({word_diff/baseline['avg_word_count']*100:+.1f}%)")
            lines.append(f"    • Medical Terms: {med_diff:+.1f}")
    
    if full:
        lines.append(f"\nFull Fine-Tuned Model:")
        lines.append(f"  • Average Response Time: {full['avg_time']:.2f}s")
        lines.append(f"  • Average Word Count: {full['avg_word_count']:.0f}")
        lines.append(f"  • Medical Terminology Score: {full['avg_medical_terms']:.1f}")
        lines.append(f"  • Structured Responses: {full['structured_rate']:.0f}%")
    
    # Recommendations
    lines.append("\n" + "=" * 70)
    lines.append("RECOMMENDATIONS")
    lines.append("-" * 70)
    
    if qlora and baseline:
        if qlora['avg_medical_terms'] > baseline['avg_medical_terms']:
            lines.append("✓ QLoRA model shows improved medical terminology usage")
        if qlora['structured_rate'] > baseline['structured_rate']:
            lines.append("✓ QLoRA model produces more structured responses")
        if qlora['avg_time'] < baseline['avg_time']:
            lines.append("✓ QLoRA model has faster response times")
    
    lines.append("\n• For production use: Deploy QLoRA model for best efficiency/quality balance")
    lines.append("• For maximum quality: Consider full fine-tuning if VRAM available")
    lines.append("• For detailed comparison: See results/model_comparison.csv")
    
    # Footer
    lines.append("\n" + "=" * 70)
    lines.append("END OF REPORT")
    lines.append("=" * 70)
    
    return "\n".join(lines)


# =============================================================================
# MAIN
# =============================================================================

def main():
    """Main execution function."""
    print_header("SCRIPT 8: SCORE AND COMPARE RESULTS")
    
    # Setup
    logger = setup_logging("8_score_results")
    Config.ensure_directories()
    
    try:
        # Load all results
        results_files = {
            "Baseline": "baseline_results.json",
            "QLoRA": "qlora_results.json",
            "Full Fine-Tuned": "full_model_results.json"
        }
        
        all_results = {}
        all_metrics = {}
        
        for model_name, filename in results_files.items():
            filepath = Config.RESULTS_DIR / filename
            results = load_json(filepath)
            
            if results:
                all_results[model_name] = results
                metrics = calculate_metrics(results, logger)
                all_metrics[model_name] = metrics
                print(f"✓ Loaded {model_name}: {len(results)} results")
                logger.info(f"Loaded {model_name}: {len(results)} results")
            else:
                if model_name == "Full Fine-Tuned":
                    print(f"ℹ {model_name} results not found (optional)")
                else:
                    print(f"⚠ {model_name} results not found")
                logger.warning(f"{model_name} results not found")
        
        if not all_results:
            print("\n✗ ERROR: No results files found!")
            print("Please run scripts 2, 5, and/or 7 first.")
            sys.exit(1)
        
        # Create comparison table
        df = create_comparison_table(all_metrics)
        
        if not df.empty:
            csv_path = Config.RESULTS_DIR / "model_comparison.csv"
            df.to_csv(csv_path, index=False)
            print(f"\n✓ Comparison table saved to {csv_path}")
            logger.info(f"Comparison saved to {csv_path}")
            
            print_section("COMPARISON TABLE")
            print(df.to_string(index=False))
        
        # Generate report
        report = generate_report(all_metrics, logger)
        
        report_path = Config.RESULTS_DIR / "final_report.txt"
        with open(report_path, 'w') as f:
            f.write(report)
        print(f"\n✓ Final report saved to {report_path}")
        logger.info(f"Report saved to {report_path}")
        
        # Print report
        print("\n" + report)
        
        print_header("✓ Script 8 completed successfully!")
        logger.info("Script completed successfully")
        
    except Exception as e:
        logger.error(f"Error: {str(e)}", exc_info=True)
        print(f"\n✗ Error: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
