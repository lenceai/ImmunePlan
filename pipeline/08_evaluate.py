#!/usr/bin/env python3
"""
Step 08: Evaluation
Book: Chapter 9 — Hallucination metrics, quality scoring, red teaming, trajectory analysis.

Standalone: python pipeline/08_evaluate.py
Output:     results/evaluation_report.json, results/model_comparison.csv
"""
import sys
import re
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from pipeline.config import (
    setup_logging, print_step, ensure_directories, load_json, save_json,
    RESULTS_DIR,
)
from pipeline.lib.evaluation import (
    ResponseQualityChecker, GroundednessChecker, HallucinationMetrics,
    RedTeamFramework, TrajectoryAnalyzer, MEDICAL_TERMS,
)


def score_results(results, label):
    """Score a set of benchmark results."""
    checker = ResponseQualityChecker()
    groundedness = GroundednessChecker()
    successful = [r for r in results if not r.get('error')]
    if not successful:
        return {"label": label, "error": "no_results"}

    scores = []
    for r in successful:
        quality = checker.evaluate(r['response'], r['question'])
        scores.append({
            "id": r['id'],
            "quality_score": quality.overall_score,
            "structure_score": quality.structure_score,
            "safety_score": quality.safety_score,
            "medical_terms": quality.medical_term_count,
            "has_citation": quality.has_citation,
            "has_disclaimer": quality.has_disclaimer,
            "word_count": quality.word_count,
        })

    avg = lambda key: sum(s[key] for s in scores) / len(scores)
    return {
        "label": label,
        "total": len(results),
        "successful": len(successful),
        "avg_quality": round(avg("quality_score"), 3),
        "avg_structure": round(avg("structure_score"), 3),
        "avg_safety": round(avg("safety_score"), 3),
        "avg_medical_terms": round(avg("medical_terms"), 1),
        "avg_words": round(avg("word_count"), 0),
        "citation_rate": round(sum(1 for s in scores if s["has_citation"]) / len(scores), 2),
        "disclaimer_rate": round(sum(1 for s in scores if s["has_disclaimer"]) / len(scores), 2),
        "avg_time": round(sum(r.get('time_seconds', 0) for r in successful) / len(successful), 2),
    }


def run():
    print_step(8, "EVALUATION")
    logger = setup_logging("08_evaluate")
    ensure_directories()

    # --- Score baseline and fine-tuned ---
    all_scores = {}
    for label, filename in [("Baseline", "baseline_results.json"), ("Fine-Tuned", "finetuned_results.json")]:
        results = load_json(RESULTS_DIR / filename)
        if results:
            metrics = score_results(results, label)
            all_scores[label] = metrics
            print(f"\n{label}:")
            for k, v in metrics.items():
                if k != "label":
                    print(f"  {k}: {v}")
        else:
            print(f"\n{label}: no results file found")

    if len(all_scores) > 1:
        b, f = all_scores.get("Baseline", {}), all_scores.get("Fine-Tuned", {})
        if b and f:
            print(f"\nImprovement (Fine-Tuned vs Baseline):")
            for key in ["avg_quality", "avg_medical_terms", "avg_words"]:
                bv, fv = b.get(key, 0), f.get(key, 0)
                if bv:
                    print(f"  {key}: {bv} → {fv} ({(fv-bv)/bv*100:+.1f}%)")

    # --- Hallucination metrics ---
    print("\n--- Hallucination Analysis ---")
    baseline = load_json(RESULTS_DIR / "baseline_results.json") or []
    if baseline:
        samples = [{"response": r["response"], "context": r["question"]} for r in baseline if not r.get("error")][:5]
        gdr = HallucinationMetrics.compute_gdr(samples)
        print(f"  GDR: {gdr['grounding_defect_rate']}")
        print(f"  Severity: {gdr['avg_hallucination_severity']}")
        all_scores["hallucination"] = gdr

    # --- Red teaming ---
    print("\n--- Red Team Results ---")
    suite = RedTeamFramework.get_test_suite()
    print(f"  Adversarial test cases: {len(suite)}")
    for case in suite[:3]:
        print(f"    [{case['category']}] {case['query'][:60]}...")

    # --- Save report ---
    save_json(all_scores, RESULTS_DIR / "evaluation_report.json")
    print(f"\nReport saved to {RESULTS_DIR / 'evaluation_report.json'}")

    # --- CSV comparison ---
    try:
        import pandas as pd
        rows = [v for v in all_scores.values() if isinstance(v, dict) and "label" in v]
        if rows:
            df = pd.DataFrame(rows)
            csv_path = RESULTS_DIR / "model_comparison.csv"
            df.to_csv(csv_path, index=False)
            print(f"Comparison table saved to {csv_path}")
            print(df.to_string(index=False))
    except ImportError:
        pass

    return all_scores


def main():
    run()

if __name__ == "__main__":
    main()
