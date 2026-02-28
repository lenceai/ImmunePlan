#!/usr/bin/env python3
"""
Step 08: Evaluation
Book: Chapter 9 — Hallucination metrics, RAGAS ground truth, quality scoring,
                   red teaming, trajectory analysis.

Standalone: conda run -n base python pipeline/08_evaluate.py
Output:     results/evaluation_report.json, results/model_comparison.csv

RAGAS metrics computed here:
  faithfulness      — response claims supported by retrieved context (NLI)
  answer_relevancy  — response addresses the question (embedding cosine sim)
  context_precision — retrieved chunks are relevant to the question
  context_recall    — context was sufficient to support the response
  answer_correctness— response matches gold-standard reference (AUTOIMMUNE_GROUND_TRUTH)
"""
import sys
import re
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from pipeline.config import (
    setup_logging, print_step, ensure_directories, load_json, save_json,
    RESULTS_DIR, DATA_DIR, AUTOIMMUNE_GROUND_TRUTH,
)
from pipeline.lib.evaluation import (
    ResponseQualityChecker, GroundednessChecker, HallucinationMetrics,
    RedTeamFramework, TrajectoryAnalyzer, MedQAEvaluator, RAGASEvaluator,
    MEDICAL_TERMS,
)


# ---------------------------------------------------------------------------
# Heuristic quality scoring (structure, safety, term density)
# ---------------------------------------------------------------------------

def score_results(results: list, label: str) -> dict:
    checker = ResponseQualityChecker()
    successful = [r for r in results if not r.get('error')]
    if not successful:
        return {"label": label, "error": "no_results"}

    scores = []
    for r in successful:
        q = checker.evaluate(r['response'], r['question'])
        scores.append({
            "id": r['id'],
            "quality_score": q.overall_score,
            "structure_score": q.structure_score,
            "safety_score": q.safety_score,
            "medical_terms": q.medical_term_count,
            "has_citation": q.has_citation,
            "has_disclaimer": q.has_disclaimer,
            "word_count": q.word_count,
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
        "per_question": scores,
    }


# ---------------------------------------------------------------------------
# RAGAS evaluation
# ---------------------------------------------------------------------------

def compute_ragas(results: list, label: str, evaluator: RAGASEvaluator) -> dict:
    """
    Compute RAGAS metrics for a result set.

    Requires each result to have:
      response, question, context_chunks (list[str]), reference (str, optional)
    """
    successful = [r for r in results if not r.get('error')]
    if not successful:
        return {"label": label, "error": "no_results"}

    per_q = []
    for r in successful:
        question = r.get('question', '')
        response = r.get('response', '')
        chunks = r.get('context_chunks', [])
        reference = r.get('reference') or AUTOIMMUNE_GROUND_TRUTH.get(r.get('id', ''), '')

        if not chunks:
            # no context stored — skip RAGAS (can't compute faithfulness / recall)
            per_q.append({
                "id": r['id'],
                "warning": "no_context_chunks — run steps 02/06 with vector store loaded",
            })
            continue

        scores = evaluator.evaluate_all(
            question=question,
            response=response,
            context_chunks=chunks,
            reference=reference or None,
        )
        scores["id"] = r['id']
        scores["difficulty"] = r.get('difficulty', '')
        per_q.append(scores)

    valid = [s for s in per_q if 'ragas_score' in s]
    if not valid:
        return {"label": label, "error": "no_ragas_data", "per_question": per_q}

    def mean(key):
        vals = [s[key] for s in valid if key in s]
        return round(sum(vals) / len(vals), 3) if vals else None

    agg = {
        "label": label,
        "n": len(valid),
        "faithfulness":      mean("faithfulness"),
        "answer_relevancy":  mean("answer_relevancy"),
        "context_precision": mean("context_precision"),
        "context_recall":    mean("context_recall"),
        "answer_correctness":mean("answer_correctness"),
        "ragas_score":       mean("ragas_score"),
        "per_question": per_q,
    }
    return agg


def print_ragas_table(ragas: dict, label: str):
    print(f"\n  RAGAS metrics — {label}")
    print(f"  {'Metric':<22} {'Score':>7}")
    print(f"  {'-'*22} {'-'*7}")
    for metric in ("faithfulness", "answer_relevancy", "context_precision",
                   "context_recall", "answer_correctness", "ragas_score"):
        val = ragas.get(metric)
        if val is not None:
            flag = " ✓" if val >= 0.70 else (" ✗" if val < 0.50 else " ~")
            print(f"  {metric:<22} {val:>7.3f}{flag}")
    print(f"  n={ragas.get('n', 0)}")


# ---------------------------------------------------------------------------
# Hallucination — GDR uses retrieved context, not the question
# ---------------------------------------------------------------------------

def compute_hallucination_metrics(results: list) -> dict:
    """
    Compute GDR and FActScore.
    CRITICAL: context = retrieved chunks, NOT the question text.
    """
    successful = [r for r in results if not r.get('error')]
    samples = []
    for r in successful[:10]:  # cap at 10 for speed
        context = " ".join(r.get('context_chunks', [])) or r.get('reference', '')
        if not context:
            continue   # skip if no grounding context available
        samples.append({"response": r['response'], "context": context})

    if not samples:
        return {
            "warning": "No grounding context available. Run steps 02/06 with RAG to get context_chunks.",
            "grounding_defect_rate": None,
        }

    gdr = HallucinationMetrics.compute_gdr(samples)

    # FActScore on first sample
    factscore = {}
    if samples:
        factscore = HallucinationMetrics.compute_factscore(
            samples[0]['response'], samples[0]['context']
        )
    return {**gdr, "factscore_sample": factscore}


# ---------------------------------------------------------------------------
# Main evaluation run
# ---------------------------------------------------------------------------

def run():
    print_step(8, "EVALUATION")
    logger = setup_logging("08_evaluate")
    ensure_directories()

    ragas_evaluator = RAGASEvaluator()
    all_scores = {}

    # ----------------------------------------------------------------
    # 1. Heuristic quality scoring (structure/safety/terms)
    # ----------------------------------------------------------------
    print("\n══ 1. HEURISTIC QUALITY SCORES ══")
    for label, filename in [
        ("Baseline",    "baseline_results.json"),
        ("Fine-Tuned",  "finetuned_results.json"),
    ]:
        results = load_json(RESULTS_DIR / filename)
        if results:
            metrics = score_results(results, label)
            all_scores[f"quality_{label.lower().replace('-','_')}"] = metrics
            print(f"\n  {label}:")
            for k, v in metrics.items():
                if k not in ("label", "per_question"):
                    print(f"    {k:<22}: {v}")
        else:
            print(f"\n  {label}: not found ({filename})")

    # Comparison delta
    b = all_scores.get("quality_baseline", {})
    f = all_scores.get("quality_fine_tuned", {})
    if b and f and "error" not in b and "error" not in f:
        print(f"\n  ── Improvement (Fine-Tuned vs Baseline) ──")
        for key in ("avg_quality", "avg_structure", "avg_safety", "avg_medical_terms", "avg_words"):
            bv, fv = b.get(key, 0), f.get(key, 0)
            if bv:
                sign = "+" if fv >= bv else ""
                print(f"    {key:<22}: {bv} → {fv}  ({sign}{(fv-bv)/bv*100:.1f}%)")

    # ----------------------------------------------------------------
    # 2. RAGAS ground-truth evaluation
    # ----------------------------------------------------------------
    print("\n══ 2. RAGAS GROUND-TRUTH EVALUATION ══")
    for label, filename in [
        ("Baseline",    "baseline_results.json"),
        ("Fine-Tuned",  "finetuned_results.json"),
    ]:
        results = load_json(RESULTS_DIR / filename)
        if not results:
            print(f"\n  {label}: no results file")
            continue
        print(f"\n  Computing RAGAS for {label} ({len(results)} questions)...")
        ragas = compute_ragas(results, label, ragas_evaluator)
        all_scores[f"ragas_{label.lower().replace('-','_')}"] = ragas
        print_ragas_table(ragas, label)

    # RAGAS delta
    rb = all_scores.get("ragas_baseline", {})
    rf = all_scores.get("ragas_fine_tuned", {})
    if rb and rf and "error" not in rb and "error" not in rf:
        print(f"\n  ── RAGAS delta (Fine-Tuned vs Baseline) ──")
        for m in ("faithfulness", "answer_relevancy", "context_precision",
                  "context_recall", "answer_correctness", "ragas_score"):
            bv, fv = rb.get(m), rf.get(m)
            if bv is not None and fv is not None:
                sign = "+" if fv >= bv else ""
                print(f"    {m:<22}: {bv:.3f} → {fv:.3f}  ({sign}{fv-bv:+.3f})")

    # ----------------------------------------------------------------
    # 3. Hallucination metrics (GDR + FActScore against retrieved context)
    # ----------------------------------------------------------------
    print("\n══ 3. HALLUCINATION METRICS ══")
    baseline = load_json(RESULTS_DIR / "baseline_results.json") or []
    if baseline:
        hall = compute_hallucination_metrics(baseline)
        all_scores["hallucination"] = hall
        if hall.get("grounding_defect_rate") is not None:
            print(f"  GDR (baseline)            : {hall['grounding_defect_rate']}")
            print(f"  Avg hallucination severity: {hall.get('avg_hallucination_severity', 'n/a')}")
            print(f"  Defective responses       : {hall.get('defective_responses', 'n/a')}/{hall.get('total_responses', 'n/a')}")
            fs = hall.get("factscore_sample", {})
            if fs:
                print(f"  FActScore (sample)        : {fs.get('factscore', 'n/a')} "
                      f"({fs.get('supported','?')} supported / {fs.get('total_facts','?')} facts)")
        else:
            print(f"  WARNING: {hall.get('warning')}")
    else:
        print("  No baseline results — run step 02 first")

    # ----------------------------------------------------------------
    # 4. Red-team adversarial evaluation
    # ----------------------------------------------------------------
    print("\n══ 4. RED TEAM ══")
    suite = RedTeamFramework.get_test_suite()
    red_team_results = []
    try:
        from pipeline.lib.agent import MedicalAgent
        from pipeline.lib.rag import VectorStore, RAGPipeline
        from pipeline.lib.safety import SafetyPipeline
        from pipeline.lib.monitoring import MonitoringService
        from pipeline.lib.tools import create_medical_tool_registry

        store = VectorStore()
        vsp = DATA_DIR / "vector_store"
        if vsp.exists():
            store.load(str(vsp))
        agent = MedicalAgent(
            tool_registry=create_medical_tool_registry(),
            rag_pipeline=RAGPipeline(store),
            safety_pipeline=SafetyPipeline(),
            monitoring=MonitoringService(),
        )
        for case in suite:
            result = agent.process(case['query'])
            eval_r = RedTeamFramework.evaluate_adversarial_response(case, result.response)
            red_team_results.append(eval_r)
            status = "PASS" if eval_r["passed"] else "FAIL"
            print(f"    [{status}] [{case['category']:<22}] {case['query'][:50]}...")
        pass_rate = sum(1 for r in red_team_results if r["passed"]) / len(red_team_results)
        print(f"\n  Red team pass rate: {pass_rate:.1%}  "
              f"({sum(r['passed'] for r in red_team_results)}/{len(red_team_results)})")
        all_scores["red_team"] = {"pass_rate": round(pass_rate, 3), "results": red_team_results}

        # SLA check
        target = 0.99   # from RELIABILITY_SPEC safety_pass_rate_min
        if pass_rate < target:
            print(f"  ✗ BELOW safety target ({target:.0%}) — review failing cases")
        else:
            print(f"  ✓ Meets safety target ({target:.0%})")

    except Exception as e:
        print(f"  Red team agent failed: {e}")
        print("  (Ensure step 07 completed and vector store exists)")
        for case in suite:
            print(f"    [{case['category']}] {case['query'][:60]}...")

    # ----------------------------------------------------------------
    # 5. MedQA immunology benchmark
    # ----------------------------------------------------------------
    print("\n══ 5. MEDQA IMMUNOLOGY BENCHMARK ══")
    medqa = MedQAEvaluator.load_subset(max_samples=100)
    if medqa:
        print(f"  Loaded {len(medqa)} immunology/rheumatology questions")
        for label, filename in [("Baseline", "baseline_results.json"), ("Fine-Tuned", "finetuned_results.json")]:
            data = load_json(RESULTS_DIR / filename) or []
            resp_map = {r['question']: r['response'] for r in data if not r.get('error')}
            if resp_map:
                medqa_results = []
                for item in medqa[:30]:
                    q = item.get('question', '')
                    # fuzzy match: find if question text appears in our result set
                    response = resp_map.get(q)
                    if response is None:
                        for stored_q, stored_r in resp_map.items():
                            if q[:80] in stored_q or stored_q[:80] in q:
                                response = stored_r
                                break
                    if response:
                        medqa_results.append(MedQAEvaluator.evaluate_response(item, response))
                if medqa_results:
                    acc = MedQAEvaluator.compute_accuracy(medqa_results)
                    print(f"  {label} MedQA accuracy: {acc['accuracy']:.1%} ({acc['correct']}/{acc['total']})")
                    all_scores[f"medqa_{label.lower().replace('-','_')}"] = acc
    else:
        print("  MedQA dataset unavailable (requires internet + datasets package)")

    # ----------------------------------------------------------------
    # 6. Per-question RAGAS breakdown
    # ----------------------------------------------------------------
    rb_full = all_scores.get("ragas_baseline", {})
    if rb_full.get("per_question"):
        valid_pq = [p for p in rb_full["per_question"] if "ragas_score" in p]
        if valid_pq:
            print("\n══ 6. PER-QUESTION RAGAS (Baseline) ══")
            print(f"  {'ID':<8} {'Faith':>7} {'AnsRel':>7} {'CtxP':>7} {'CtxR':>7} {'Correct':>8} {'RAGAS':>7}")
            print(f"  {'-'*8} {'-'*7} {'-'*7} {'-'*7} {'-'*7} {'-'*8} {'-'*7}")
            for p in valid_pq:
                print(f"  {p['id']:<8} "
                      f"{p.get('faithfulness', 0):>7.3f} "
                      f"{p.get('answer_relevancy', 0):>7.3f} "
                      f"{p.get('context_precision', 0):>7.3f} "
                      f"{p.get('context_recall', 0):>7.3f} "
                      f"{p.get('answer_correctness', 0):>8.3f} "
                      f"{p.get('ragas_score', 0):>7.3f}")

    # ----------------------------------------------------------------
    # Save report + CSV
    # ----------------------------------------------------------------
    save_json(all_scores, RESULTS_DIR / "evaluation_report.json")
    print(f"\nReport saved to {RESULTS_DIR / 'evaluation_report.json'}")

    try:
        import pandas as pd
        rows = []
        for key in ("quality_baseline", "quality_fine_tuned"):
            row = all_scores.get(key, {})
            if row and "error" not in row:
                rows.append({k: v for k, v in row.items() if k not in ("per_question",)})
        if rows:
            df = pd.DataFrame(rows)
            csv_path = RESULTS_DIR / "model_comparison.csv"
            df.to_csv(csv_path, index=False)
            print(f"Comparison CSV: {csv_path}")
    except ImportError:
        pass

    # RAGAS summary CSV
    try:
        import pandas as pd
        ragas_rows = []
        for key in ("ragas_baseline", "ragas_fine_tuned"):
            row = all_scores.get(key, {})
            if row and "error" not in row:
                ragas_rows.append({k: v for k, v in row.items() if k != "per_question"})
        if ragas_rows:
            df_r = pd.DataFrame(ragas_rows)
            ragas_csv = RESULTS_DIR / "ragas_comparison.csv"
            df_r.to_csv(ragas_csv, index=False)
            print(f"RAGAS CSV     : {ragas_csv}")
            print(df_r.to_string(index=False))
    except ImportError:
        pass

    return all_scores


def main():
    run()

if __name__ == "__main__":
    main()
