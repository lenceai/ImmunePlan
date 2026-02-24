#!/usr/bin/env python3
"""
ZenML Pipeline Runner — Orchestrates all 10 steps.

If ZenML is installed, runs as a ZenML pipeline.
Otherwise, runs steps sequentially.

Usage:
    python pipeline/run_pipeline.py              # Auto-detect ZenML
    python pipeline/run_pipeline.py --sequential  # Force sequential
    python pipeline/run_pipeline.py --skip 01,02  # Skip specific steps
"""
import sys
import importlib
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

STEPS = [
    ("01", "01_setup",        "Setup & Model Download"),
    ("02", "02_baseline",     "Prompt Engineering Baseline"),
    ("03", "03_collect_data", "Collect Training Data"),
    ("04", "04_build_rag",    "Build RAG Pipeline"),
    ("05", "05_finetune",     "Adaptive Fine-Tuning"),
    ("06", "06_test_model",   "Test Fine-Tuned Model"),
    ("07", "07_build_agent",  "Build Medical Agent"),
    ("08", "08_evaluate",     "Evaluation"),
    ("09", "09_safety",       "Safety & Responsible AI"),
    ("10", "10_deploy",       "Deploy with Monitoring"),
]


def run_sequential(skip=None):
    """Run all steps sequentially."""
    skip = skip or set()
    results = {}

    for step_num, module_name, title in STEPS:
        if step_num in skip:
            print(f"\n  Skipping step {step_num}: {title}")
            continue

        try:
            module = importlib.import_module(f"pipeline.{module_name}")
            result = module.run()
            results[step_num] = result
        except SystemExit:
            print(f"\n  Step {step_num} exited (likely missing GPU or data)")
            results[step_num] = {"status": "skipped"}
        except Exception as e:
            print(f"\n  Step {step_num} failed: {e}")
            results[step_num] = {"status": "failed", "error": str(e)}

            gpu_steps = {"02", "05", "06"}
            if step_num in gpu_steps:
                print(f"  (GPU step — continuing pipeline)")
                continue
            else:
                raise

    return results


def run_zenml_pipeline(skip=None):
    """Run as a ZenML pipeline (if ZenML is installed)."""
    try:
        from zenml import pipeline, step
    except ImportError:
        print("ZenML not installed. Running sequentially.")
        return run_sequential(skip)

    skip = skip or set()

    @step
    def step_01():
        if "01" not in skip:
            from pipeline import _01_setup as m
            return m.run(skip_model_download=True)

    @step
    def step_03():
        if "03" not in skip:
            from pipeline import _03_collect_data as m
            return m.run()

    @step
    def step_04():
        if "04" not in skip:
            from pipeline import _04_build_rag as m
            return m.run()

    @step
    def step_07():
        if "07" not in skip:
            from pipeline import _07_build_agent as m
            return m.run()

    @step
    def step_08():
        if "08" not in skip:
            from pipeline import _08_evaluate as m
            return m.run()

    @step
    def step_09():
        if "09" not in skip:
            from pipeline import _09_safety as m
            return m.run()

    @step
    def step_10():
        if "10" not in skip:
            from pipeline import _10_deploy as m
            return m.run()

    @pipeline
    def immuneplan_pipeline():
        s01 = step_01()
        s03 = step_03()
        s04 = step_04()
        s07 = step_07()
        s08 = step_08()
        s09 = step_09()
        s10 = step_10()

    immuneplan_pipeline()


def main():
    skip = set()
    if "--skip" in sys.argv:
        idx = sys.argv.index("--skip")
        if idx + 1 < len(sys.argv):
            skip = set(sys.argv[idx + 1].split(","))

    if "--sequential" in sys.argv or "--skip" in sys.argv:
        run_sequential(skip)
    else:
        run_zenml_pipeline(skip)


if __name__ == "__main__":
    main()
