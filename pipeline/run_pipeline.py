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
import gc
import subprocess
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
    ("5b", "05b_dpo",         "DPO Alignment"),
    ("06", "06_test_model",   "Test Fine-Tuned Model"),
    ("07", "07_build_agent",  "Build Medical Agent"),
    ("08", "08_evaluate",     "Evaluation"),
    ("09", "09_safety",       "Safety & Responsible AI"),
    ("10", "10_deploy",       "Deploy with Monitoring"),
]

GPU_STEPS = {"02", "05", "5b", "06", "07", "08"}


def _clear_gpu():
    """Release GPU memory between steps to avoid CUDA state corruption."""
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
            torch.cuda.synchronize()
    except Exception:
        pass
    gc.collect()


def _run_step_in_subprocess(module_name: str) -> int:
    """Run a pipeline step in a fresh subprocess to isolate CUDA context."""
    root = str(Path(__file__).parent.parent)
    script = (
        f"import sys, importlib; sys.path.insert(0, {root!r}); "
        f"m = importlib.import_module('pipeline.{module_name}'); m.run()"
    )
    result = subprocess.run(
        [sys.executable, "-c", script],
        cwd=root,
    )
    return result.returncode


def run_sequential(skip=None, isolate_gpu=True):
    """Run all steps sequentially.

    Args:
        skip: set of step numbers to skip.
        isolate_gpu: if True, run GPU-heavy steps in subprocesses so each
                     gets a clean CUDA context (prevents illegal-memory-access
                     errors from corrupted state).
    """
    skip = skip or set()
    results = {}

    for step_num, module_name, title in STEPS:
        if step_num in skip:
            print(f"\n  Skipping step {step_num}: {title}")
            continue

        use_subprocess = isolate_gpu and step_num in GPU_STEPS

        try:
            if use_subprocess:
                print(f"\n  Running step {step_num} in subprocess (GPU isolation)…")
                rc = _run_step_in_subprocess(module_name)
                if rc != 0:
                    raise RuntimeError(f"subprocess exited with code {rc}")
                results[step_num] = {"status": "ok"}
            else:
                module = importlib.import_module(f"pipeline.{module_name}")
                result = module.run()
                results[step_num] = result
        except SystemExit:
            print(f"\n  Step {step_num} exited (likely missing GPU or data)")
            results[step_num] = {"status": "skipped"}
        except Exception as e:
            print(f"\n  Step {step_num} failed: {e}")
            results[step_num] = {"status": "failed", "error": str(e)}

            if step_num in GPU_STEPS:
                print(f"  (GPU step — continuing pipeline)")
                continue
            else:
                raise
        finally:
            _clear_gpu()

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
    def step_02():
        if "02" not in skip:
            from pipeline import _02_baseline as m
            return m.run()

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
    def step_05():
        if "05" not in skip:
            from pipeline import _05_finetune as m
            return m.run()

    @step
    def step_05b():
        if "5b" not in skip:
            from pipeline import _05b_dpo as m
            return m.run()

    @step
    def step_06():
        if "06" not in skip:
            from pipeline import _06_test_model as m
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
        s02 = step_02()
        s03 = step_03()
        s04 = step_04()
        s05 = step_05()
        s05b = step_05b()
        s06 = step_06()
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
