# ImmunePlan — Reliable Medical AI System

10-step pipeline for building a **reliable** autoimmune disease AI assistant, implementing all concepts from *Building Reliable AI Systems* (Manning, 2026).

See [ARCHITECTURE.md](ARCHITECTURE.md) for the complete concept-to-code mapping.

## Pipeline

| Step | Script | Book Chapter | What It Does |
|------|--------|-------------|--------------|
| 01 | `pipeline/01_setup.py` | Ch 1 | Config, download model, detect GPU, select training tier |
| 02 | `pipeline/02_baseline.py` | Ch 2 | Prompt engineering baseline — test model before any changes |
| 03 | `pipeline/03_collect_data.py` | Ch 3-4 | Download papers from arXiv/PubMed, extract, chunk |
| 04 | `pipeline/04_build_rag.py` | Ch 3-4 | Build vector store, embeddings, test hybrid retrieval |
| 05 | `pipeline/05_finetune.py` | Ch 5 | **Adaptive** fine-tuning — auto-selects tier by GPU memory |
| 06 | `pipeline/06_test_model.py` | Ch 5 | Test fine-tuned model, compare against baseline |
| 07 | `pipeline/07_build_agent.py` | Ch 6-7 | Build agent with tools, memory, MCP interfaces |
| 08 | `pipeline/08_evaluate.py` | Ch 9 | FActScore, GDR, quality metrics, red teaming |
| 09 | `pipeline/09_safety.py` | Ch 11 | PII detection, bias testing, name experiment |
| 10 | `pipeline/10_deploy.py` | Ch 10 | Deploy API with monitoring, dashboards, alerts |

## Quick Start

```bash
# Run any step standalone
python3 pipeline/01_setup.py --skip-download
python3 pipeline/07_build_agent.py
python3 pipeline/09_safety.py

# Run entire pipeline
bash pipeline/run_all.sh

# Skip GPU-dependent steps
SKIP="02,05,06" bash pipeline/run_all.sh

# Start the API
python3 api.py
```

## Adaptive Fine-Tuning (Step 05)

The fine-tuning step automatically selects the optimal training strategy based on available GPU memory:

| VRAM | Tier | Method | Rank | Targets | Depth |
|------|------|--------|------|---------|-------|
| < 16 GB | minimal | QLoRA | 8 | Attention only | Lightweight |
| 16-23 GB | standard | QLoRA | 16 | Attention only | Standard |
| 24-39 GB | enhanced | DoRA | 32 | Attention + MLP | Deep |
| 40+ GB | maximum | High-Rank LoRA | 256 | All layers | Maximum |

## Project Structure

```
pipeline/
├── config.py           # Single source of truth (config, utils, questions)
├── 01_setup.py         # Ch 1: Setup
├── 02_baseline.py      # Ch 2: Baseline
├── 03_collect_data.py  # Ch 3-4: Data collection
├── 04_build_rag.py     # Ch 3-4: RAG pipeline
├── 05_finetune.py      # Ch 5: Adaptive fine-tuning
├── 06_test_model.py    # Ch 5: Test fine-tuned
├── 07_build_agent.py   # Ch 6-7: Agent + tools
├── 08_evaluate.py      # Ch 9: Evaluation
├── 09_safety.py        # Ch 11: Safety
├── 10_deploy.py        # Ch 10: Deploy
├── run_all.sh          # Run all steps
├── run_pipeline.py     # ZenML pipeline runner
└── lib/                # Shared library
    ├── prompts.py      # Structured prompt templates
    ├── rag.py          # Vector store, hybrid retrieval
    ├── tools.py        # Medical tools, MCP interfaces
    ├── agent.py        # Agent framework, multi-agent
    ├── evaluation.py   # Hallucination metrics, quality
    ├── safety.py       # PII, bias, content safety
    └── monitoring.py   # Tracing, dashboards, alerts

api.py                  # REST API with full reliability pipeline
```

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/chat` | POST | Chat with full reliability pipeline |
| `/api/doctors` | GET | List available specialist doctors |
| `/api/tools` | GET | List registered medical tools |
| `/api/tools/<name>` | POST | Execute a medical tool directly |
| `/api/dashboard` | GET | Monitoring dashboard |
| `/api/feedback` | POST | Submit user feedback |
| `/api/reliability` | GET | System reliability specification |

## Hardware Requirements

### Minimum (RAG + Agent only, no fine-tuning)
- CPU only
- 8 GB RAM

### Recommended (Full pipeline with fine-tuning)
- NVIDIA GPU with 24+ GB VRAM
- 32 GB RAM
- 50 GB storage
