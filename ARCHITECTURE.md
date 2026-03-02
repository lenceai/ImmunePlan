# ImmunePlan Architecture

## Three-Layer Reliability Framework

```
┌───────────────────────────────────────────────────────────┐
│              Layer 3: Reliable Operations                  │
│  08 Evaluate │ 09 Safety │ 10 Deploy│
│      11 CLI Chatbot — Interactive doctor          │
├───────────────────────────────────────────────────────────┤
│              Layer 2: Reliable Agents                      │
│         07 Build Agent — Tools + MCP              │
├───────────────────────────────────────────────────────────┤
│              Layer 1: Reliable Outputs                     │
│  02 Baseline │ 03-04 RAG │ 05-06 FT│
└───────────────────────────────────────────────────────────┘
```

## Pipeline Steps

| Step | File | Concept |
|------|------|---------|
| 01 | `01_setup.py` | Define reliability spec BEFORE choosing architecture |
| 02 | `02_baseline.py` | Prompt engineering as behavior specification |
| 03 | `03_collect_data.py` | Data preparation for RAG and fine-tuning |
| 04 | `04_build_rag.py` | Vector store, hybrid retrieval, quality scoring |
| 05 | `05_finetune.py` | Adaptive fine-tuning (auto GPU tier selection) |
| 06 | `06_test_model.py` | Compare fine-tuned vs baseline |
| 07 | `07_build_agent.py` | Agent with ReAct loop, MCP tools, memory |
| 08 | `08_evaluate.py` | FActScore, GDR, red teaming, trajectory analysis |
| 09 | `09_safety.py` | PII/HIPAA, bias, name experiment, disclaimer |
| 10 | `10_deploy.py` | API with monitoring, drift detection, failure patterns |
| 11 | `11_cli_chatbot.py` | Interactive command-line doctor with memory + paper references |

## Adaptive Fine-Tuning (Step 05)

Single script that auto-selects training strategy based on GPU memory:

```
GPU VRAM → select_training_tier() → Training Configuration
< 16 GB  → minimal   → QLoRA rank 8,   attention only,  seq 512
16-23 GB → standard  → QLoRA rank 16,  attention only,  seq 1024
24-39 GB → enhanced  → DoRA rank 32,   attention + MLP, seq 2048
40+ GB   → maximum   → LoRA rank 256,  all layers,      seq 512
```

More VRAM = higher rank = more trainable parameters = deeper model adaptation.

## Library Modules (pipeline/lib/)

| Module | Purpose |
|--------|---------|
| `prompts.py` | Structured templates, self-consistency, query rephrasing, parameter profiles |
| `rag.py` | Vector store, hybrid retrieval (dense + keyword + RRF), quality scoring |
| `tools.py` | Tool registry, 4 medical tools, MCP-style structured responses |
| `agent.py` | MedicalAgent (ReAct loop), MultiAgentOrchestrator, intent routing |
| `evaluation.py` | FActScore, GDR, quality checker, red teaming, trajectory analysis |
| `safety.py` | PIIDetector, ContentSafetyFilter, BiasDetector, SafetyPipeline |
| `monitoring.py` | RequestTrace, MonitoringService, drift detection, failure patterns, name experiment |

## End-to-End Request Flow

```
User Query
    │
    ▼
[1] Safety check input → PII? redact. Crisis? return resources.
    ▼
[2] Classify intent → diagnosis / treatment / lab / drug / guidelines / qa
    ▼
[3] Use tools (if needed) → lab lookup / drug info / guidelines
    ▼
[4] RAG retrieval → hybrid search → quality scoring → context assembly
    ▼
[5] Render prompt → select template by type → inject context + history
    ▼
[6] Generate response → LLM or structured fallback
    ▼
[7] Evaluate quality → groundedness + structure + safety score
    ▼
[8] Safety check output → PII redaction + disclaimer + bias check
    ▼
[9] Log trace → monitoring + metrics + alerts
    ▼
Response + metadata (confidence, citations, safety, steps)
```

## Six Principles of Reliable AI

1. **Ground outputs in verified information** — RAG + citations + "I don't know"
2. **Constrain agents to safe actions** — least privilege + confirmation steps
3. **Monitor systems continuously** — semantic quality, not just uptime
4. **Treat all users fairly** — name experiment + fairness metrics
5. **Protect user privacy** — PII detection + HIPAA/GDPR compliance
6. **Build in evaluation from day one** — metrics before code
