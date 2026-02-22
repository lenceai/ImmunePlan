# ImmunePlan Reliability Architecture

## Overview

This document maps every component of the ImmunePlan project to the concepts and principles from **"Building Reliable AI Systems"**. The system implements all three reliability layers across 11 chapters.

## Three-Layer Reliability Framework

```
┌─────────────────────────────────────────────────────────────┐
│                    Layer 3: Reliable Operations              │
│  Evaluation (Ch.9) │ Monitoring (Ch.10) │ Safety (Ch.11)     │
├─────────────────────────────────────────────────────────────┤
│                    Layer 2: Reliable Agents                   │
│  Agents (Ch.6) │ Tools/MCP (Ch.7) │ Multi-Agent (Ch.8)       │
├─────────────────────────────────────────────────────────────┤
│                    Layer 1: Reliable Outputs                  │
│  Prompts (Ch.2) │ RAG (Ch.3) │ Embeddings (Ch.4) │ FT (Ch.5)│
└─────────────────────────────────────────────────────────────┘
```

---

## Chapter-by-Chapter Implementation Map

### Chapter 1: AI Reliability — Defining Requirements First

**File**: `reliability/config.py`

Before choosing architecture, we define:
- **ReliabilitySpec**: What "reliable" means for ImmunePlan (clinical support tier)
- **Unacceptable failures**: Hallucinated diagnosis, incorrect dosage, PII leakage
- **Grounding requirements**: All claims must cite medical literature
- **Allowed/forbidden actions**: Can retrieve literature; cannot prescribe
- **Quality targets**: <5% hallucination rate, >90% citation rate, >99% safety pass rate
- **Latency/cost targets**: <5s simple queries, <$0.05/query

### Chapter 2: Prompt Engineering as Behavior Specification

**File**: `reliability/prompts.py`

Implements structured prompt templates with 7 reliability components:
1. **Role/Identity**: Dr. Immunity persona with clear scope limitations
2. **Task Objective**: Evidence-based answers from retrieved context only
3. **Constraints**: No fabrication, no prescriptions, no definitive diagnoses
4. **Grounding Instructions**: Cite sources, use only provided context
5. **Output Schema**: Summary → Details → Evidence → Confidence → Next Steps → Disclaimer
6. **Fallback Behavior**: Explicit uncertainty handling and refusal patterns
7. **Few-shot Examples**: Curated examples of ideal responses

Prompt templates:
- `MEDICAL_QA_PROMPT` — General medical questions
- `DIAGNOSIS_PROMPT` — Differential diagnosis support
- `TREATMENT_QUERY_PROMPT` — Treatment information
- `SAFETY_CHECK_PROMPT` — Output safety review

Query type classification routes to the appropriate template automatically.

### Chapter 3: RAG for Factual Grounding

**File**: `reliability/rag.py`

RAG pipeline with reliability-critical features:
- **Document chunking** with paragraph-aware splitting and overlap
- **Metadata preservation** (source, paper title, section type)
- **Quality scoring** of retrieval results
- **Insufficient context detection** — triggers uncertainty behavior
- **Source citation generation** for transparency

### Chapter 4: Embeddings and Vector Search

**File**: `reliability/rag.py` (VectorStore class)

Production retrieval features:
- **Dense retrieval** via sentence-transformers embeddings
- **Keyword fallback** (BM25-style) when embeddings unavailable
- **Hybrid retrieval** combining dense + keyword with Reciprocal Rank Fusion
- **Metadata filtering** by source, section, topic
- **Vector store persistence** (save/load to disk)
- **Source diversity scoring** in quality assessment

### Chapter 5: Fine-Tuning for Behavior Consistency

**Files**: `scripts/4_finetune.py`, `scripts/4_qlora_finetune.py`, `scripts/6_full_finetune.py`

Fine-tuning pipeline with reliability practices:
- **Decision framework**: Prompting vs RAG vs Fine-tuning vs Hybrid
- **Training data quality**: Includes normal, edge, uncertain, and refusal cases
- **Multiple approaches**: QLoRA, DoRA, High-rank LoRA
- **Evaluation**: Before/after benchmark comparison (scripts 5, 7, 8)

### Chapter 6: Creating Effective AI Agents

**File**: `reliability/agent.py` (MedicalAgent class)

Agent with reliability controls:
- **ReAct-style loop**: Classify → Check Safety → Use Tools → Retrieve → Generate → Evaluate → Sanitize
- **Memory management**: Session context, extracted facts, tool result history
- **Intent classification**: Routes to appropriate processing path
- **Step-level observability**: Every step is logged as an `AgentStep`
- **Fallback responses**: Structured output even without LLM

### Chapter 7: Tool Integration and MCP

**File**: `reliability/tools.py`

MCP-style standardized tool interfaces:
- **ToolSchema**: Self-describing tools with name, description, parameters, examples
- **ToolResponse**: Consistent response format with success/error_code/data/retryable/timestamp
- **ToolRegistry**: Discovery mechanism — model can list all available tools
- **Structured errors**: Distinguishes "not found" from "timeout" from "execution error"

Medical tools implemented:
- `lookup_lab_reference` — Reference ranges for CRP, ESR, RF, anti-CCP, ANA, etc.
- `assess_disease_activity` — DAS28, CDAI scoring with interpretation
- `check_drug_info` — Drug interactions, monitoring, contraindications
- `search_clinical_guidelines` — ACR, EULAR, AGA guideline lookup

### Chapter 8: Multi-Agent Systems

**File**: `reliability/agent.py` (MultiAgentOrchestrator class)

Multi-agent coordination:
- **IntentRouter**: Classifies queries into 8 intent categories
- **Agent registry**: Multiple specialist agents registered by name
- **Shared state**: Cross-agent context preservation
- **Routing decisions**: Logged for observability
- **Fallback routing**: Default agent when specialist unavailable

### Chapter 9: Evaluation and Performance

**File**: `reliability/evaluation.py`

Comprehensive evaluation stack:
- **GroundednessChecker**: Verifies claims against retrieved context using pattern matching
- **ResponseQualityChecker**: Structure, citations, disclaimers, medical terminology, safety
- **PerformanceTracker**: Latency, throughput, cost per request
- **SemanticCache**: Reduces redundant LLM calls via embedding similarity
- **Scoring rubric**: Weighted composite of structure (30%), accuracy (30%), safety (20%), completeness (20%)

### Chapter 10: Deploying and Monitoring (LLMOps)

**File**: `reliability/monitoring.py`

LLM-native monitoring:
- **RequestTrace**: Full trace of every request through the system
- **MonitoringService**: Central dashboard with real-time metrics
- **Alert system**: Threshold-based alerts for latency, error rate, cost
- **Quality tracking**: Groundedness, user satisfaction, safety issues
- **Improvement recommendations**: Automated analysis of metrics with actionable suggestions
- **Three pillars**: Prevention (prompt engineering) → Detection (metrics) → Correction (feedback loops)

Dashboard metrics:
- System health status (healthy/degraded)
- Performance (avg/p95 latency, cache hit rate)
- Quality (avg quality score, groundedness, safety issues, user rating)
- Cost (total, per-request, hourly)
- Query distribution by type
- Active alerts

### Chapter 11: Bias, Privacy, and Responsible AI

**File**: `reliability/safety.py`

Defense-in-depth architecture with 5 layers:

1. **Privacy Layer (PIIDetector)**:
   - 10 PII pattern categories (SSN, phone, email, MRN, DOB, address, insurance, credit card, IP, dates)
   - Patient name indicator detection
   - Full text redaction capability
   - HIPAA identifier coverage

2. **Safety Layer (ContentSafetyFilter)**:
   - Input safety: Crisis/self-harm detection → immediate crisis resources
   - Output safety: Definitive diagnosis, specific dosage, false guarantees, discouraging care
   - Medical disclaimer enforcement on all substantive responses

3. **Fairness Layer (BiasDetector)**:
   - Gender absolutism detection
   - Racial generalization detection
   - Age discrimination detection

4. **SafetyPipeline**: Orchestrates all layers
   - `check_input()` → PII detection + content safety
   - `check_output()` → Content safety + PII + bias
   - `sanitize_input()` → Redact PII before processing
   - `sanitize_output()` → Redact PII + ensure disclaimer
   - `get_crisis_response()` → Emergency resource information

---

## End-to-End Request Flow

```
User Query
    │
    ▼
[1] Input Safety Check ──── PII detected? → Redact
    │                   ──── Crisis? → Return crisis resources
    ▼
[2] Intent Classification ── medical_qa / diagnosis / treatment /
    │                        lab / drug / guidelines / crisis
    ▼
[3] Tool Usage (if needed) ── Lab lookup / Drug info / Guidelines
    │                         Structured ToolResponse format
    ▼
[4] RAG Retrieval ─────────── Dense + Keyword hybrid search
    │                         Quality scoring + context assembly
    ▼
[5] Prompt Rendering ──────── Select template by query type
    │                         Inject context + history + tools
    ▼
[6] LLM Generation ────────── Local model or fallback response
    │
    ▼
[7] Quality Evaluation ────── Groundedness check
    │                         Structure/citation/disclaimer check
    ▼
[8] Output Safety ─────────── Content filter + PII redaction
    │                         Disclaimer enforcement
    ▼
[9] Monitoring ────────────── Trace logging + metrics + alerts
    │
    ▼
Response with metadata (confidence, citations, safety check, steps)
```

---

## File Structure

```
reliability/
├── __init__.py          # Framework version and description
├── config.py            # Ch. 1: Reliability spec, model/RAG/safety/monitoring config
├── prompts.py           # Ch. 2: Structured prompt templates
├── rag.py               # Ch. 3-4: RAG pipeline, vector store, hybrid retrieval
├── evaluation.py        # Ch. 9: Quality, groundedness, performance, caching
├── tools.py             # Ch. 6-7: Tool registry, medical tools, MCP interfaces
├── safety.py            # Ch. 11: PII, content safety, bias, responsible AI
├── monitoring.py        # Ch. 10: Tracing, dashboards, alerts, LLMOps
├── agent.py             # Ch. 6, 8: Agent framework, multi-agent orchestration
└── pipeline.py          # Integration: End-to-end reliability pipeline

book_examples/
├── chapter2/            # Temperature, top-p, function calling examples
├── chapter3/            # RAG project with evaluation
├── chapter5/            # Fine-tuning decision framework
├── chapter6/            # Agent concepts, ReAct, guardrails
├── chapter8/            # Multi-agent orchestration patterns
├── chapter9/            # Evaluation suite, LLM-as-judge, fallback
└── chapter11/           # Responsible AI, PII, bias, defense-in-depth

scripts/
├── 1_download_and_test.py      # Download and test base model
├── 2_autoimmune_questions.py   # Benchmark question testing
├── 3_download_papers.py        # Research paper collection + chunking
├── 4_finetune.py               # DoRA fine-tuning
├── 4_qlora_finetune.py         # QLoRA fine-tuning
├── 5_test_qlora.py             # Test fine-tuned model
├── 6_full_finetune.py          # High-rank LoRA fine-tuning
├── 7_test_full_model.py        # Test full fine-tuned model
├── 8_score_results.py          # Compare and score results
├── build_vector_store.py       # Build vector store from training data
├── test_reliability.py         # Test entire reliability framework
└── common.py                   # Shared utilities

api.py                   # REST API v2.0 with all reliability endpoints
```

---

## API Endpoints

| Endpoint | Method | Description | Reliability Layer |
|----------|--------|-------------|-------------------|
| `/` | GET | API info and capabilities | — |
| `/health` | GET | System health with reliability status | Layer 3 |
| `/api/chat` | POST | Full reliability pipeline chat | All layers |
| `/api/doctors` | GET | List available specialist doctors | Layer 2 |
| `/api/doctor/<type>` | GET | Doctor specialty info | Layer 2 |
| `/api/tools` | GET | List registered medical tools | Layer 2 |
| `/api/tools/<name>` | POST | Execute a medical tool directly | Layer 2 |
| `/api/dashboard` | GET | Monitoring dashboard + recommendations | Layer 3 |
| `/api/feedback` | POST | Submit user feedback | Layer 3 |
| `/api/reliability` | GET | System reliability specification | Layer 1 |

---

## Advanced Capabilities (from detailed reference review)

### Self-Consistency Prompting
- Generate multiple outputs at moderate temperature
- Aggregate via claim-overlap consistency scoring
- Select the most consistent response as the final answer
- Configurable parameter profiles: `factual`, `clinical_support`, `creative`, `self_consistency`

### Formal Hallucination Metrics
- **FActScore**: Break responses into atomic facts, verify each against context
- **GDR (Grounding Defect Rate)**: Proportion of responses with ungrounded claims
- **Hallucination Severity Score**: Distribution across none/low/medium/high

### Red Teaming Framework
8 adversarial test categories: fabricated drugs, fabricated studies, prescription requests, dangerous advice, PII elicitation, prompt injection, complex multi-hop, ambiguous queries

### Agent Trajectory Analysis
- Tool selection accuracy (did the agent pick the right tool?)
- Reasoning quality (was the thought process sound?)
- Execution efficiency (how many steps were needed?)
- Safety check verification

### Production Failure Pattern Detection
Automated detection of 5 failure patterns:
1. **Invisible drift** — week-over-week quality degradation
2. **Cost explosion** — hourly cost spikes
3. **Confident hallucination** — low groundedness with high confidence
4. **Missing optimization** — underutilized semantic caching
5. **Evaluation afterthought** — missing monitoring data

### Name Experiment Bias Detection
- Demographic test pairs across race, ethnicity, age/gender
- Automated response comparison for length, detail, and quality parity
- Bias direction detection

### Tool Output Validation
- Required key verification before agent uses tool output
- Value range checking (e.g., temperature between -50 and 60)
- Response size limits

---

## The Six Principles + 10 Rules

### Six Principles of Reliable AI (from the book)

1. **Ground outputs in verified information** — RAG + citations + "I don't know"
2. **Constrain agents to safe actions** — least privilege + confirmation steps
3. **Monitor systems continuously** — semantic quality, not just uptime
4. **Treat all users fairly** — name experiment + fairness metrics
5. **Protect user privacy** — PII detection + HIPAA/GDPR compliance
6. **Build in evaluation from day one** — metrics before code

### 10 Implementation Rules

1. **Reliability is a systems property, not a model property**
2. **Grounding beats confidence** — RAG/tools > "smart-sounding" answers
3. **Retrieval quality determines RAG quality**
4. **Fine-tuning is for behavior consistency, not live facts**
5. **Agents multiply capability and failure modes together**
6. **Tool interfaces are part of the prompt**
7. **Evaluate process, not just outputs** (especially for agents)
8. **Monitor quality/cost/user outcomes, not just uptime**
9. **Responsible AI controls belong in production architecture**
10. **Start simple, instrument early, add complexity only when metrics justify it**

---

## Common Failure Patterns to Avoid

| Pattern | Description | Our Mitigation |
|---------|-------------|----------------|
| "Works in the lab" trap | Great on test sets, fails on real queries | Red teaming + adversarial test suite |
| "Invisible drift" | Gradual quality degradation unnoticed for weeks | Week-over-week comparison + drift alerts |
| "Cost explosion" | Prompt change doubles monthly bill | Hourly cost tracking + per-token alerts |
| "Confident hallucination" | Wrong answers with authoritative tone | FActScore + groundedness checking + GDR |
| "Monolithic agent" | One agent trying to do everything | Multi-agent orchestrator + specialist routing |
| "Evaluation afterthought" | Building system then adding tests last | Evaluation framework built before production code |
