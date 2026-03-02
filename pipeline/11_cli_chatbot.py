#!/usr/bin/env python3
"""
Step 11: CLI Autoimmune Doctor Chatbot
Interactive agent with memory, RAG grounding, and citations.

Standalone:
    python pipeline/11_cli_chatbot.py
    python pipeline/11_cli_chatbot.py --model auto
    python pipeline/11_cli_chatbot.py --model fallback

Behavior:
  - Launches an interactive command-line chatbot ("Dr. Immunity")
  - Uses RAG citations so each answer includes paper references
  - Prefers DPO or fine-tuned adapters when available, otherwise base model
  - Falls back to deterministic agent responses if LLM loading fails
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from pipeline.config import (
    DATA_DIR,
    MODELS_DIR,
    MODEL_NAME,
    ensure_directories,
    generate_response,
    get_quantization_config,
    load_model_and_tokenizer,
    print_step,
    setup_logging,
)
from pipeline.lib.agent import MedicalAgent, MultiAgentOrchestrator
from pipeline.lib.monitoring import MonitoringService
from pipeline.lib.rag import RAGPipeline, VectorStore
from pipeline.lib.safety import SafetyPipeline
from pipeline.lib.tools import create_medical_tool_registry


def _parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Interactive autoimmune doctor chatbot with paper references."
    )
    parser.add_argument(
        "--model",
        choices=["auto", "dpo", "finetuned", "base", "fallback"],
        default="auto",
        help="Model backend preference. 'auto' tries dpo -> finetuned -> base.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.2,
        help="Sampling temperature for LLM generation.",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=768,
        help="Maximum tokens to generate per response.",
    )
    parser.add_argument(
        "--max-citations",
        type=int,
        default=4,
        help="Maximum number of paper references printed per answer.",
    )
    parser.add_argument(
        "--allow-no-rag",
        action="store_true",
        help="Allow running even when vector store is missing (not recommended).",
    )
    return parser.parse_args(argv)


def _load_local_or_adapter_model(model_dir: Path):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(str(model_dir), trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    adapter_config = model_dir / "adapter_config.json"
    if adapter_config.exists():
        from peft import PeftModel

        base_model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME,
            quantization_config=get_quantization_config(),
            dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True,
            use_safetensors=False,
        )
        model = PeftModel.from_pretrained(base_model, str(model_dir))
    else:
        model = AutoModelForCausalLM.from_pretrained(
            str(model_dir),
            quantization_config=get_quantization_config(),
            dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True,
            use_safetensors=False,
        )

    model.eval()
    return model, tokenizer


def _candidate_backends(model_pref: str):
    if model_pref == "auto":
        return [
            ("dpo", MODELS_DIR / "dpo_model"),
            ("finetuned", MODELS_DIR / "finetuned_model"),
            ("base", None),
        ]
    if model_pref == "dpo":
        return [("dpo", MODELS_DIR / "dpo_model")]
    if model_pref == "finetuned":
        return [("finetuned", MODELS_DIR / "finetuned_model")]
    if model_pref == "base":
        return [("base", None)]
    return [("fallback", None)]


def _attach_generator(
    agent: MedicalAgent,
    model_pref: str,
    temperature: float,
    max_new_tokens: int,
    logger,
) -> str:
    if model_pref == "fallback":
        return "fallback"

    for backend_name, backend_path in _candidate_backends(model_pref):
        try:
            if backend_name == "base":
                model, tokenizer = load_model_and_tokenizer(quantize=True)
            else:
                if backend_path is None or not backend_path.exists():
                    logger.info(f"Skipping {backend_name}: model path missing")
                    continue
                model, tokenizer = _load_local_or_adapter_model(backend_path)

            def _generate(prompt: str) -> str:
                do_sample = temperature > 0.0
                response, _, _ = generate_response(
                    model,
                    tokenizer,
                    prompt,
                    max_new_tokens=max_new_tokens,
                    temperature=max(0.0, temperature),
                    top_p=0.9,
                    do_sample=do_sample,
                    repetition_penalty=1.05,
                )
                return response.strip()

            agent.set_generator(_generate)
            # Use exact token counting in RAG if tokenizer is available
            agent.rag.tokenizer = tokenizer
            return backend_name
        except Exception as e:
            logger.warning(f"Failed loading {backend_name} backend: {e}")

    logger.warning("No LLM backend available, using deterministic fallback responses.")
    return "fallback"


def _build_agent_stack():
    monitoring = MonitoringService()
    tools = create_medical_tool_registry()
    store = VectorStore()
    rag = RAGPipeline(store)
    safety = SafetyPipeline()

    agent = MedicalAgent(
        tool_registry=tools,
        rag_pipeline=rag,
        safety_pipeline=safety,
        monitoring=monitoring,
    )
    orchestrator = MultiAgentOrchestrator(monitoring)
    orchestrator.register_agent("immune", agent)
    return agent, orchestrator, store


def _render_references(citations, max_citations: int) -> str:
    if citations:
        refs = citations[: max(1, max_citations)]
        lines = [f"[{i}] {c}" for i, c in enumerate(refs, start=1)]
        return "\n".join(lines)
    return (
        "[none] No relevant papers retrieved for this answer. "
        "Try asking with more disease-specific detail."
    )


def _chat_loop(orchestrator: MultiAgentOrchestrator, agent: MedicalAgent, max_citations: int):
    print("\nDr. Immunity is ready.")
    print("Commands: /help, /reset, /exit")
    print("Ask autoimmune questions in natural language.\n")

    while True:
        try:
            user_text = input("You > ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nEnding chat session.")
            break

        if not user_text:
            continue

        lowered = user_text.lower()
        if lowered in {"/exit", "exit", "quit", "/quit"}:
            print("Session ended.")
            break
        if lowered in {"/help", "help"}:
            print("Commands: /help, /reset, /exit")
            print("Tip: mention condition + labs + treatment context for stronger citations.")
            continue
        if lowered in {"/reset", "reset"}:
            agent.memory.clear()
            print("Conversation memory cleared.")
            continue

        result = orchestrator.route_and_process(user_text, doctor_type="immune")
        refs_text = _render_references(result.citations, max_citations=max_citations)

        print("\nDr. Immunity >")
        print(result.response)
        print("\nPaper references:")
        print(refs_text)
        print(
            f"\n(meta) confidence={result.confidence}, "
            f"intent={result.intent.value}, "
            f"latency={result.processing_time_seconds:.2f}s\n"
        )


def run(argv=None):
    args = _parse_args(argv)
    print_step(11, "CLI AUTOIMMUNE DOCTOR CHATBOT")
    logger = setup_logging("11_cli_chatbot")
    ensure_directories()

    agent, orchestrator, store = _build_agent_stack()

    vs_path = DATA_DIR / "vector_store"
    has_store = vs_path.exists() and store.load(str(vs_path))
    if not has_store:
        msg = (
            f"Vector store not found at {vs_path}. "
            "Run step 04 first so answers can reference papers."
        )
        if not args.allow_no_rag:
            print(msg)
            return {"status": "skipped", "reason": "missing_vector_store"}
        print(f"WARNING: {msg}")
    else:
        print(f"Loaded vector store with {len(store.chunks)} chunks.")

    backend = _attach_generator(
        agent,
        model_pref=args.model,
        temperature=args.temperature,
        max_new_tokens=args.max_new_tokens,
        logger=logger,
    )
    print(f"Generation backend: {backend}")

    if not sys.stdin.isatty():
        print("Non-interactive terminal detected; chatbot initialized but chat loop not started.")
        return {
            "status": "ready",
            "backend": backend,
            "vector_store_chunks": len(store.chunks),
        }

    _chat_loop(orchestrator, agent, max_citations=args.max_citations)
    return {
        "status": "completed",
        "backend": backend,
        "vector_store_chunks": len(store.chunks),
    }


def main():
    run()


if __name__ == "__main__":
    main()
