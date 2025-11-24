# API Reference

## Script Functions

### Script 1: download_and_test.py

#### `download_and_load_model() -> Tuple[AutoModelForCausalLM, AutoTokenizer]`
Downloads and loads the model with 4-bit quantization.

**Returns:**
- `model`: Loaded model
- `tokenizer`: Tokenizer

#### `test_inference(model, tokenizer, prompt: str) -> Tuple[str, float, float]`
Runs inference test on the model.

**Parameters:**
- `model`: The model
- `tokenizer`: The tokenizer
- `prompt`: Input prompt

**Returns:**
- `response`: Generated text
- `generation_time`: Time in seconds
- `tokens_per_second`: Generation speed

---

### Script 2: autoimmune_questions.py

#### `load_model() -> Tuple[Model, Tokenizer]`
Loads the model and tokenizer.

#### `get_model_response(model, tokenizer, question: str) -> Tuple[str, float]`
Gets response from model.

**Returns:**
- `response`: Model response
- `generation_time`: Time in seconds

#### `run_benchmark(model, tokenizer) -> List[Dict]`
Runs benchmark on all questions.

**Returns:** List of result dictionaries

---

### Script 3: download_papers.py

#### `search_pubmed(query: str, max_results: int = 20) -> List[Dict]`
Searches PubMed for papers.

**Parameters:**
- `query`: PubMed search query
- `max_results`: Maximum results

**Returns:** List of paper dictionaries

#### `search_arxiv(query: str, max_results: int = 20) -> List[Dict]`
Searches arXiv for papers.

**Parameters:**
- `query`: arXiv search query
- `max_results`: Maximum results

**Returns:** List of paper dictionaries

#### `download_papers() -> List[Dict]`
Downloads papers from all sources.

**Returns:** Combined list of papers

#### `process_papers_for_training(papers: List[Dict]) -> List[Dict]`
Processes papers into training format.

**Returns:** Formatted training data

---

### Script 4: qlora_finetune.py

#### `load_training_data() -> List[str]`
Loads training data from JSON.

**Returns:** List of training texts

#### `load_model_and_tokenizer() -> Tuple[Model, Tokenizer]`
Loads model with 4-bit quantization.

#### `create_lora_config() -> LoraConfig`
Creates LoRA configuration.

**Returns:** LoRA config object

#### `prepare_dataset(texts, tokenizer) -> Dataset`
Prepares dataset for training.

**Returns:** Tokenized dataset

#### `train_model(model, tokenizer, dataset) -> Trainer`
Trains the model.

**Returns:** Trainer object

---

### Script 5: test_qlora.py

#### `load_qlora_model() -> Tuple[Model, Tokenizer]`
Loads base model and QLoRA adapters.

**Returns:** Model with LoRA, tokenizer

#### `get_model_response(model, tokenizer, question: str) -> Tuple[str, float]`
Gets response from QLoRA model.

---

### Script 6: full_finetune.py

#### `check_vram() -> Tuple[bool, float]`
Checks VRAM availability.

**Returns:** (has_sufficient_vram, total_vram_gb)

#### `load_model_and_tokenizer() -> Tuple[Model, Tokenizer]`
Loads model without quantization for full fine-tuning.

---

### Script 7: test_full_model.py

#### `load_full_model() -> Tuple[Model, Tokenizer]`
Loads fully fine-tuned model.

---

### Script 8: score_results.py

#### `load_results(filename: str) -> List[Dict]`
Loads results from JSON file.

#### `calculate_metrics(results: List[Dict]) -> Dict`
Calculates metrics for results.

**Returns:** Dictionary of metrics

#### `create_comparison_table(...) -> pd.DataFrame`
Creates comparison DataFrame.

**Returns:** Pandas DataFrame

#### `generate_report(...) -> str`
Generates final report text.

**Returns:** Report string

---

## Data Structures

### Paper Dictionary

```python
{
    "source": "PubMed" | "arXiv",
    "pmid": str,              # PubMed ID (if applicable)
    "arxiv_id": str,          # arXiv ID (if applicable)
    "title": str,
    "abstract": str,
    "journal": str,
    "pub_date": str,
    "authors": List[str]      # First 5 authors
}
```

### Question Dictionary

```python
{
    "id": str,                # Q001, Q002, etc.
    "category": str,          # Disease category
    "difficulty": str,        # easy, medium, hard, very_hard
    "question": str           # Full question text
}
```

### Result Dictionary

```python
{
    "id": str,
    "category": str,
    "difficulty": str,
    "question": str,
    "response": str,
    "time_seconds": float,
    "timestamp": str          # ISO format
}
```

### Metrics Dictionary

```python
{
    "total_questions": int,
    "avg_time": float,
    "total_time": float,
    "avg_word_count": float,
    "avg_sentence_count": float,
    "has_reasoning_tags": int,
    "medical_terms_count": int,
    "structured_output": int,
    "confidence_mentioned": int,
    "next_steps_mentioned": int,
    "has_reasoning_tags_pct": float,
    "structured_output_pct": float,
    "confidence_mentioned_pct": float,
    "next_steps_mentioned_pct": float
}
```

---

## Environment Variables

All scripts use environment variables from `.env`:

- `MODEL_NAME`: Base model name
- `MAX_SEQ_LENGTH`: Maximum sequence length
- `PUBMED_EMAIL`: Email for PubMed API
- `DATA_DIR`: Data directory path
- `RESULTS_DIR`: Results directory path
- `MODELS_DIR`: Models directory path
- `QLORA_EPOCHS`: QLoRA training epochs
- `QLORA_BATCH_SIZE`: QLoRA batch size
- `QLORA_LR`: QLoRA learning rate
- `FULL_EPOCHS`: Full fine-tuning epochs
- `FULL_BATCH_SIZE`: Full fine-tuning batch size
- `FULL_LR`: Full fine-tuning learning rate

---

## Error Handling

All scripts follow this pattern:

```python
try:
    # Main logic
except Exception as e:
    print(f"Error: {str(e)}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
```

Common exit codes:
- `0`: Success
- `1`: Error occurred

