# guardrail-model

A **small, fast, privacy-first LLM input guardrail** built with teacher distillation, SFT, and DPO.

## What it does

Before any user input reaches a cloud LLM (ChatGPT, Gemini, Claude), this model:
1. **Detects** private/sensitive data spans (PII, credentials, financial info, medical data)
2. **Masks** them with `[MASKED_*]` tokens
3. **Classifies** the intent as `ALLOW`, `BLOCK`, or `SANITIZE`
4. Forwards the **sanitized text** to the cloud LLM — raw PII never leaves your system

```
User Input  →  Guardrail Model  →  masked_input + decision  →  Gate  →  Cloud LLM
```

## Output Schema

```json
{
  "decision":     "ALLOW | BLOCK | SANITIZE",
  "category":     "benign | violence | fraud | privacy | ...",
  "risk_level":   "low | medium | high",
  "confidence":   0.95,
  "masked_input": "Hi [MASKED_NAME], your card [MASKED_FINANCIAL] is approved.",
  "mask_spans": [
    { "token_start": 1, "token_end": 2, "type": "NAME",      "original": "John" },
    { "token_start": 5, "token_end": 8, "type": "FINANCIAL", "original": "4111-1111-1111-1111" }
  ]
}
```

## Student Model

- **Primary:** `Qwen2.5-1.5B-Instruct` (Apache 2.0)
- **Fine-tuning:** QLoRA 4-bit — fits on a free Kaggle T4 (16GB VRAM)

## Training Pipeline

```
Frontier Teacher (Gemini / Claude)
    → Synthetic dataset (500 seed → 10K scaled)
    → SFT (guardrail-v1)
    → DPO on hard examples (guardrail-v2)
    → Evaluate on sealed independent eval set
```

## Project Structure

```
guardrail-model/
├── configs/          # YAML configs for data, SFT, DPO
├── data/             # Datasets (gitignored except eval/)
├── src/              # All source code
│   ├── data_generation/
│   ├── training/
│   ├── evaluation/
│   └── inference/
├── notebooks/        # Kaggle training notebooks
├── scripts/          # CLI utilities
├── tests/            # Pytest unit tests
├── docs/             # Concept explanations (read before coding)
└── models/           # Checkpoints (gitignored)
```

## Setup

```bash
# Install uv (if not already installed)
pip install uv

# Create virtual environment and install all deps
uv sync

# Install with optional groups
uv sync --extra dev          # + pytest, ruff
uv sync --extra data         # + teacher API clients
```

## Key Metrics

| Metric | Why it matters |
|---|---|
| **Span Recall** | Missed PII = real privacy leak — #1 priority |
| Span F1 | Combined masking quality |
| FNR | Missed harmful content |
| FPR | Over-blocking safe content |
| P95 Latency | Must stay ~45ms for production viability |

## License

Apache 2.0 — same as `Qwen2.5-1.5B-Instruct`.
