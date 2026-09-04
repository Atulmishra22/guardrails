# Supervised Fine-Tuning and Completion-Only Training

> Read this before writing `src/training/sft.py`.
> Prerequisites: `docs/05_quantization.md`, `docs/01_what_is_lora.md`

---

## What SFT Does

Supervised Fine-Tuning (SFT) is the first training stage. The model is shown
input-output pairs and trained to reproduce the correct output given the input.

For this project:
- Input: raw user text (the guardrail prompt)
- Output: the structured JSON verdict with masked_input and mask_spans

The model is trained using **next-token prediction** — the standard language
modelling objective. At each position, predict the next token. The loss is
the cross-entropy between the predicted token distribution and the true token.

---

## The Causal LM Loss

Given a sequence of tokens `[t_0, t_1, t_2, ..., t_n]`, the loss is:

```
L = -1/n * sum( log P(t_i | t_0, ..., t_{i-1}) )
```

The model predicts each token from all preceding tokens. Loss is averaged
over all positions in the sequence.

---

## Completion-Only Training (Loss Masking)

This is the critical detail that separates SFT from naive language model training.

In a chat-formatted input the sequence looks like:

```
[SYSTEM prompt tokens] [USER tokens] [ASSISTANT tokens]
       ^                    ^                ^
     prompt              prompt           completion
```

If you compute loss over the entire sequence, the model wastes capacity
learning to reproduce the system prompt and user question — text it will
always have access to at inference time anyway. Worse, a long system prompt
drowns out the gradient signal from the short JSON completion.

**Completion-only training**: mask the loss on all prompt tokens, compute
loss only on the assistant (completion) tokens.

```
Tokens:   [SYSTEM]  [USER]   [ASST JSON output]
Loss:       0         0          computed here
```

The model only backpropagates through the JSON output. It learns to produce
the correct JSON given the input, not to reproduce the input itself.

In TRL this is handled by `DataCollatorForCompletionOnlyLM`:

```python
from trl import DataCollatorForCompletionOnlyLM

# The string that marks the start of the completion in the chat template
response_template = "<|im_start|>assistant\n"

collator = DataCollatorForCompletionOnlyLM(
    response_template=response_template,
    tokenizer=tokenizer,
)
```

The collator replaces all label token IDs before the response_template with
`-100`. PyTorch's CrossEntropyLoss ignores positions where the label is `-100`.

---

## Formatting Function

The SFTTrainer expects a dataset where each example is a single string
containing the full formatted conversation. We write a formatting function
that takes a batch of raw records and returns formatted strings:

```python
def formatting_func(batch: dict) -> list[str]:
    outputs = []
    for text, decision, category, risk_level, confidence, masked_input, mask_spans in zip(
        batch["text"], batch["decision"], batch["category"],
        batch["risk_level"], batch["confidence"],
        batch["masked_input"], batch["mask_spans"],
    ):
        completion = json.dumps({
            "decision":     decision,
            "category":     category,
            "risk_level":   risk_level,
            "confidence":   confidence,
            "masked_input": masked_input,
            "mask_spans":   mask_spans,
        }, ensure_ascii=False)

        messages = [
            {"role": "system",    "content": SYSTEM_PROMPT},
            {"role": "user",      "content": text},
            {"role": "assistant", "content": completion},
        ]
        outputs.append(
            tokenizer.apply_chat_template(messages, tokenize=False)
        )
    return outputs
```

This produces the full conversation string that gets tokenized and fed to
the collator, which then masks everything before the assistant turn.

---

## Training Arguments That Matter on a T4

```yaml
per_device_train_batch_size: 2      # Small batch — T4 has 16GB
gradient_accumulation_steps: 8      # Effective batch = 2 * 8 = 16
gradient_checkpointing:      true   # Recompute activations to save VRAM
bf16:                        true   # Mixed precision — faster, more stable than fp16
max_seq_length:              512    # Cap sequence length — longer = more VRAM
```

### Why gradient accumulation?

Running a batch of 2 per step, accumulating gradients over 8 steps, then
doing one optimizer update is mathematically equivalent to running a batch
of 16 — but uses a fraction of the VRAM since only 2 examples are in memory
at any time.

```
Batch of 16 (naive):   forward(16) -> backward(16) -> optimizer step
                        [requires 16 examples in VRAM simultaneously]

Gradient accumulation:
  Step 1: forward(2) -> backward(2) -> accumulate gradients
  Step 2: forward(2) -> backward(2) -> accumulate gradients
  ...
  Step 8: forward(2) -> backward(2) -> accumulate -> optimizer step
  [only 2 examples in VRAM at any time, same gradient signal as batch 16]
```

---

## What to Watch During Training

| Metric | What it means |
|---|---|
| `train/loss` | Should decrease steadily. If it flatlines early, lr is too low. |
| `eval/loss` | Should track train loss. Large gap = overfitting. |
| JSON validity rate | Measure manually on eval set after training. Target: >99%. |
| Grad norm | Should stay below `max_grad_norm` (1.0). Spikes indicate instability. |

### Common Failure Modes

| Symptom | Likely cause | Fix |
|---|---|---|
| Loss does not decrease | Learning rate too low, or data format wrong | Increase lr, check formatting_func |
| Loss collapses to near-zero immediately | Data leaking into labels — loss mask broken | Check DataCollatorForCompletionOnlyLM setup |
| OOM (out of memory) | Sequence too long or batch too large | Reduce max_seq_length or batch size |
| Model outputs freeform text instead of JSON | System prompt not strong enough or response_template mismatch | Verify chat template formatting |

---

## The Training Loop at a Glance

```
Load dataset (data/processed/train.jsonl)
    |
    v
formatting_func: raw records -> chat-formatted strings
    |
    v
DataCollatorForCompletionOnlyLM: tokenize + mask prompt tokens (label = -100)
    |
    v
SFTTrainer.train():
    for each batch:
        forward pass  -> logits
        loss = CrossEntropy(logits, labels)  [only on completion tokens]
        backward pass -> gradients
        [accumulate for N steps]
        optimizer.step()  [AdamW]
        lr_scheduler.step()
    |
    v
Save LoRA adapter -> models/guardrail-v1-sft/
```

---

## Summary

- SFT trains the model to produce the target JSON given the user input.
- Loss masking on the prompt is non-negotiable — without it the gradient signal is diluted.
- `DataCollatorForCompletionOnlyLM` handles masking via label=-100 at prompt positions.
- Gradient accumulation simulates large batch training within 16GB VRAM.
- First goal is >99% valid JSON output. Accuracy improvement comes after that.

Next: `src/training/sft.py` — implement what is described here.
