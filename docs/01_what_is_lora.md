# LoRA and QLoRA

> Read this before writing `src/training/sft.py`.
> Prerequisite: [`docs/05_quantization.md`](05_quantization.md)

---

## The Problem: Fine-Tuning Is Expensive

When you fine-tune a model the normal way (**full fine-tuning**), every single weight gets updated:

```
Qwen2.5-1.5B has ~1.5 billion parameters
Full fine-tuning = update all 1.5B weights
                 = store gradients for all 1.5B weights
                 = store optimizer states for all 1.5B weights (Adam = 2x)
                 ≈ 4.5B floats just for optimizer states = ~18 GB  [exceeds T4 limit]
```

On a 16GB Kaggle T4, this is impossible. We need a smarter approach.

---

## The Insight Behind LoRA

In 2021, Microsoft researchers discovered something surprising:

> When you fine-tune a large model on a specific task, the **change in weights** has a very low "intrinsic rank".

What does that mean? Let's break it down.

A weight matrix `W` in a transformer might be shape `[4096 × 4096]` — that's 16 million numbers.
But the **meaningful update** `ΔW` when adapting to a new task can be expressed as the product of two **tiny** matrices:

```
Full weight matrix W:          [4096 × 4096]  = 16,777,216 numbers

LoRA approximation of ΔW:
    Matrix A:  [4096 × 16]   =     65,536 numbers  (rank = 16)
    Matrix B:  [16 × 4096]   =     65,536 numbers
    ΔW = A × B               = 131,072 numbers total

Compression ratio:  16,777,216 / 131,072  =  128x fewer parameters
```

LoRA = **Low-Rank Adaptation** — represent the weight update as a product of two small matrices.

---

## How LoRA Works

```
Original model (frozen):                New LoRA adapter (trainable):

Input x                                 Input x
    │                                       │
    ▼                                       ▼
  W (frozen)          +            A (small, trainable)
    │                                       │
    ▼                                       ▼
 Output₁                           B (small, trainable)
                                           │
                                           ▼
                                        Output₂

Final output = Output₁ + (alpha/r) × Output₂
```

- `W` (the original weight matrix) is **completely frozen** — no gradients, no updates.
- Only `A` and `B` (the tiny adapter matrices) are trained.
- At inference time, you can **merge** `ΔW = A × B` back into `W` — zero extra latency.

---

## The Hyperparameters Explained

### `r` — Rank

```yaml
lora:
  r: 16
```

`r` is the "width" of the bottleneck — the size of the small dimension.

```
r = 4:   A [4096×4],  B [4×4096]  →  32K params per layer  (very small, less capacity)
r = 8:   A [4096×8],  B [8×4096]  →  65K params per layer
r = 16:  A [4096×16], B [16×4096] →  131K params per layer ← we use this
r = 64:  A [4096×64], B [64×4096] →  524K params per layer (more capacity, more VRAM)
```

Higher `r` = more parameters = more capacity to learn, but also more VRAM and more risk of overfitting.
**Start with r=16. Drop to r=8 if VRAM is tight.**

### `lora_alpha` — Scaling Factor

```yaml
lora:
  lora_alpha: 32
```

The final update applied is: `ΔW = (alpha / r) × A × B`

- `alpha = 32`, `r = 16` → scaling factor = `32/16 = 2.0`
- Rule of thumb: **alpha = 2 × r** (scaling factor of 2.0 is stable)
- Higher alpha → LoRA updates have stronger influence on the model

### `target_modules` — Which Layers Get LoRA

```yaml
target_modules:
  - "q_proj"    # Query projection
  - "k_proj"    # Key projection
  - "v_proj"    # Value projection
  - "o_proj"    # Output projection
  - "gate_proj" # FFN gate
  - "up_proj"   # FFN up-projection
  - "down_proj" # FFN down-projection
```

These are the weight matrices inside each transformer layer.
We apply LoRA to all of them — this is the "full" LoRA setup.
Applying LoRA only to `q_proj` and `v_proj` is the minimal setup (saves more VRAM).

### `lora_dropout`

```yaml
lora_dropout: 0.05
```

Standard dropout applied to the LoRA layers during training — reduces overfitting.
Keep it small (0.05–0.1). Set to 0 if the dataset is large (>10K examples).

---

## QLoRA = Quantization + LoRA

QLoRA combines the two techniques we've learned:

```
Base model weights:  stored in 4-bit NF4 (quantized, frozen)  ← from 05_quantization.md
LoRA adapters:       stored in bfloat16 (trainable, small)     ← from this doc
```

```
VRAM breakdown for Qwen2.5-1.5B with QLoRA (r=16):

  Base model (4-bit):          ~0.9  GB
  LoRA adapter weights:        ~0.05 GB  (tiny — only 131K params × 7 layers × bfloat16)
  Optimizer states (Adam):     ~0.1  GB  (only for LoRA params, not base model!)
  Activations + gradients:     ~4.0  GB  (with gradient checkpointing)
  Tokenizer + misc:            ~0.5  GB
  ─────────────────────────────────────
  Total:                       ~5.5  GB  — fits on 16GB T4
```

This is the magic — you get full fine-tuning quality at a fraction of the cost.

---

## Gradient Checkpointing

```yaml
training:
  gradient_checkpointing: true
```

During training, PyTorch stores all intermediate activations (the output of each layer)
to use during the backward pass (gradient computation).

On a large model this takes a lot of VRAM. Gradient checkpointing solves this by:
- **Not storing** intermediate activations
- **Recomputing** them during the backward pass when needed

```
Without checkpointing:   store everything → fast backward, huge VRAM usage
With checkpointing:      store nothing    → slow backward (~20% slower), tiny VRAM usage
```

On a 16GB T4, gradient checkpointing is **non-negotiable** — always enable it.

---

## After Training: Merging the Adapter

Once training is done, you can merge the LoRA adapter back into the base model:

```python
from peft import PeftModel

# Load base model + adapter
model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-1.5B-Instruct")
model = PeftModel.from_pretrained(model, "models/guardrail-v1-sft")

# Merge adapter into base model weights
model = model.merge_and_unload()

# Save the merged model (no adapter files needed at inference)
model.save_pretrained("models/guardrail-v1-merged")
```

Merged model = **same speed as the original model** at inference — LoRA has zero latency cost.

---

## Summary

```
LoRA:
  Don't update all 1.5B weights
  Instead: inject tiny A×B matrices into each layer
  Only train A and B → 128x fewer trainable parameters

QLoRA (LoRA + Quantization):
  Base model = frozen + 4-bit NF4 (saves ~5 GB VRAM)
  Adapters   = trainable + bfloat16 (tiny, ~50 MB)
  Result     = full fine-tuning quality on a free T4 GPU

Key hyperparameters:
  r=16         → rank of the bottleneck
  alpha=32     → scaling = alpha/r = 2.0
  target all 7 linear projection layers
  gradient_checkpointing = true (always on T4)
```

**Next concept to read:** [`docs/04_tokenization.md`](04_tokenization.md) — tokenization and token-level span offsets.
