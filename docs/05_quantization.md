# 4-Bit Quantization and BitsAndBytes

> Read this before touching `sft.yaml`'s quantization section or any model loading code.

---

## The Problem: Models Are Too Big for Free GPUs

A language model stores its knowledge in **weights** — millions or billions of floating point numbers.

By default, each weight is stored as a **32-bit float** (`float32`):

```
1.5B parameters × 4 bytes (float32) = 6 GB  just for weights
+ optimizer states (Adam) × 2       = 12 GB more during training
+ activations + gradients           = another 6–10 GB
────────────────────────────────────────────────────────
Total needed for full fine-tuning   ≈ 24–28 GB  [does not fit on T4 (16GB)]
```

We need to **shrink the weights** without destroying the model's knowledge.
That's what quantization does.

---

## What Is Quantization?

Quantization = represent weights using **fewer bits**.

```
float32  →  32 bits per weight  (full precision)
float16  →  16 bits per weight  (half precision)
int8     →   8 bits per weight  (8-bit quant)
nf4      →   4 bits per weight  (4-bit quant)  ← what we use
```

Think of it like rounding. Instead of storing `3.141592653589793`, you store `3.14`.
You lose a tiny bit of precision, but the model still works almost as well.

---

## NF4: Normal Float 4

The 4-bit format we use is called **NF4 (Normal Float 4)**.

Regular 4-bit integers (`int4`) would give you 16 evenly-spaced values: 0, 1, 2 ... 15.
But LLM weights are NOT evenly distributed — they follow a **bell curve (normal distribution)**.

NF4 is smarter:
- It places more values near 0 (where most weights cluster)
- Fewer values at the extremes
- Result: better precision where it matters most

```
int4 values:   |----|----|----|----|----| (evenly spaced)
nf4 values:    |--|-|-|--|-----|--------|  (denser near 0)
               ↑                       ↑
             -1.0                    +1.0
```

This is why `bnb_4bit_quant_type: "nf4"` is always preferred over `"int4"` for LLMs.

---

## Double Quantization

`bnb_4bit_use_double_quant: true` — what does this do?

When you quantize weights into 4-bit, you need small numbers called **quantization constants**
to remember how to convert back. These constants themselves take up memory.

Double quantization = **quantize the quantization constants too**.

```
Normal quant:   weights (4-bit) + constants (32-bit) ≈ 4.5 bits per weight effective
Double quant:   weights (4-bit) + constants (8-bit)  ≈ 4.1 bits per weight effective
                                                          ↑
                                              Saves ~0.4 GB on a 1.5B model
```

Small saving, but worth enabling — it's free performance on a memory-constrained GPU.

---

## Compute Dtype vs Storage Dtype

This is the most confusing part. Pay attention:

```yaml
bnb_4bit_compute_dtype: "bfloat16"
```

- **Storage dtype**: weights are stored in 4-bit (NF4) on disk and in VRAM.
- **Compute dtype**: when doing the actual matrix multiplication, weights are
  temporarily **dequantized** to `bfloat16` for the math, then discarded.

```
VRAM (stored):    [4-bit NF4 weights]
                         ↓
                  dequantize on-the-fly
                         ↓
GPU compute:      [bfloat16 weights]  ← math happens here
                         ↓
                  result stored in bfloat16
                  original weights stay in 4-bit
```

Why `bfloat16` and not `float16`?
- `bfloat16` has a wider exponent range than `float16` → less overflow during training
- Ampere GPUs (A100) and newer prefer `bfloat16`
- T4 supports `bfloat16` via software emulation — slightly slower but stable

---

## Memory Savings: Before vs After

For `Qwen2.5-1.5B-Instruct`:

| Setup | VRAM for weights | Can train on T4? |
|---|---|---|
| Full float32 | ~6.0 GB | No — optimizer states push total to ~24 GB |
| float16 | ~3.0 GB | Marginal — no headroom for gradients |
| 4-bit NF4 | ~0.9 GB | Yes — sufficient room for LoRA and activations |

This is why 4-bit quantization is the **key** that unlocks free Kaggle GPU training.

---

## The BitsAndBytes Config in Code

This is what the `quantization` block in `sft.yaml` translates to in Python:

```python
from transformers import BitsAndBytesConfig
import torch

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",           # NF4 format (better than int4)
    bnb_4bit_use_double_quant=True,      # Quantize the constants too
    bnb_4bit_compute_dtype=torch.bfloat16  # Compute in bf16, store in 4-bit
)

model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen2.5-1.5B-Instruct",
    quantization_config=bnb_config,
    device_map="auto"                    # Auto-place layers on GPU
)
```

You'll write this in `src/training/sft.py` (commit G2). For now just understand **why** each line exists.

---

## Common Mistakes

| Mistake | Why it's wrong |
|---|---|
| Setting `load_in_8bit=True` AND `load_in_4bit=True` | You can only pick one |
| Using `float16` compute dtype on T4 | More unstable than bfloat16 on that GPU |
| Skipping double quant | Free memory saving — always enable it |
| Using `int4` instead of `nf4` | int4 is less accurate for LLM weight distributions |

---

## Summary

```
4-bit NF4 quantization
    ↓
Weights stored in 4-bit (bell-curve-aware spacing)
    ↓
Double quant: constants also quantized → ~0.4 GB extra saved
    ↓
Compute dtype = bfloat16: math done in bf16, result accurate
    ↓
1.5B model fits in ~1 GB VRAM instead of 6 GB
    ↓
Room for LoRA adapters + optimizer states + activations on 16GB T4
```

**Next concept to read:** [`docs/01_what_is_lora.md`](01_what_is_lora.md) — LoRA and QLoRA explained.
