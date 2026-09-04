# Tokenization and Token-Level Span Offsets

> Read this before writing `src/data_generation/validators.py`.
> This explains why we use **token-level** offsets for PII spans (not character-level).

---

## What Is Tokenization?

A language model doesn't read text character-by-character or word-by-word.
It reads **tokens** — chunks of text determined by a learned vocabulary.

```python
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-1.5B-Instruct")

text = "My name is John and my SSN is 123-45-6789"
tokens = tokenizer.tokenize(text)

# Output (approximate):
# ['My', ' name', ' is', ' John', ' and', ' my', ' SSN', ' is', '▁123', '-', '45', '-', '67', '89']
#   0       1       2      3        4       5      6       7      8       9   10   11   12   13
```

Notice:
- `"John"` → 1 token (common name, in vocabulary)
- `"123-45-6789"` → 5 tokens (numbers get split into pieces)
- `"SSN"` → 1 token

---

## Why Tokens Are Not Words

The tokenizer uses **Byte-Pair Encoding (BPE)** — a compression algorithm that:
1. Starts with individual characters
2. Repeatedly merges the most common adjacent pair into a new token
3. Repeats until vocabulary size (e.g., 150,000 for Qwen) is reached

This means:
- Common words = **1 token** (`"the"`, `"name"`, `"John"`)
- Rare words = **multiple tokens** (`"cryptocurrency"` → `["crypto", "currency"]`)
- Numbers = **always split** (digits are treated individually, then merged where frequent)
- Special characters often break tokens

```
"Sarah"          → ['Sarah']            1 token
"s4r4h"          → ['s', '4', 'r', '4', 'h']   5 tokens (l33t speak defeats tokenizer)
"john@gmail.com" → ['john', '@', 'gmail', '.', 'com']   5 tokens
"4111111111111111" → ['4111', '111', '111', '111', '11']   5 tokens
```

This has huge implications for **adversarial inputs** — attackers can deliberately misspell
or encode PII to confuse a character-level detector. Our model must learn at the token level.

---

## Token IDs vs Token Strings

The model never sees text — it sees **integer token IDs**:

```python
text = "Hi, I'm John."
token_ids = tokenizer.encode(text)
# → [13347, 11, 358, 2846, 3842, 13]

# To go back:
tokenizer.decode(token_ids)
# → "Hi, I'm John."
```

Training, inference, and span offsets all operate on these integer IDs.

---

## What Are Token-Level Span Offsets?

A **span** is a contiguous range of tokens that represents a piece of information.

In our output schema:

```json
"mask_spans": [
  {
    "token_start": 3,
    "token_end":   4,
    "type":        "NAME",
    "original":    "John"
  }
]
```

`token_start = 3`, `token_end = 4` means:
- Look at the token sequence: `[0:'My', 1:' name', 2:' is', 3:' John', 4:' and', ...]`
- Tokens at positions **3 to 3** (inclusive, or 3 to 4 exclusive — we'll use exclusive)
- That range corresponds to `"John"` → replace with `[MASKED_NAME]`

---

## Why Token-Level, Not Character-Level?

You might think character-level offsets (like `{"char_start": 11, "char_end": 15}`) are simpler.
They are simpler for **humans**, but they cause problems for the model:

### Problem 1: The Model Thinks in Tokens, Not Characters

The model's attention, its hidden states, everything is aligned to **tokens**.
If you ask a model to output character offsets, it has to internally convert from its native token space — an extra learned step that adds error.

Token offsets are **native** to the model.

### Problem 2: Tokenization Changes After Masking

When you replace `"John"` with `[MASKED_NAME]`, the character positions of everything after it **shift**:

```
Before: "My name is John and my SSN is 123-45-6789"
         0         10        20        30        40
                   ↑ John at char 11–14

After:  "My name is [MASKED_NAME] and my SSN is 123-45-6789"
         0         10             25   ← everything shifted by 11 chars
```

Token-level offsets **don't have this problem** — the `[MASKED_NAME]` token is treated as 1 token regardless of how many characters it represents.

---

## Special Tokens

Qwen uses a **chat template** that wraps input/output in special tokens:

```
<|im_start|>system
You are a guardrail model...
<|im_end|>
<|im_start|>user
My name is John. What's the capital of France?
<|im_end|>
<|im_start|>assistant
{"decision": "ALLOW", ...}
<|im_end|>
```

When computing span offsets, you need to be careful:
- The prompt tokens (system + user turn) appear **before** the assistant tokens
- Your span offsets must be relative to the **full tokenized sequence**, including the chat template
- OR: relative to just the **user input text** (we'll standardize on this — offsets relative to user text)

We'll define this precisely in `src/data_generation/validators.py`.

---

## Validating a Span Offset

Here's how we'll verify that a span annotation is correct in the validator:

```python
def verify_span(text: str, span: dict, tokenizer) -> bool:
    """Check that token_start:token_end correctly covers the original text."""
    tokens = tokenizer.tokenize(text)

    # Re-join the tokens in the span range
    span_tokens = tokens[span["token_start"]: span["token_end"]]
    recovered = tokenizer.convert_tokens_to_string(span_tokens).strip()

    # Check that it matches the annotated original
    return recovered == span["original"].strip()
```

This is what `validators.py` will do for every example — verify that the token offsets
actually point to the text they claim to cover.

---

## Practical Example: Full Annotation

```
Input text:  "Send the report to sarah.k@gmail.com by Friday"

Tokenized:
  Index:   0       1        2    3          4    5       6
  Token: ['Send', ' the', ' report', ' to', ' sarah', '.', 'k']
  Index:   7      8       9      10
  Token: ['@', 'gmail', '.', 'com', ' by', ' Friday']
           7      8       9   10     11      12

PII annotation:
  {
    "token_start": 4,   ← ' sarah' token
    "token_end":   10,  ← exclusive end — covers sarah.k@gmail.com
    "type":        "EMAIL",
    "original":    "sarah.k@gmail.com"
  }

Masked output:
  "Send the report to [MASKED_EMAIL] by Friday"
```

---

## Summary

```
Text → Tokenizer → [token_0, token_1, token_2, ...]
                        ↑
               Model operates here (token space)

Span = { token_start, token_end, type, original }

Why token-level:
  - Native to the model — no conversion needed
  - Stable after masking (no character offset shifts)
  - Handles multi-byte characters and special chars naturally
  - Adversarial inputs (l33tspeak, encoding) stay in token space

Key rule:
  token_end is EXCLUSIVE (like Python slicing)
  tokens[token_start : token_end] = the masked span
```

**Next:** Commit C docs, then move to the smoke test notebook → `notebooks/01_smoke_test.ipynb`.
