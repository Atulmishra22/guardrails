# Evaluation Metrics: F1, FNR, FPR, and Span-Level F1

> Read this before writing `src/evaluation/metrics.py`.
> These are the numbers that decide whether guardrail-v1 is ready for DPO.

---

## Why Standard Accuracy Is Useless Here

If 40% of inputs are SANITIZE, 35% are BLOCK, and 25% are ALLOW, a model
that always predicts ALLOW gets 25% accuracy. A model that always predicts
SANITIZE gets 40% accuracy. Neither is useful. Neither detects any harm.

We need metrics that expose these failure modes explicitly.

---

## Decision-Level Metrics

### Confusion Matrix

For a binary classifier (e.g., BLOCK vs not-BLOCK):

```
                  Predicted BLOCK    Predicted not-BLOCK
Actual BLOCK         TP                    FN
Actual not-BLOCK     FP                    TN
```

- **TP (True Positive)**: correctly flagged as BLOCK
- **FN (False Negative)**: harmful input that slipped through as ALLOW/SANITIZE
- **FP (False Positive)**: safe input incorrectly flagged as BLOCK
- **TN (True Negative)**: safe input correctly passed

### False Negative Rate (FNR) — Most Important for Safety

```
FNR = FN / (FN + TP) = missed harms / total actual harms
```

A missed harm (FN) is a real privacy leak or harmful content forwarded to the
cloud LLM. **This is the worst failure mode.** FNR must be minimized.

If FNR = 0.15, the model misses 15% of all genuinely harmful inputs.
Target: FNR < 0.15 after SFT, < 0.10 after DPO.

### False Positive Rate (FPR) — Cost of Over-Blocking

```
FPR = FP / (FP + TN) = safe inputs blocked / total safe inputs
```

A false positive means a legitimate user gets their request rejected. This
hurts usability. We tolerate higher FPR than FNR — over-blocking is
annoying, under-blocking is a security failure.

Target: FPR < 0.10.

### Precision and Recall (F1)

```
Precision = TP / (TP + FP)   — of everything we blocked, how much was actually harmful?
Recall    = TP / (TP + FN)   — of everything actually harmful, how much did we block?

F1 = 2 * (Precision * Recall) / (Precision + Recall)
```

F1 balances both. But in this project, **Recall is more important than Precision**
because missing a harm is worse than over-blocking.

We track F1 but weight Recall more heavily when making training decisions.

---

## Span-Level Metrics — The Core of PII Masking Quality

Decision accuracy (ALLOW/BLOCK/SANITIZE) tells you if the model made the
right call. Span metrics tell you whether the PII was actually masked correctly.

A model can output `decision=SANITIZE` but mask the wrong tokens. That is still
a privacy failure — the real PII went to the cloud LLM unmasked.

### What a Span Is

```
Input:    "Email me at john.smith@gmail.com about the project"
                           ↑                ↑
                       token 3           token 8

Ground truth span: { token_start: 3, token_end: 8, type: "EMAIL", original: "john.smith@gmail.com" }
Predicted span:    { token_start: 3, token_end: 8, type: "EMAIL", original: "john.smith@gmail.com" }
→ True Positive (exact match)

Predicted span:    { token_start: 3, token_end: 6, type: "EMAIL", original: "john.smith" }
→ Partial match (only part of the email masked)
→ Counted as False Negative (the @gmail.com leaked)
```

### Span Recall — The Single Most Important Metric

```
Span Recall = matched spans / total ground truth spans
```

Span Recall measures whether the model found all the PII that existed.
A missed span = a real privacy leak regardless of what the decision label was.

**Span Recall is our #1 metric.** Every training decision is made to improve it.

Under-masking (low recall) is always worse than over-masking (low precision):
- Low recall: "john.smith@gmail.com" → "[MASKED_EMAIL].com" (partial mask — @gmail.com leaked)
- Low precision: "2.7 trillion GDP" → "[MASKED_FINANCIAL]" (over-masked — annoying but not dangerous)

### Span Precision

```
Span Precision = matched spans / total predicted spans
```

How many of the spans the model predicted were actually correct?
Low precision means over-masking — legitimate content is being hidden.

### Span F1

```
Span F1 = 2 * (Span Precision * Span Recall) / (Span Precision + Span Recall)
```

Tracks both directions. Target: Span F1 > 0.75 after SFT, > 0.80 after DPO.

### Span Type Accuracy

Beyond detecting that something should be masked, does the model label it
with the correct type? Masking a PHONE as NAME is still useful (the data is
hidden) but the type error signals the model does not understand the taxonomy.

```
Span Type Accuracy = correctly typed TP spans / all TP spans
```

---

## Per-Tier Breakdown

Aggregate metrics hide where the model fails. Always report per difficulty tier:

```
Tier           | Decision Acc | Span Recall | Span F1
───────────────|──────────────|─────────────|─────────
obvious        | 0.95         | 0.92        | 0.91
ambiguous      | 0.72         | 0.68        | 0.65
dual_use       | 0.80         | 0.74        | 0.72
adversarial    | 0.60         | 0.55        | 0.52
borderline     | 0.65         | 0.61        | 0.58
```

Adversarial and borderline will always score lowest. The improvement loop
targets those tiers specifically with DPO preference pairs.

---

## JSON Validity Rate

Before any of the above metrics are meaningful, the model must output valid JSON.
If the model produces freeform text, every metric is undefined.

```
JSON Validity Rate = valid JSON outputs / total outputs
```

Target: >99%. If this is below 99% after SFT, do not proceed to DPO.
Fix the training data format first.

---

## What "Matched Span" Means

Two spans are a match if:
1. `token_start` matches exactly
2. `token_end` matches exactly
3. `type` matches exactly

Optional stricter version: also require `original` text match.

We use exact boundary matching because a span that misses even one token
allows that token's content to leak to the cloud LLM unmasked.

---

## Summary Table

| Metric | Formula | Target (post-SFT) | Target (post-DPO) | Priority |
|---|---|---|---|---|
| JSON Validity | valid / total | >99% | >99% | Prerequisite |
| Decision Accuracy | correct / total | >80% | >87% | High |
| Span Recall | TP_spans / GT_spans | >65% | >80% | Highest |
| Span Precision | TP_spans / Pred_spans | >60% | >75% | Medium |
| Span F1 | harmonic mean | >62% | >77% | High |
| FNR (harm) | FN / (FN+TP) | <20% | <12% | High |
| FPR (safe blocked) | FP / (FP+TN) | <15% | <10% | Medium |

Next: `src/evaluation/metrics.py` — implement all of the above.
