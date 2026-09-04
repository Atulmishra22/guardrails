"""
validators.py — Dataset quality control for generated JSONL examples.

Two responsibilities split into two sections:
  Part 1 (this commit): JSON schema validation + token-level span offset verification
  Part 2 (next commit):  Class balance checking + near-duplicate detection

Read first: docs/04_tokenization.md (why token-level spans, how to verify them)
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# ── Valid value sets (mirrors configs/data.yaml) ───────────────────────────────

VALID_DECISIONS = {"ALLOW", "BLOCK", "SANITIZE"}

VALID_CATEGORIES = {
    "benign", "self_harm", "violence", "sexual_content", "child_safety",
    "cyber_abuse", "fraud", "privacy", "illegal_activity", "weapons",
    "hate", "harassment", "extremism", "regulated_goods", "prompt_injection", "other",
}

VALID_RISK_LEVELS = {"low", "medium", "high"}

VALID_MASK_TYPES = {
    "NAME", "PII", "EMAIL", "PHONE", "ADDRESS",
    "CREDENTIAL", "FINANCIAL", "MEDICAL", "HARM",
}

REQUIRED_TOP_KEYS = {
    "text", "decision", "category", "risk_level",
    "confidence", "masked_input", "mask_spans",
}

REQUIRED_SPAN_KEYS = {"token_start", "token_end", "type", "original"}


# ── Validation result dataclass ────────────────────────────────────────────────

@dataclass
class ValidationResult:
    """Outcome of validating one JSONL example."""
    is_valid:  bool
    errors:    list[str] = field(default_factory=list)
    warnings:  list[str] = field(default_factory=list)

    def add_error(self, msg: str) -> None:
        self.errors.append(msg)
        self.is_valid = False

    def add_warning(self, msg: str) -> None:
        self.warnings.append(msg)


# ── Part 1A: JSON schema validation ───────────────────────────────────────────

def validate_schema(record: dict[str, Any]) -> ValidationResult:
    """
    Validate a single parsed JSON record against the guardrail output schema.

    Checks:
      - All required top-level keys are present
      - decision, category, risk_level are valid enum values
      - confidence is a float in [0.0, 1.0]
      - mask_spans is a list, each span has required keys and valid type
      - masked_input is non-empty string
      - SANITIZE decision must have at least one mask_span
      - ALLOW/BLOCK with HARM span type → should be BLOCK (warning)
    """
    result = ValidationResult(is_valid=True)

    # ── Required keys ──────────────────────────────────────────────────────────
    missing = REQUIRED_TOP_KEYS - set(record.keys())
    if missing:
        result.add_error(f"Missing keys: {missing}")
        return result   # Cannot continue checking without required keys

    # ── Decision ──────────────────────────────────────────────────────────────
    if record["decision"] not in VALID_DECISIONS:
        result.add_error(f"Invalid decision '{record['decision']}'. Must be one of {VALID_DECISIONS}")

    # ── Category ──────────────────────────────────────────────────────────────
    if record["category"] not in VALID_CATEGORIES:
        result.add_error(f"Invalid category '{record['category']}'")

    # ── Risk level ────────────────────────────────────────────────────────────
    if record["risk_level"] not in VALID_RISK_LEVELS:
        result.add_error(f"Invalid risk_level '{record['risk_level']}'")

    # ── Confidence ────────────────────────────────────────────────────────────
    try:
        conf = float(record["confidence"])
        if not (0.0 <= conf <= 1.0):
            result.add_error(f"confidence {conf} out of range [0.0, 1.0]")
    except (TypeError, ValueError):
        result.add_error(f"confidence must be a float, got '{record['confidence']}'")

    # ── masked_input ──────────────────────────────────────────────────────────
    if not isinstance(record["masked_input"], str) or not record["masked_input"].strip():
        result.add_error("masked_input must be a non-empty string")

    # ── mask_spans ────────────────────────────────────────────────────────────
    if not isinstance(record["mask_spans"], list):
        result.add_error("mask_spans must be a list")
        return result

    if record["decision"] == "SANITIZE" and len(record["mask_spans"]) == 0:
        result.add_error("SANITIZE decision must have at least one mask_span")

    for i, span in enumerate(record["mask_spans"]):
        span_result = _validate_span(span, i)
        result.errors.extend(span_result.errors)
        result.warnings.extend(span_result.warnings)
        if not span_result.is_valid:
            result.is_valid = False

    # ── Business logic warnings ────────────────────────────────────────────────
    harm_spans = [s for s in record["mask_spans"] if s.get("type") == "HARM"]
    if harm_spans and record["decision"] != "BLOCK":
        result.add_warning(
            f"HARM span detected but decision is '{record['decision']}' — should be BLOCK"
        )

    return result


def _validate_span(span: Any, idx: int) -> ValidationResult:
    """Validate a single span object within mask_spans."""
    result = ValidationResult(is_valid=True)
    prefix = f"mask_spans[{idx}]"

    if not isinstance(span, dict):
        result.add_error(f"{prefix}: must be a dict, got {type(span).__name__}")
        return result

    missing = REQUIRED_SPAN_KEYS - set(span.keys())
    if missing:
        result.add_error(f"{prefix}: missing keys {missing}")
        return result

    # token_start and token_end must be non-negative integers
    for key in ("token_start", "token_end"):
        val = span[key]
        if not isinstance(val, int) or val < 0:
            result.add_error(f"{prefix}.{key} must be a non-negative int, got '{val}'")

    # token_start < token_end
    if isinstance(span.get("token_start"), int) and isinstance(span.get("token_end"), int):
        if span["token_start"] >= span["token_end"]:
            result.add_error(
                f"{prefix}: token_start ({span['token_start']}) must be < token_end ({span['token_end']})"
            )

    # type must be valid
    if span.get("type") not in VALID_MASK_TYPES:
        result.add_error(f"{prefix}.type '{span.get('type')}' invalid. Must be one of {VALID_MASK_TYPES}")

    # original must be a non-empty string
    if not isinstance(span.get("original"), str) or not span["original"].strip():
        result.add_error(f"{prefix}.original must be a non-empty string")

    return result


# ── Part 1B: Token-level span offset verification ─────────────────────────────

def verify_span_offsets(record: dict[str, Any], tokenizer: Any) -> ValidationResult:
    """
    Re-tokenize the input text and verify that each span's token offsets
    actually correspond to the claimed `original` text.

    This catches cases where the teacher model produced plausible-looking
    offsets that don't actually align with the tokenized text.

    Args:
        record:    A parsed JSONL record (must have passed validate_schema first).
        tokenizer: A HuggingFace tokenizer (AutoTokenizer instance).

    Returns:
        ValidationResult with errors for any misaligned spans.
    """
    result = ValidationResult(is_valid=True)

    tokens = tokenizer.tokenize(record["text"])

    for i, span in enumerate(record.get("mask_spans", [])):
        start = span.get("token_start")
        end   = span.get("token_end")

        if not (isinstance(start, int) and isinstance(end, int)):
            continue  # Already caught by validate_schema

        if end > len(tokens):
            result.add_error(
                f"mask_spans[{i}]: token_end {end} > total tokens {len(tokens)}"
            )
            continue

        # Reconstruct the text covered by the span
        span_tokens   = tokens[start:end]
        reconstructed = tokenizer.convert_tokens_to_string(span_tokens).strip()
        original      = span.get("original", "").strip()

        # Allow minor whitespace differences
        if reconstructed.lower() != original.lower():
            result.add_error(
                f"mask_spans[{i}]: offset mismatch — "
                f"tokens[{start}:{end}] = '{reconstructed}' "
                f"but original = '{original}'"
            )

    return result


# ── Batch validation over a JSONL file ────────────────────────────────────────

@dataclass
class BatchValidationReport:
    """Summary of validating an entire JSONL file."""
    total:       int = 0
    valid:       int = 0
    invalid:     int = 0
    warnings:    int = 0
    error_details: list[dict] = field(default_factory=list)

    @property
    def validity_rate(self) -> float:
        return self.valid / self.total if self.total else 0.0


def validate_jsonl_file(
    path: Path,
    tokenizer: Any | None = None,
    max_errors_logged: int = 20,
) -> BatchValidationReport:
    """
    Validate every record in a JSONL file.

    Args:
        path:               Path to the JSONL file.
        tokenizer:          If provided, also runs span offset verification.
        max_errors_logged:  Cap on how many error examples to store.

    Returns:
        BatchValidationReport with overall statistics.
    """
    report = BatchValidationReport()

    with open(path, encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue

            report.total += 1

            # Parse JSON
            try:
                record = json.loads(line)
            except json.JSONDecodeError as e:
                report.invalid += 1
                if len(report.error_details) < max_errors_logged:
                    report.error_details.append({
                        "line": line_no, "errors": [f"JSON parse error: {e}"]
                    })
                continue

            # Schema validation
            schema_result = validate_schema(record)

            # Span offset verification (optional — requires tokenizer)
            span_result = ValidationResult(is_valid=True)
            if tokenizer is not None and schema_result.is_valid:
                span_result = verify_span_offsets(record, tokenizer)

            all_errors   = schema_result.errors + span_result.errors
            all_warnings = schema_result.warnings + span_result.warnings

            if all_errors:
                report.invalid += 1
                if len(report.error_details) < max_errors_logged:
                    report.error_details.append({"line": line_no, "errors": all_errors})
            else:
                report.valid += 1

            if all_warnings:
                report.warnings += len(all_warnings)

    return report


# ── Part 2: Class balance analysis ────────────────────────────────────────────

from collections import Counter  # noqa: E402


@dataclass
class BalanceReport:
    """
    Summary of class distribution in a JSONL dataset.

    Why this matters:
      If 80% of examples are ALLOW, the model learns to predict ALLOW for
      everything and gets 80% accuracy — but completely fails on BLOCK cases.
    """
    total:              int
    decision_counts:    dict[str, int]
    category_counts:    dict[str, int]
    tier_counts:        dict[str, int]
    imbalance_warnings: list[str]

    @property
    def decision_fractions(self) -> dict[str, float]:
        return {k: v / self.total for k, v in self.decision_counts.items()}


def check_class_balance(
    path: Path,
    target_fractions: dict[str, float] | None = None,
    tolerance: float = 0.10,
) -> BalanceReport:
    """
    Analyze class distribution and warn on imbalances vs target fractions.

    Default targets (from configs/data.yaml):
        ALLOW=0.40, BLOCK=0.30, SANITIZE=0.30
    """
    if target_fractions is None:
        target_fractions = {"ALLOW": 0.40, "BLOCK": 0.30, "SANITIZE": 0.30}

    decision_counts: Counter = Counter()
    category_counts: Counter = Counter()
    tier_counts:     Counter = Counter()
    total = 0

    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            total += 1
            decision_counts[record.get("decision", "UNKNOWN")] += 1
            category_counts[record.get("category", "UNKNOWN")] += 1
            tier_counts[record.get("tier", "UNKNOWN")]         += 1

    warnings: list[str] = []
    if total > 0:
        for cls, target in target_fractions.items():
            actual = decision_counts.get(cls, 0) / total
            if abs(actual - target) > tolerance:
                warnings.append(
                    f"Class '{cls}': target={target:.0%}  actual={actual:.0%}  "
                    f"diff={abs(actual-target):.0%} > tolerance={tolerance:.0%}"
                )

    return BalanceReport(
        total=total,
        decision_counts=dict(decision_counts),
        category_counts=dict(category_counts),
        tier_counts=dict(tier_counts),
        imbalance_warnings=warnings,
    )


def print_balance_report(report: BalanceReport) -> None:
    """Pretty-print a BalanceReport to stdout."""
    print(f"\n{'='*52}")
    print(f"  Dataset balance report   (n={report.total})")
    print(f"{'='*52}")
    print("\nDecision distribution:")
    for decision, count in sorted(report.decision_counts.items()):
        frac = count / report.total if report.total else 0
        bar  = "█" * int(frac * 30)
        print(f"  {decision:10s}: {count:5d}  {frac:.1%}  {bar}")
    print("\nTop 10 categories:")
    for cat, count in Counter(report.category_counts).most_common(10):
        print(f"  {cat:20s}: {count}")
    print("\nTier distribution:")
    for tier, count in sorted(report.tier_counts.items()):
        print(f"  {tier:12s}: {count}")
    if report.imbalance_warnings:
        print("\nWARNING — Imbalance detected:")
        for w in report.imbalance_warnings:
            print(f"   {w}")
    else:
        print("\nClass balance within tolerance.")


# ── Part 2: Near-duplicate detection ──────────────────────────────────────────

def _shingle(text: str, k: int = 3) -> set[str]:
    """Return k-character shingles for Jaccard similarity estimation."""
    text = text.lower().strip()
    return {text[i: i + k] for i in range(len(text) - k + 1)}


def find_near_duplicates(
    path: Path,
    similarity_threshold: float = 0.85,
) -> list[tuple[int, int, float]]:
    """
    Flag near-duplicate records using Jaccard similarity over 3-character shingles.
    Only run on the seed set (≤500 examples) — O(n²) complexity.

    Why: Near-duplicates bias the model to memorize instead of generalize,
    and inflate evaluation metrics by leaking into the train set.

    Returns: list of (line_i, line_j, similarity) for flagged pairs.
    """
    texts: list[str] = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
                texts.append(record.get("text", ""))
            except json.JSONDecodeError:
                texts.append("")

    shingles = [_shingle(t) for t in texts]
    duplicates: list[tuple[int, int, float]] = []

    for i in range(len(shingles)):
        for j in range(i + 1, len(shingles)):
            a, b = shingles[i], shingles[j]
            if not a or not b:
                continue
            jaccard = len(a & b) / len(a | b)
            if jaccard >= similarity_threshold:
                duplicates.append((i + 1, j + 1, round(jaccard, 3)))

    return duplicates


def detect_label_contradictions(path: Path) -> list[dict]:
    """
    Find examples with identical text but conflicting decision labels.
    Contradictions produce conflicting training signal and must be removed.

    Returns: list of {text_snippet, lines, decisions} dicts.
    """
    seen: dict[str, list[tuple[int, str]]] = {}

    with open(path, encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                record   = json.loads(line)
                text     = record.get("text", "").strip().lower()
                decision = record.get("decision", "")
                if text:
                    seen.setdefault(text, []).append((line_no, decision))
            except json.JSONDecodeError:
                continue

    contradictions = []
    for text, entries in seen.items():
        if len({d for _, d in entries}) > 1:
            contradictions.append({
                "text_snippet": text[:80],
                "lines":        [ln for ln, _ in entries],
                "decisions":    list({d for _, d in entries}),
            })

    return contradictions
