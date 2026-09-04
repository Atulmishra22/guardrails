"""
test_metrics.py — Unit tests for decision and span-level evaluation metrics.
"""

import pytest
from src.evaluation.metrics import (
    compute_decision_metrics,
    compute_json_validity,
    compute_span_metrics,
    spans_match,
)


def test_spans_match():
    span_a = {"token_start": 3, "token_end": 5, "type": "EMAIL"}
    span_b = {"token_start": 3, "token_end": 5, "type": "EMAIL"}
    span_diff_type = {"token_start": 3, "token_end": 5, "type": "NAME"}
    span_diff_end = {"token_start": 3, "token_end": 6, "type": "EMAIL"}

    assert spans_match(span_a, span_b)
    assert not spans_match(span_a, span_diff_type)
    assert not spans_match(span_a, span_diff_end)


def test_compute_span_metrics_perfect_match():
    gold = [
        {"token_start": 2, "token_end": 4, "type": "NAME"},
        {"token_start": 8, "token_end": 10, "type": "PHONE"},
    ]
    pred = [
        {"token_start": 2, "token_end": 4, "type": "NAME"},
        {"token_start": 8, "token_end": 10, "type": "PHONE"},
    ]
    metrics = compute_span_metrics(pred, gold)
    assert metrics["span_precision"] == 1.0
    assert metrics["span_recall"] == 1.0
    assert metrics["span_f1"] == 1.0
    assert metrics["tp_spans"] == 2
    assert metrics["fp_spans"] == 0
    assert metrics["fn_spans"] == 0


def test_compute_span_metrics_partial():
    gold = [
        {"token_start": 2, "token_end": 4, "type": "NAME"},
        {"token_start": 8, "token_end": 10, "type": "PHONE"},
    ]
    # pred found NAME, missed PHONE, predicted spurious EMAIL
    pred = [
        {"token_start": 2, "token_end": 4, "type": "NAME"},
        {"token_start": 12, "token_end": 15, "type": "EMAIL"},
    ]
    metrics = compute_span_metrics(pred, gold)
    assert metrics["tp_spans"] == 1
    assert metrics["fp_spans"] == 1
    assert metrics["fn_spans"] == 1
    assert metrics["span_precision"] == 0.5
    assert metrics["span_recall"] == 0.5
    assert metrics["span_f1"] == 0.5


def test_compute_decision_metrics():
    preds = ["ALLOW", "BLOCK", "SANITIZE", "ALLOW", "BLOCK"]
    golds = ["ALLOW", "BLOCK", "ALLOW", "ALLOW", "SANITIZE"]

    res = compute_decision_metrics(preds, golds)
    assert res["accuracy"] == 0.6  # 3/5 correct
    assert "BLOCK" in res
    assert "ALLOW" in res
    assert "SANITIZE" in res

    # BLOCK class: 1 TP (index 1), 1 FP (index 4 predicted BLOCK for gold SANITIZE)
    assert res["BLOCK"]["tp"] == 1
    assert res["BLOCK"]["fp"] == 1
    assert res["BLOCK"]["fn"] == 0
    assert res["BLOCK"]["precision"] == 0.5
    assert res["BLOCK"]["recall"] == 1.0


def test_compute_json_validity():
    outputs = [
        '{"decision": "ALLOW", "category": "benign", "risk_level": "low", "confidence": 0.9, "masked_input": "...", "mask_spans": []}',
        'Invalid JSON format here',
        '{"incomplete": "keys"}',
    ]
    validity = compute_json_validity(outputs)
    assert validity["total"] == 3
    assert validity["json_valid_count"] == 2
    assert validity["schema_complete_count"] == 1
    assert round(validity["json_validity_rate"], 2) == 0.67
    assert round(validity["schema_complete_rate"], 2) == 0.33
