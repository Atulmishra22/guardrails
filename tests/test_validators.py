"""
test_validators.py — Unit tests for dataset validation routines.
"""

import json
from pathlib import Path
import pytest

from src.data_generation.validators import (
    check_class_balance,
    detect_label_contradictions,
    find_near_duplicates,
    validate_schema,
)


@pytest.fixture
def valid_allow_record():
    return {
        "text": "What is the capital of France?",
        "decision": "ALLOW",
        "category": "benign",
        "risk_level": "low",
        "confidence": 0.98,
        "masked_input": "What is the capital of France?",
        "mask_spans": [],
    }


@pytest.fixture
def valid_sanitize_record():
    return {
        "text": "Call me at 555-0199 regarding the job.",
        "decision": "SANITIZE",
        "category": "privacy",
        "risk_level": "medium",
        "confidence": 0.95,
        "masked_input": "Call me at [MASKED_PHONE] regarding the job.",
        "mask_spans": [
            {
                "token_start": 3,
                "token_end": 5,
                "type": "PHONE",
                "original": "555-0199",
            }
        ],
    }


def test_validate_schema_valid_records(valid_allow_record, valid_sanitize_record):
    res_allow = validate_schema(valid_allow_record)
    assert res_allow.is_valid
    assert len(res_allow.errors) == 0

    res_san = validate_schema(valid_sanitize_record)
    assert res_san.is_valid
    assert len(res_san.errors) == 0


def test_validate_schema_missing_key(valid_allow_record):
    del valid_allow_record["confidence"]
    res = validate_schema(valid_allow_record)
    assert not res.is_valid
    assert any("Missing keys" in err for err in res.errors)


def test_validate_schema_invalid_decision(valid_allow_record):
    valid_allow_record["decision"] = "IGNORE"
    res = validate_schema(valid_allow_record)
    assert not res.is_valid
    assert any("Invalid decision" in err for err in res.errors)


def test_validate_schema_sanitize_without_spans():
    record = {
        "text": "My email is test@corp.com",
        "decision": "SANITIZE",
        "category": "privacy",
        "risk_level": "medium",
        "confidence": 0.90,
        "masked_input": "My email is [MASKED_EMAIL]",
        "mask_spans": [],
    }
    res = validate_schema(record)
    assert not res.is_valid
    assert any("SANITIZE decision must have at least one mask_span" in err for err in res.errors)


def test_validate_schema_invalid_span_boundary():
    record = {
        "text": "User Jane Doe logged in",
        "decision": "SANITIZE",
        "category": "privacy",
        "risk_level": "medium",
        "confidence": 0.92,
        "masked_input": "User [MASKED_NAME] logged in",
        "mask_spans": [
            {
                "token_start": 4,
                "token_end": 2,  # start > end
                "type": "NAME",
                "original": "Jane Doe",
            }
        ],
    }
    res = validate_schema(record)
    assert not res.is_valid
    assert any("must be < token_end" in err for err in res.errors)


def test_label_contradiction_detection(tmp_path: Path):
    jsonl_path = tmp_path / "test_contradictions.jsonl"
    records = [
        {"text": "How do I pick a padlock?", "decision": "BLOCK"},
        {"text": "How do I pick a padlock?", "decision": "ALLOW"},
        {"text": "Safe benign query", "decision": "ALLOW"},
    ]
    with open(jsonl_path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")

    contradictions = detect_label_contradictions(jsonl_path)
    assert len(contradictions) == 1
    assert "how do i pick a padlock?" in contradictions[0]["text_snippet"].lower()
    assert set(contradictions[0]["decisions"]) == {"BLOCK", "ALLOW"}


def test_find_near_duplicates(tmp_path: Path):
    jsonl_path = tmp_path / "test_dupes.jsonl"
    records = [
        {"text": "What is the capital city of France?"},
        {"text": "What is the capital city of France?"},  # exact match
        {"text": "Explain quantum entanglement in physics."},
    ]
    with open(jsonl_path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")

    dupes = find_near_duplicates(jsonl_path, similarity_threshold=0.90)
    assert len(dupes) >= 1
    assert dupes[0][0] == 1
    assert dupes[0][1] == 2
