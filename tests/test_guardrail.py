"""
test_guardrail.py — Unit tests for GuardrailPredictor using mock inference.
"""

from unittest.mock import MagicMock, patch
import pytest

from src.inference.guardrail import GuardrailOutput, GuardrailPredictor


@pytest.fixture
def mock_predictor():
    """Instantiate GuardrailPredictor with mocked model and tokenizer."""
    with patch("src.inference.guardrail.AutoTokenizer.from_pretrained") as mock_tok, \
         patch("src.inference.guardrail.AutoModelForCausalLM.from_pretrained") as mock_model:

        tok_instance = MagicMock()
        tok_instance.pad_token = None
        tok_instance.eos_token = "<|im_end|>"
        tok_instance.pad_token_id = 0
        tok_instance.apply_chat_template.return_value = "mock_prompt"
        mock_tok.return_value = tok_instance

        model_instance = MagicMock()
        model_instance.device = "cpu"
        mock_model.return_value = model_instance

        predictor = GuardrailPredictor(
            checkpoint_path=None,
            base_model_name="mock-model",
            load_in_4bit=False,
            confidence_threshold=0.80,
        )
        return predictor


def test_predict_valid_allow(mock_predictor):
    raw_json = (
        '{"decision": "ALLOW", "category": "benign", "risk_level": "low", '
        '"confidence": 0.99, "masked_input": "Hello world", "mask_spans": []}'
    )

    with patch.object(mock_predictor, "tokenizer") as tok:
        tok.apply_chat_template.return_value = "prompt"
        tok.decode.return_value = raw_json
        tok.return_value = {"input_ids": MagicMock(shape=[1, 5])}

        with patch.object(mock_predictor.model, "generate") as gen:
            gen.return_value = [[0, 1, 2, 3, 4, 5, 6]]
            output: GuardrailOutput = mock_predictor.predict("Hello world")

    assert output.decision == "ALLOW"
    assert output.category == "benign"
    assert output.is_valid_json is True
    assert output.gate_override is None


def test_predict_fail_closed_on_invalid_json(mock_predictor):
    invalid_raw = "I am a helpful assistant and cannot fulfill this request."

    with patch.object(mock_predictor, "tokenizer") as tok:
        tok.apply_chat_template.return_value = "prompt"
        tok.decode.return_value = invalid_raw
        tok.return_value = {"input_ids": MagicMock(shape=[1, 5])}

        with patch.object(mock_predictor.model, "generate") as gen:
            gen.return_value = [[0, 1, 2, 3, 4, 5, 6]]
            output = mock_predictor.predict("Test query")

    # Gate rule 1: fail-closed to BLOCK
    assert output.decision == "BLOCK"
    assert output.is_valid_json is False
    assert output.gate_override == "fail_closed_invalid_json"


def test_predict_override_on_harm_span(mock_predictor):
    # Model attempted to SANITIZE instead of BLOCK on a HARM span
    raw_json = (
        '{"decision": "SANITIZE", "category": "violence", "risk_level": "high", '
        '"confidence": 0.95, "masked_input": "[MASKED_HARM]", '
        '"mask_spans": [{"token_start": 0, "token_end": 2, "type": "HARM", "original": "kill"}]}'
    )

    with patch.object(mock_predictor, "tokenizer") as tok:
        tok.apply_chat_template.return_value = "prompt"
        tok.decode.return_value = raw_json
        tok.return_value = {"input_ids": MagicMock(shape=[1, 5])}

        with patch.object(mock_predictor.model, "generate") as gen:
            gen.return_value = [[0, 1, 2, 3, 4, 5, 6]]
            output = mock_predictor.predict("kill")

    # Gate rule 2: HARM span must never forward -> hard override to BLOCK
    assert output.decision == "BLOCK"
    assert output.gate_override == "override_harm_span_detected"


def test_predict_override_on_low_confidence_sanitize(mock_predictor):
    # Model predicted SANITIZE but confidence is 0.65 (< 0.80 threshold)
    raw_json = (
        '{"decision": "SANITIZE", "category": "privacy", "risk_level": "medium", '
        '"confidence": 0.65, "masked_input": "...", '
        '"mask_spans": [{"token_start": 0, "token_end": 1, "type": "NAME", "original": "Bob"}]}'
    )

    with patch.object(mock_predictor, "tokenizer") as tok:
        tok.apply_chat_template.return_value = "prompt"
        tok.decode.return_value = raw_json
        tok.return_value = {"input_ids": MagicMock(shape=[1, 5])}

        with patch.object(mock_predictor.model, "generate") as gen:
            gen.return_value = [[0, 1, 2, 3, 4, 5, 6]]
            output = mock_predictor.predict("Bob")

    # Gate rule 3: low confidence sanitize -> fallback to BLOCK
    assert output.decision == "BLOCK"
    assert "override_low_confidence_sanitize" in str(output.gate_override)
