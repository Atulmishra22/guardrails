"""
teachers.py — Teacher API client with self-consistency voting.

Calls a frontier model (Gemini / Claude) to label each PromptSpec
with the full guardrail JSON schema including PII span annotations.

Self-consistency: same prompt is sent k times at temperature > 0.
The majority-vote decision is used. Disagreements flag a low-confidence example.

Read first: docs/04_tokenization.md (token-level spans)
"""

import asyncio
import json
import logging
import os
import time
from collections import Counter
from dataclasses import dataclass, field
from typing import Any

import aiohttp
from tenacity import retry, stop_after_attempt, wait_exponential

from .generators import PromptSpec

logger = logging.getLogger(__name__)

# ── Output schema ──────────────────────────────────────────────────────────────

VALID_DECISIONS   = {"ALLOW", "BLOCK", "SANITIZE"}
VALID_RISK_LEVELS = {"low", "medium", "high"}
VALID_MASK_TYPES  = {
    "NAME", "PII", "EMAIL", "PHONE", "ADDRESS",
    "CREDENTIAL", "FINANCIAL", "MEDICAL", "HARM",
}

# ── System prompt sent to the teacher model ────────────────────────────────────

TEACHER_SYSTEM_PROMPT = """You are a safety data labeling expert. Analyze the user input and produce ONLY a valid JSON object with this exact schema — no markdown, no explanation, no extra text.

{
  "decision": "ALLOW" | "BLOCK" | "SANITIZE",
  "category": "<one of: benign, self_harm, violence, sexual_content, child_safety, cyber_abuse, fraud, privacy, illegal_activity, weapons, hate, harassment, extremism, regulated_goods, prompt_injection, other>",
  "risk_level": "low" | "medium" | "high",
  "confidence": <float 0.0–1.0>,
  "masked_input": "<input with PII replaced by [MASKED_TYPE] tokens>",
  "mask_spans": [
    {"token_start": <int>, "token_end": <int>, "type": "<mask type>", "original": "<original text>"}
  ]
}

Rules:
- BLOCK: harmful content (violence, weapons, illegal, jailbreaks, self-harm, CSAM, prompt injection)
- SANITIZE: benign intent but contains private data (PII, credentials, financial, medical)
- ALLOW: safe, no private data
- mask_spans: token-level offsets relative to the input text (NOT character-level)
- mask_spans must be non-empty when decision is SANITIZE
- Always output valid JSON. mask_spans can be [] for ALLOW and BLOCK."""


# ── Dataclass for a single labeled example ────────────────────────────────────

@dataclass
class LabeledExample:
    """One fully labeled training example ready for JSONL serialization."""
    text:          str
    decision:      str
    category:      str
    risk_level:    str
    confidence:    float
    masked_input:  str
    mask_spans:    list[dict]
    tier:          str
    # Metadata — not saved to training JSONL, used for QC
    raw_responses: list[str] = field(default_factory=list, repr=False)
    agreement:     float     = 1.0   # Fraction of k samples that agreed on decision


# ── Teacher API client ─────────────────────────────────────────────────────────

class TeacherClient:
    """
    Async teacher API client supporting Gemini and Claude.

    Usage:
        async with TeacherClient(model="gemini-1.5-pro") as client:
            examples = await client.label_batch(prompt_specs)
    """

    SUPPORTED_MODELS = {
        "gemini-1.5-pro":        "google",
        "gemini-1.5-flash":      "google",
        "claude-3-5-sonnet-20241022": "anthropic",
        "claude-3-haiku-20240307":    "anthropic",
    }

    def __init__(
        self,
        model:        str   = "gemini-1.5-pro",
        temperature:  float = 0.7,
        k:            int   = 3,      # Self-consistency samples
        max_tokens:   int   = 512,
        batch_size:   int   = 10,
        retry_attempts: int = 3,
    ):
        self.model        = model
        self.temperature  = temperature
        self.k            = k
        self.max_tokens   = max_tokens
        self.batch_size   = batch_size
        self.retry_attempts = retry_attempts
        self._provider    = self.SUPPORTED_MODELS.get(model, "google")
        self._session: aiohttp.ClientSession | None = None

    async def __aenter__(self):
        self._session = aiohttp.ClientSession()
        return self

    async def __aexit__(self, *args):
        if self._session:
            await self._session.close()

    # ── Single example labeling ────────────────────────────────────────────────

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(min=2, max=10))
    async def _call_api(self, user_text: str) -> str:
        """Call the teacher API once. Returns the raw string response."""
        if self._provider == "google":
            return await self._call_gemini(user_text)
        return await self._call_claude(user_text)

    async def _call_gemini(self, user_text: str) -> str:
        api_key = os.environ["GEMINI_API_KEY"]
        url = (
            f"https://generativelanguage.googleapis.com/v1beta/models/"
            f"{self.model}:generateContent?key={api_key}"
        )
        payload = {
            "system_instruction": {"parts": [{"text": TEACHER_SYSTEM_PROMPT}]},
            "contents": [{"parts": [{"text": user_text}]}],
            "generationConfig": {
                "temperature":    self.temperature,
                "maxOutputTokens": self.max_tokens,
                "responseMimeType": "application/json",
            },
        }
        async with self._session.post(url, json=payload) as resp:
            resp.raise_for_status()
            data = await resp.json()
            return data["candidates"][0]["content"]["parts"][0]["text"]

    async def _call_claude(self, user_text: str) -> str:
        api_key = os.environ["ANTHROPIC_API_KEY"]
        url = "https://api.anthropic.com/v1/messages"
        headers = {
            "x-api-key":         api_key,
            "anthropic-version": "2023-06-01",
            "content-type":      "application/json",
        }
        payload = {
            "model":      self.model,
            "max_tokens": self.max_tokens,
            "system":     TEACHER_SYSTEM_PROMPT,
            "messages":   [{"role": "user", "content": user_text}],
            "temperature": self.temperature,
        }
        async with self._session.post(url, json=payload, headers=headers) as resp:
            resp.raise_for_status()
            data = await resp.json()
            return data["content"][0]["text"]

    # ── Self-consistency voting ────────────────────────────────────────────────

    async def label_one(self, spec: PromptSpec) -> LabeledExample | None:
        """
        Label a single prompt using self-consistency (k samples, majority vote).
        Returns None if all k calls fail to produce valid JSON.
        """
        raw_responses: list[str] = []
        parsed:        list[dict] = []

        # Gather k responses
        tasks = [self._call_api(spec.text) for _ in range(self.k)]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        for r in results:
            if isinstance(r, Exception):
                logger.warning("API call failed: %s", r)
                continue
            raw_responses.append(r)
            try:
                parsed.append(json.loads(r))
            except json.JSONDecodeError:
                logger.warning("Invalid JSON from teacher: %s", r[:100])

        if not parsed:
            return None  # All calls failed

        # Majority vote on decision
        decisions = [p.get("decision") for p in parsed if p.get("decision") in VALID_DECISIONS]
        if not decisions:
            return None

        majority_decision, majority_count = Counter(decisions).most_common(1)[0]
        agreement = majority_count / len(parsed)

        # Pick the response whose decision matches the majority
        best = next(p for p in parsed if p.get("decision") == majority_decision)

        return LabeledExample(
            text=spec.text,
            decision=majority_decision,
            category=best.get("category", "other"),
            risk_level=best.get("risk_level", "medium"),
            confidence=round(float(best.get("confidence", 0.5)) * agreement, 3),
            masked_input=best.get("masked_input", spec.text),
            mask_spans=best.get("mask_spans", []),
            tier=spec.tier,
            raw_responses=raw_responses,
            agreement=agreement,
        )

    # ── Batch labeling ─────────────────────────────────────────────────────────

    async def label_batch(
        self,
        specs: list[PromptSpec],
        progress_cb: Any = None,  # Optional callback(done, total)
    ) -> list[LabeledExample]:
        """
        Label a list of prompts in batches to respect API rate limits.
        Returns only successfully labeled examples (failed ones are skipped).
        """
        examples: list[LabeledExample] = []
        total = len(specs)

        for i in range(0, total, self.batch_size):
            batch = specs[i: i + self.batch_size]
            tasks = [self.label_one(s) for s in batch]
            batch_results = await asyncio.gather(*tasks)

            for result in batch_results:
                if result is not None:
                    examples.append(result)

            done = min(i + self.batch_size, total)
            if progress_cb:
                progress_cb(done, total)
            else:
                logger.info("Labeled %d / %d", done, total)

            # Polite rate-limit pause between batches
            if i + self.batch_size < total:
                await asyncio.sleep(1.0)

        return examples


# ── JSONL serialization ────────────────────────────────────────────────────────

def example_to_jsonl(ex: LabeledExample) -> str:
    """Serialize a LabeledExample to a single JSONL line (training format)."""
    record = {
        "text":         ex.text,
        "decision":     ex.decision,
        "category":     ex.category,
        "risk_level":   ex.risk_level,
        "confidence":   ex.confidence,
        "masked_input": ex.masked_input,
        "mask_spans":   ex.mask_spans,
        "tier":         ex.tier,
    }
    return json.dumps(record, ensure_ascii=False)
