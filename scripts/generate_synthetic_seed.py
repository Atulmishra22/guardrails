#!/usr/bin/env python3
import json
import random
import re
import argparse
import sys
from pathlib import Path

# Add src to path to import generators
sys.path.insert(0, str(Path(__file__).parent.parent))

from transformers import AutoTokenizer

from src.data_generation.generators import (
    _sample_pii, 
    OBVIOUS_SAFE, 
    OBVIOUS_UNSAFE, 
    AMBIGUOUS, 
    DUAL_USE, 
    ADVERSARIAL, 
    BORDERLINE, 
    PII_SANITIZE
)

def get_token_offsets(tokenizer, text: str, char_start: int, char_end: int) -> tuple[int, int]:
    """Convert character offsets to token offsets using the fast tokenizer's offset mapping."""
    # We must not add special tokens to keep index mapping aligned with raw tokenization
    encoded = tokenizer(text, return_offsets_mapping=True, add_special_tokens=False)
    offsets = encoded["offset_mapping"]
    
    token_start = None
    token_end = None
    
    for idx, (ostart, oend) in enumerate(offsets):
        # some tokens might have (0,0), skip them
        if ostart == oend == 0:
            continue
            
        if token_start is None and oend > char_start:
            token_start = idx
            
        if ostart < char_end:
            token_end = idx + 1
            
    # Fallback if somehow not found
    if token_start is None: token_start = 0
    if token_end is None: token_end = token_start + 1
    
    # In some edge cases token_end could be <= token_start if it's a single token
    if token_end <= token_start:
        token_end = token_start + 1
        
    return token_start, token_end


def synthesize_dataset(output_path: Path, n: int = 500, seed: int = 42):
    random.seed(seed)
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-1.5B-Instruct", use_fast=True)
    
    tier_weights = {
        "obvious":     0.20,
        "ambiguous":   0.20,
        "dual_use":    0.20,
        "adversarial": 0.25,
        "borderline":  0.15,
    }
    
    tier_pool = list(tier_weights.keys())
    tier_probs = [tier_weights[t] for t in tier_pool]
    
    template_map = {
        "obvious":     OBVIOUS_SAFE + OBVIOUS_UNSAFE,
        "ambiguous":   AMBIGUOUS,
        "dual_use":    DUAL_USE,
        "adversarial": ADVERSARIAL,
        "borderline":  BORDERLINE,
    }
    all_pii = PII_SANITIZE

    # Target counts:
    target_allow = int(n * 0.40)
    target_block = int(n * 0.30)
    target_sanitize = int(n * 0.30)
    
    counts = {"ALLOW": 0, "BLOCK": 0, "SANITIZE": 0}
    
    # Map PII keys to VALID_MASK_TYPES
    pii_type_map = {
        "name": "NAME",
        "email": "EMAIL",
        "ssn": "PII",
        "phone": "PHONE",
        "address": "ADDRESS",
        "api_key": "CREDENTIAL",
        "card_number": "FINANCIAL",
        "expiry": "FINANCIAL",
        "iban": "FINANCIAL",
        "dob": "MEDICAL",
        "symptom": "MEDICAL",
        "country": "ADDRESS", 
        "harmful_request": "HARM",
        "harmful_request_leet": "HARM",
        "b64_encoded_harm": "HARM",
        "weapon": "HARM",
        "malware_action": "HARM",
        "drug": "HARM",
        "victim_desc": "HARM",
        "violent_act": "HARM",
        "medication": "MEDICAL",
        "hacking_concept": "HARM",
        "dangerous_topic": "HARM",
    }

    results = []
    
    # We loop until we reach exactly the counts we need
    while counts["ALLOW"] < target_allow or counts["BLOCK"] < target_block or counts["SANITIZE"] < target_sanitize:
        tier = random.choices(tier_pool, weights=tier_probs, k=1)[0]
        
        # Decide if this is a sanitize prompt
        is_sanitize = random.random() < 0.30
        
        if is_sanitize:
            template, decision, category = random.choice(all_pii)
        else:
            template, decision, category = random.choice(template_map[tier])
            
        # We enforce exactly matching target proportions
        if counts[decision] >= (target_allow if decision == "ALLOW" else target_block if decision == "BLOCK" else target_sanitize):
            continue
            
        pii = _sample_pii()
        
        # Find which placeholders are in the template
        placeholders = re.findall(r'\{([^{}]+)\}', template)
        
        text = template.format(**pii)
        
        mask_spans = []
        masked_input = text
        
        # Track replaced indices to avoid double-replacing overlapping strings
        replaced = set()
        
        for ph in placeholders:
            if ph not in pii: continue
            
            val = str(pii[ph])
            if ph in pii_type_map and (decision == "SANITIZE" or decision == "BLOCK" or pii_type_map[ph] == "HARM"):
                span_type = pii_type_map[ph]
                
                if decision == "ALLOW" and span_type == "HARM":
                    continue  # Do not emit HARM spans for ALLOW.
                
                if decision == "BLOCK" and span_type != "HARM":
                    continue
                
                char_start = text.find(val)
                if char_start != -1 and char_start not in replaced:
                    char_end = char_start + len(val)
                    token_start, token_end = get_token_offsets(tokenizer, text, char_start, char_end)
                    
                    mask_spans.append({
                        "token_start": token_start,
                        "token_end": token_end,
                        "type": span_type,
                        "original": val
                    })
                    
                    masked_input = masked_input.replace(val, f"[{span_type}]")
                    replaced.add(char_start)
                    
        # Sort spans by token_start
        mask_spans.sort(key=lambda x: x["token_start"])
        
        record = {
            "text": text,
            "decision": decision,
            "category": category,
            "risk_level": "low" if decision == "ALLOW" else "high",
            "confidence": 0.99,
            "masked_input": masked_input,
            "mask_spans": mask_spans,
            "tier": tier
        }
        
        results.append(record)
        counts[decision] += 1
        
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
            
    print(f"Generated {len(results)} records to {output_path}")
    print(f"Counts: {counts}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("data/generated/seed.jsonl"))
    parser.add_argument("--n", type=int, default=500)
    args = parser.parse_args()
    
    synthesize_dataset(args.output, args.n)
