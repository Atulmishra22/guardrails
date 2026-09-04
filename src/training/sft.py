"""
sft.py — Supervised Fine-Tuning pipeline for the guardrail model.

Two sections (two commits):
  G2 (this commit): model loading, quantization config, LoRA config
  G3 (next commit):  SFTTrainer setup, training args, formatting function, entry point

Read first:
  docs/05_quantization.md  — why 4-bit NF4
  docs/01_what_is_lora.md  — LoRA rank, alpha, target modules
  docs/02_sft_explained.md — completion-only loss masking
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import torch
import yaml
from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

logger = logging.getLogger(__name__)


# ── Config loader ──────────────────────────────────────────────────────────────

def load_config(config_path: str | Path = "configs/sft.yaml") -> dict:
    """Load and return the SFT config from YAML."""
    with open(config_path, encoding="utf-8") as f:
        return yaml.safe_load(f)


# ── Quantization config ────────────────────────────────────────────────────────

def build_bnb_config(cfg: dict) -> BitsAndBytesConfig:
    """
    Build BitsAndBytesConfig from the quantization section of sft.yaml.

    Concepts covered in docs/05_quantization.md:
      - load_in_4bit: store weights as 4-bit NF4 on GPU
      - bnb_4bit_quant_type: NF4 is bell-curve-aware, better than int4 for LLMs
      - bnb_4bit_use_double_quant: quantize the quantization constants too (~0.4 GB saved)
      - bnb_4bit_compute_dtype: dequantize to bf16 for actual matrix multiply
    """
    q = cfg["quantization"]
    dtype_map = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}

    return BitsAndBytesConfig(
        load_in_4bit=q["load_in_4bit"],
        bnb_4bit_quant_type=q["bnb_4bit_quant_type"],
        bnb_4bit_use_double_quant=q["bnb_4bit_use_double_quant"],
        bnb_4bit_compute_dtype=dtype_map[q["bnb_4bit_compute_dtype"]],
    )


# ── LoRA config ────────────────────────────────────────────────────────────────

def build_lora_config(cfg: dict) -> LoraConfig:
    """
    Build LoraConfig from the lora section of sft.yaml.

    Concepts covered in docs/01_what_is_lora.md:
      - r: rank of the low-rank adapter matrices A and B
      - lora_alpha: scaling factor applied to LoRA updates (alpha/r = effective scale)
      - target_modules: which linear layers get LoRA adapters injected
      - lora_dropout: regularization applied to adapter outputs during training
      - bias="none": do not add trainable bias terms to LoRA layers
    """
    l = cfg["lora"]
    return LoraConfig(
        r=l["r"],
        lora_alpha=l["lora_alpha"],
        lora_dropout=l["lora_dropout"],
        bias=l["bias"],
        task_type=TaskType.CAUSAL_LM,
        target_modules=l["target_modules"],
    )


# ── Model loader ───────────────────────────────────────────────────────────────

def load_model_and_tokenizer(
    cfg: dict,
) -> tuple[AutoModelForCausalLM, AutoTokenizer]:
    """
    Load the quantized base model and its tokenizer.

    Steps:
      1. Load tokenizer from HuggingFace Hub (or local cache).
      2. Load model in 4-bit using BitsAndBytesConfig.
      3. Call prepare_model_for_kbit_training — enables gradient checkpointing
         and casts non-quantized layers (norms, embeddings) to float32 for
         training stability.
      4. Inject LoRA adapters via get_peft_model.

    Args:
        cfg: Parsed sft.yaml config dict.

    Returns:
        (model, tokenizer) tuple ready for SFTTrainer.
    """
    model_name = cfg["model"]["name"]
    logger.info("Loading tokenizer: %s", model_name)

    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=True,
        padding_side="right",   # Right-pad for causal LM training
    )
    # Qwen2.5 uses eos_token as pad_token by default — make explicit
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    logger.info("Loading model in 4-bit: %s", model_name)
    bnb_config = build_bnb_config(cfg)

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map="auto",          # Spread layers across available GPU(s)
        trust_remote_code=True,
        torch_dtype=torch.bfloat16, # Non-quantized layers use bf16
    )

    # Required before injecting LoRA into a quantized model:
    # - Enables gradient checkpointing inside the quantized forward pass
    # - Upcasts layer norms and input embeddings to float32 for stability
    model = prepare_model_for_kbit_training(
        model,
        use_gradient_checkpointing=cfg["training"]["gradient_checkpointing"],
    )

    # Inject LoRA adapters — only adapter params will be trainable
    lora_config = build_lora_config(cfg)
    model = get_peft_model(model, lora_config)

    # Log trainable parameter count
    trainable, total = 0, 0
    for param in model.parameters():
        total += param.numel()
        if param.requires_grad:
            trainable += param.numel()
    logger.info(
        "Trainable parameters: %s / %s  (%.2f%%)",
        f"{trainable:,}", f"{total:,}", 100 * trainable / total,
    )

    return model, tokenizer
