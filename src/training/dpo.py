"""
dpo.py — Direct Preference Optimization training pipeline.

Fine-tunes the guardrail-v1 (SFT) model using preference pairs mined from
hard examples and security edge cases. Operates directly on (prompt, chosen, rejected)
triplets without requiring a separate reward model.

Prerequisites:
  - guardrail-v1-sft adapter saved at models/guardrail-v1-sft
  - preference data mined at data/preferences/train_pref.jsonl
  - docs/03_dpo_explained.md

Memory footprint on 16GB GPU:
  - 4-bit policy model:    ~0.9 GB
  - 4-bit reference model: ~0.9 GB (frozen, no optimizer states)
  - LoRA adapters:         ~0.1 GB
  - Activations & buffers: ~4.0 GB
  Total:                   ~6.0 GB (comfortably fits T4 16GB limit)
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Any

import torch
import yaml
from datasets import load_dataset
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from trl import DPOConfig, DPOTrainer

logger = logging.getLogger(__name__)


# ── Config Loader ──────────────────────────────────────────────────────────────

def load_dpo_config(config_path: str | Path = "configs/dpo.yaml") -> dict[str, Any]:
    """Load DPO configuration from YAML file."""
    with open(config_path, encoding="utf-8") as f:
        return yaml.safe_load(f)


# ── Quantization Configuration ────────────────────────────────────────────────

def get_bnb_config() -> BitsAndBytesConfig:
    """Standard 4-bit NF4 configuration matching SFT stage."""
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
    )


# ── Model and Reference Loader ─────────────────────────────────────────────────

def load_dpo_models(
    cfg: dict[str, Any],
    base_model_name: str = "Qwen/Qwen2.5-1.5B-Instruct",
) -> tuple[Any, Any, Any]:
    """
    Load the policy model (with trainable LoRA adapters) and the frozen
    reference model (SFT baseline).

    Both models are loaded in 4-bit NF4 to minimize VRAM usage.
    """
    sft_checkpoint = cfg["model"]["sft_checkpoint"]
    bnb_config = get_bnb_config()

    logger.info("Loading tokenizer: %s", base_model_name)
    tokenizer = AutoTokenizer.from_pretrained(
        base_model_name,
        trust_remote_code=True,
        padding_side="right",
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # 1. Load policy model: base model + SFT adapter (trainable)
    logger.info("Loading policy model from SFT checkpoint: %s", sft_checkpoint)
    policy_base = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
    )
    policy_model = PeftModel.from_pretrained(
        policy_base,
        sft_checkpoint,
        is_trainable=True,
    )

    # 2. Load reference model: base model + SFT adapter (frozen)
    logger.info("Loading frozen reference model from SFT checkpoint: %s", sft_checkpoint)
    ref_base = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
    )
    ref_model = PeftModel.from_pretrained(
        ref_base,
        sft_checkpoint,
        is_trainable=False,
    )
    ref_model.eval()

    return policy_model, ref_model, tokenizer


# ── Training Arguments ─────────────────────────────────────────────────────────

def build_dpo_config(cfg: dict[str, Any]) -> DPOConfig:
    """Construct DPOConfig matching TRL specifications."""
    t = cfg["training"]
    d = cfg["dpo"]

    return DPOConfig(
        output_dir=cfg["model"]["output_dir"],
        beta=d["beta"],
        loss_type=d.get("loss_type", "sigmoid"),
        max_prompt_length=d["max_prompt_length"],
        max_length=d["max_length"],
        num_train_epochs=t["num_train_epochs"],
        per_device_train_batch_size=t["per_device_train_batch_size"],
        gradient_accumulation_steps=t["gradient_accumulation_steps"],
        learning_rate=float(t["learning_rate"]),
        lr_scheduler_type=t["lr_scheduler_type"],
        warmup_ratio=t["warmup_ratio"],
        weight_decay=t["weight_decay"],
        max_grad_norm=t["max_grad_norm"],
        bf16=t["bf16"],
        gradient_checkpointing=t["gradient_checkpointing"],
        logging_steps=t["logging_steps"],
        eval_steps=t["eval_steps"],
        save_steps=t["save_steps"],
        save_total_limit=t["save_total_limit"],
        load_best_model_at_end=t["load_best_model_at_end"],
        metric_for_best_model=t["metric_for_best_model"],
        report_to=t["report_to"],
        eval_strategy="steps",
        remove_unused_columns=False,
    )


# ── Dataset Split Loader ───────────────────────────────────────────────────────

def load_preference_dataset(
    data_path: str | Path,
    eval_split: float = 0.10,
    seed: int = 42,
):
    """
    Load preference dataset JSONL.
    Expects rows: {"prompt": str, "chosen": str, "rejected": str}
    """
    raw_dataset = load_dataset("json", data_files=str(data_path), split="train")
    split = raw_dataset.train_test_split(test_size=eval_split, seed=seed)
    logger.info(
        "Preference dataset: %d train pairs, %d eval pairs",
        len(split["train"]), len(split["test"]),
    )
    return split["train"], split["test"]


# ── Main Training Routine ──────────────────────────────────────────────────────

def train_dpo(
    config_path:     str | Path = "configs/dpo.yaml",
    pref_data_path:  str | Path = "data/preferences/train_pref.jsonl",
    base_model_name: str        = "Qwen/Qwen2.5-1.5B-Instruct",
) -> None:
    """Run full DPO training pipeline and export adapter checkpoint."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(message)s",
        datefmt="%H:%M:%S",
    )

    cfg = load_dpo_config(config_path)

    # Validate SFT checkpoint exists
    sft_ckpt = Path(cfg["model"]["sft_checkpoint"])
    if not sft_ckpt.exists():
        logger.error(
            "SFT checkpoint not found at %s. Train guardrail-v1-sft first.",
            sft_ckpt,
        )
        raise FileNotFoundError(f"Missing SFT checkpoint: {sft_ckpt}")

    policy_model, ref_model, tokenizer = load_dpo_models(
        cfg, base_model_name=base_model_name
    )

    train_ds, eval_ds = load_preference_dataset(pref_data_path)
    dpo_args = build_dpo_config(cfg)

    trainer = DPOTrainer(
        model=policy_model,
        ref_model=ref_model,
        args=dpo_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        tokenizer=tokenizer,
    )

    logger.info("Starting DPO training...")
    trainer.train()

    output_dir = Path(cfg["model"]["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    trainer.model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    logger.info("DPO fine-tuned adapter saved to %s", output_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run DPO training on SFT guardrail model.")
    parser.add_argument("--config", default="configs/dpo.yaml", help="Path to dpo.yaml config.")
    parser.add_argument("--data",   default="data/preferences/train_pref.jsonl", help="Path to mined preference pairs.")
    parser.add_argument("--base",   default="Qwen/Qwen2.5-1.5B-Instruct", help="Base model identifier.")
    args = parser.parse_args()

    train_dpo(
        config_path=args.config,
        pref_data_path=args.data,
        base_model_name=args.base,
    )
