"""
generate_data.py — CLI script to run the full data generation pipeline.

Orchestrates: generators.py -> teachers.py -> validators.py -> JSONL output.

Usage:
    uv run scripts/generate_data.py --n 500 --output data/generated/seed.jsonl
    uv run scripts/generate_data.py --n 10000 --output data/generated/train.jsonl --seed 123
"""

import argparse
import asyncio
import json
import logging
import sys
from pathlib import Path

# Add project root to path so src imports work
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data_generation.generators import generate_prompts
from src.data_generation.teachers import TeacherClient, example_to_jsonl
from src.data_generation.validators import validate_jsonl_file

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate synthetic guardrail training data via teacher API."
    )
    parser.add_argument(
        "--n", type=int, required=True,
        help="Number of examples to generate (500 for seed, 10000 for full dataset).",
    )
    parser.add_argument(
        "--output", type=Path, required=True,
        help="Output JSONL file path. E.g. data/generated/seed.jsonl",
    )
    parser.add_argument(
        "--model", type=str, default="gemini-2.5-flash",
        help="Teacher model name. Default: gemini-2.5-flash (fast & free-tier friendly). Also supports gemini-1.5-flash, gpt-4o-mini, aipipe.",
    )
    parser.add_argument(
        "--temperature", type=float, default=0.7,
        help="Sampling temperature for teacher API. Default: 0.7",
    )
    parser.add_argument(
        "--k", type=int, default=1,
        help="Number of self-consistency samples per prompt. Default: 1 (fast/quota-efficient).",
    )
    parser.add_argument(
        "--batch-size", type=int, default=5,
        help="Concurrent API calls per batch. Default: 5 (safe for free-tier RPM).",
    )
    parser.add_argument(
        "--base-url", type=str, default=None,
        help="Custom OpenAI-compatible base URL (e.g. for AIPipe, OpenRouter, Groq).",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for prompt generator reproducibility. Default: 42",
    )
    parser.add_argument(
        "--validate", action="store_true",
        help="Run schema validation on the output file after generation.",
    )
    return parser.parse_args()


async def run(args: argparse.Namespace) -> None:
    output_path: Path = args.output
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Generate prompts
    logger.info("Generating %d prompts (seed=%d)...", args.n, args.seed)
    specs = list(generate_prompts(n=args.n, seed=args.seed))
    logger.info("Prompt generation complete.")

    # Label via teacher API
    def progress(done: int, total: int) -> None:
        pct = done / total * 100
        logger.info("Labeled %d / %d  (%.1f%%)", done, total, pct)

    async with TeacherClient(
        model=args.model,
        temperature=args.temperature,
        k=args.k,
        batch_size=args.batch_size,
        base_url=args.base_url,
    ) as client:
        examples = await client.label_batch(specs, progress_cb=progress)

    logger.info("Received %d labeled examples (%.1f%% success rate).",
                len(examples), len(examples) / args.n * 100)

    # Write JSONL
    with open(output_path, "w", encoding="utf-8") as f:
        for ex in examples:
            f.write(example_to_jsonl(ex) + "\n")

    logger.info("Written to %s", output_path)

    # Optional post-generation validation
    if args.validate:
        logger.info("Running schema validation...")
        report = validate_jsonl_file(output_path)
        logger.info(
            "Validation: %d/%d valid  (%.1f%%)  |  %d warnings",
            report.valid, report.total,
            report.validity_rate * 100,
            report.warnings,
        )
        if report.error_details:
            logger.warning("First %d errors:", len(report.error_details))
            for detail in report.error_details:
                logger.warning("  Line %d: %s", detail["line"], detail["errors"])
        if report.validity_rate < 0.95:
            logger.error(
                "Validity rate %.1f%% is below 95%%. Review errors before training.",
                report.validity_rate * 100,
            )
            sys.exit(1)


def main() -> None:
    args = parse_args()
    asyncio.run(run(args))


if __name__ == "__main__":
    main()
