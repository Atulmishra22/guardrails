"""
validate_dataset.py — CLI script to run all QC checks on a JSONL dataset.

Runs schema validation, span offset verification, class balance analysis,
near-duplicate detection, and label contradiction detection.

Usage:
    uv run scripts/validate_dataset.py --input data/generated/seed.jsonl
    uv run scripts/validate_dataset.py --input data/generated/seed.jsonl --with-spans
"""

import argparse
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data_generation.validators import (
    BatchValidationReport,
    check_class_balance,
    detect_label_contradictions,
    find_near_duplicates,
    print_balance_report,
    validate_jsonl_file,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

SECTION = "=" * 56


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run all QC checks on a guardrail JSONL dataset."
    )
    parser.add_argument(
        "--input", type=Path, required=True,
        help="Path to the JSONL file to validate.",
    )
    parser.add_argument(
        "--with-spans", action="store_true",
        help="Also verify token-level span offsets (requires transformers + model download).",
    )
    parser.add_argument(
        "--model", type=str, default="Qwen/Qwen2.5-1.5B-Instruct",
        help="Tokenizer model to use for span verification. Only used with --with-spans.",
    )
    parser.add_argument(
        "--dup-threshold", type=float, default=0.85,
        help="Jaccard similarity threshold for near-duplicate detection. Default: 0.85",
    )
    parser.add_argument(
        "--balance-tolerance", type=float, default=0.10,
        help="Allowed deviation from target class fraction. Default: 0.10",
    )
    parser.add_argument(
        "--fail-below", type=float, default=0.95,
        help="Exit with code 1 if validity rate drops below this threshold. Default: 0.95",
    )
    return parser.parse_args()


def section(title: str) -> None:
    print(f"\n{SECTION}")
    print(f"  {title}")
    print(SECTION)


def main() -> None:
    args = parse_args()

    if not args.input.exists():
        logger.error("File not found: %s", args.input)
        sys.exit(1)

    tokenizer = None
    if args.with_spans:
        try:
            from transformers import AutoTokenizer
            logger.info("Loading tokenizer %s for span verification...", args.model)
            tokenizer = AutoTokenizer.from_pretrained(args.model)
        except ImportError:
            logger.error("transformers not installed. Run: uv sync --extra dev")
            sys.exit(1)

    # ── 1. Schema + span validation ───────────────────────────────────────────
    section("1. Schema and Span Offset Validation")
    report: BatchValidationReport = validate_jsonl_file(
        args.input, tokenizer=tokenizer
    )
    print(f"  Total records   : {report.total}")
    print(f"  Valid           : {report.valid}  ({report.validity_rate:.1%})")
    print(f"  Invalid         : {report.invalid}")
    print(f"  Warnings        : {report.warnings}")

    if report.error_details:
        print(f"\n  First {len(report.error_details)} error(s):")
        for detail in report.error_details:
            print(f"    Line {detail['line']}: {detail['errors']}")

    # ── 2. Class balance ──────────────────────────────────────────────────────
    section("2. Class Balance Analysis")
    balance = check_class_balance(args.input, tolerance=args.balance_tolerance)
    print_balance_report(balance)

    # ── 3. Near-duplicate detection ───────────────────────────────────────────
    section("3. Near-Duplicate Detection")
    if report.total > 500:
        print(f"  Skipping O(n^2) duplicate check — dataset has {report.total} records.")
        print("  Run on the seed set (<=500) only, then deduplicate before scaling.")
        duplicates = []
    else:
        duplicates = find_near_duplicates(args.input, similarity_threshold=args.dup_threshold)
        print(f"  Threshold       : Jaccard >= {args.dup_threshold}")
        print(f"  Duplicate pairs : {len(duplicates)}")
        if duplicates:
            print("\n  Top duplicates (line_i, line_j, similarity):")
            for pair in sorted(duplicates, key=lambda x: -x[2])[:10]:
                print(f"    Lines {pair[0]} & {pair[1]}  — similarity={pair[2]}")

    # ── 4. Label contradiction detection ─────────────────────────────────────
    section("4. Label Contradiction Detection")
    contradictions = detect_label_contradictions(args.input)
    print(f"  Contradictions  : {len(contradictions)}")
    if contradictions:
        print("\n  Examples:")
        for c in contradictions[:5]:
            print(f"    Lines {c['lines']}  decisions={c['decisions']}")
            print(f"    Text: \"{c['text_snippet']}\"")

    # ── Summary ───────────────────────────────────────────────────────────────
    section("Summary")
    issues = report.invalid + len(duplicates) + len(contradictions) + len(balance.imbalance_warnings)
    print(f"  Schema errors   : {report.invalid}")
    print(f"  Duplicate pairs : {len(duplicates)}")
    print(f"  Contradictions  : {len(contradictions)}")
    print(f"  Balance warnings: {len(balance.imbalance_warnings)}")
    print(f"  Total issues    : {issues}")

    if report.validity_rate < args.fail_below:
        logger.error(
            "Validity rate %.1f%% is below threshold %.1f%%. Fix errors before training.",
            report.validity_rate * 100, args.fail_below * 100,
        )
        sys.exit(1)

    if issues == 0:
        print("\n  All checks passed. Dataset is ready for training.")
    else:
        print("\n  Review the issues above before moving to training.")


if __name__ == "__main__":
    main()
