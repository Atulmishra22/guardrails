#!/usr/bin/env python3
import sys
import argparse
import time
import json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.inference.guardrail import GuardrailPredictor
from src.evaluation.metrics import aggregate_results, print_eval_report, score_example
from src.evaluation.benchmark import run_adversarial_suite

def main():
    parser = argparse.ArgumentParser(description="Run full evaluation benchmark")
    parser.add_argument("--eval_data", type=str, default="data/generated/eval.jsonl", help="Path to JSONL eval dataset")
    parser.add_argument("--model", type=str, default="indiginous/guardrail-qwen-1.5b", help="Hugging Face model ID")
    parser.add_argument("--output", type=str, default="data/evaluation/benchmark_report.json", help="Path to save report")
    parser.add_argument("--max_examples", type=int, default=None, help="Limit number of eval examples (for fast CPU runs)")
    parser.add_argument("--skip_adversarial", action="store_true", help="Skip running the adversarial suite")
    
    args = parser.parse_args()

    print(f"\n==================================================")
    print(f"Running Benchmark Evaluation")
    print(f"   Model:   {args.model}")
    print(f"   Dataset: {args.eval_data}")
    print(f"==================================================\n", flush=True)

    predictor = GuardrailPredictor(
        base_model_name=args.model,
        checkpoint_path=None,
        load_in_4bit=True,
        device="auto"
    )

    gold_records = []
    with open(args.eval_data, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                gold_records.append(json.loads(line))

    if args.max_examples:
        gold_records = gold_records[:args.max_examples]

    print(f"\n[1/2] Running standard evaluation suite ({len(gold_records)} examples)...", flush=True)
    
    results = []
    for i, record in enumerate(gold_records):
        print(f"  [{i+1}/{len(gold_records)}] Evaluating: {record['text'][:50]}...", flush=True)
        start = time.perf_counter()
        pred = predictor.predict(record["text"])
        latency_ms = (time.perf_counter() - start) * 1000
        
        raw_output = json.dumps({
            "decision": pred.decision,
            "category": pred.category,
            "risk_level": getattr(pred, "risk_level", "low"),
            "confidence": pred.confidence,
            "masked_input": pred.masked_input,
            "mask_spans": pred.mask_spans
        })
        
        scored = score_example(i, record, raw_output)
        scored.latency_ms = latency_ms
        results.append(scored)
        print(f"    -> {pred.decision} (expected {record['decision']}) [{latency_ms:.0f}ms]", flush=True)

    report = aggregate_results(results)
    
    adv_results = None
    if not getattr(args, "skip_adversarial", False):
        print("\n[2/2] Running adversarial jailbreak suite...", flush=True)
        adv_results = run_adversarial_suite(predictor.model, predictor.tokenizer)

    # Print formatted evaluation report
    print_eval_report(report, model_name=args.model)

    if adv_results is not None:
        print(f"\n  Adversarial Pass Rate: {adv_results['pass_rate']:.1%} ({adv_results['passed']}/{adv_results['total']})")
        print("=" * 58)

    report_dict = {
        "total": report.total,
        "json_validity": report.json_validity,
        "decision_metrics": report.decision_metrics,
        "span_aggregate": report.span_aggregate,
        "per_tier": report.per_tier,
        "per_tier_span": report.per_tier_span,
        "adversarial": adv_results,
    }
    
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(report_dict, f, indent=2, default=lambda x: x.__dict__)
        
    print(f"\nFull JSON report saved to: {args.output}")

if __name__ == "__main__":
    main()
