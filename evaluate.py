"""
CostGuard Agent — Routing Evaluation

Measures:
- Routing accuracy
- Per-tier precision and recall
- Confusion matrix
- Policy fallback rate
- Estimated routing cost
- Estimated cost against Nova Lite baseline

Important:
These metrics measure routing-tier agreement, not answer quality.
"""

import csv
import json
from datetime import datetime, timezone
from pathlib import Path

from config import (
    MODELS,
    BASELINE_MODEL,
    ESTIMATED_INPUT_TOKENS,
    ESTIMATED_OUTPUT_TOKENS,
)
from decider import CostGuardDecider


# ============================================================
# ORIGINAL EVALUATION DATASET
# ============================================================

TEST_CASES = [
    {"id": "T01", "query": "Hello", "expected": "cheap"},
    {"id": "T02", "query": "How are you?", "expected": "cheap"},
    {"id": "T03", "query": "What is the capital of India?", "expected": "cheap"},
    {"id": "T04", "query": "What is 15 multiplied by 8?", "expected": "cheap"},
    {
        "id": "T05",
        "query": "Explain the difference between a list and tuple in Python.",
        "expected": "medium",
    },
    {
        "id": "T06",
        "query": "Compare SQL and NoSQL databases with examples.",
        "expected": "medium",
    },
    {
        "id": "T07",
        "query": "Explain how HTTP requests work.",
        "expected": "medium",
    },
    {
        "id": "T08",
        "query": "Explain binary search and its time complexity.",
        "expected": "medium",
    },
    {
        "id": "T09",
        "query": (
            "Implement a binary search tree with insert, search, "
            "delete, and edge-case handling."
        ),
        "expected": "powerful",
    },
    {
        "id": "T10",
        "query": (
            "Design a multi-step debugging strategy for a "
            "distributed application with intermittent failures."
        ),
        "expected": "powerful",
    },
    {
        "id": "T11",
        "query": (
            "Analyze the time and space complexity of three "
            "different graph algorithms and compare their trade-offs."
        ),
        "expected": "powerful",
    },
    {
        "id": "T12",
        "query": (
            "Design a robust Python data-processing pipeline "
            "with validation, retries, and error recovery."
        ),
        "expected": "powerful",
    },
]


# ============================================================
# HOLDOUT DATASET
# Keep separate from policy tuning.
# ============================================================

HOLDOUT_CASES = [
    {
        "id": "H01",
        "query": "What is the boiling point of water?",
        "expected": "cheap",
    },
    {
        "id": "H02",
        "query": "Calculate 24 divided by 6.",
        "expected": "cheap",
    },
    {
        "id": "H03",
        "query": "Define a variable in Python with an example.",
        "expected": "medium",
    },
    {
        "id": "H04",
        "query": "Explain the difference between a stack and a queue.",
        "expected": "medium",
    },
    {
        "id": "H05",
        "query": "Explain how database indexes improve query performance.",
        "expected": "medium",
    },
    {
        "id": "H06",
        "query": "Describe the steps in an HTTP authentication flow.",
        "expected": "medium",
    },
    {
        "id": "H07",
        "query": (
            "Design a concurrent task scheduler with dependency "
            "management, retry handling, and failure recovery."
        ),
        "expected": "powerful",
    },
    {
        "id": "H08",
        "query": (
            "Analyze race conditions and deadlocks in a "
            "multithreaded application and propose solutions."
        ),
        "expected": "powerful",
    },
    {
        "id": "H09",
        "query": (
            "Design a scalable distributed system with fault "
            "tolerance, replication, and recovery mechanisms."
        ),
        "expected": "powerful",
    },
    {
        "id": "H10",
        "query": (
            "Compare graph traversal algorithms for a large "
            "sparse graph, including complexity trade-offs."
        ),
        "expected": "powerful",
    },
    {
        "id": "H11",
        "query": (
            "Build a production API architecture with rate limiting, "
            "authentication, observability, and graceful degradation."
        ),
        "expected": "powerful",
    },
    {
        "id": "H12",
        "query": (
            "Debug an intermittent production outage across "
            "multiple services and propose a verification plan."
        ),
        "expected": "powerful",
    },
]


OUTPUT_DIR = Path("evaluation_results")
TIERS = ("cheap", "medium", "powerful")


# ============================================================
# COST ESTIMATION
# ============================================================

def estimate_cost(
    tier,
    input_tokens=ESTIMATED_INPUT_TOKENS,
    output_tokens=ESTIMATED_OUTPUT_TOKENS,
):
    """
    Estimate cost using configured token rates.

    This is not an actual AWS billing calculation.
    """
    if tier not in MODELS:
        raise ValueError(f"Unknown model tier: {tier}")

    model = MODELS[tier]

    return (
        (input_tokens / 1000) * model["input_cost"]
        + (output_tokens / 1000) * model["output_cost"]
    )


# ============================================================
# CLASSIFICATION METRICS
# ============================================================

def calculate_tier_metrics(results):
    """Calculate one-vs-rest precision and recall for each tier."""
    metrics = {}

    for tier in TIERS:
        true_positive = sum(
            row["expected"] == tier and row["selected"] == tier
            for row in results
        )

        false_positive = sum(
            row["expected"] != tier and row["selected"] == tier
            for row in results
        )

        false_negative = sum(
            row["expected"] == tier and row["selected"] != tier
            for row in results
        )

        precision = (
            true_positive / (true_positive + false_positive)
            if true_positive + false_positive
            else 0.0
        )

        recall = (
            true_positive / (true_positive + false_negative)
            if true_positive + false_negative
            else 0.0
        )

        metrics[tier] = {
            "true_positive": true_positive,
            "false_positive": false_positive,
            "false_negative": false_negative,
            "precision_percent": round(precision * 100, 2),
            "recall_percent": round(recall * 100, 2),
        }

    return metrics


def calculate_confusion_matrix(results):
    """
    Rows represent expected tiers.
    Columns represent selected tiers.
    """
    matrix = {}

    for expected in TIERS:
        matrix[expected] = {}

        for selected in TIERS:
            matrix[expected][selected] = sum(
                row["expected"] == expected
                and row["selected"] == selected
                for row in results
            )

    return matrix


# ============================================================
# EVALUATION
# ============================================================

def evaluate(dataset, run_name):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")

    json_path = OUTPUT_DIR / f"{run_name}_{timestamp}.json"
    csv_path = OUTPUT_DIR / f"{run_name}_{timestamp}.csv"

    print("\n" + "=" * 68)
    print(f"COSTGUARD ROUTING EVALUATION: {run_name.upper()}")
    print("=" * 68)

    decider = CostGuardDecider()

    rows = []
    errors = 0
    fallback_count = 0

    for index, case in enumerate(dataset, start=1):
        print(f"\n[{index}/{len(dataset)}] {case['query']}")

        try:
            decision = decider.choose_model(case["query"])

            expected = case["expected"]
            selected = decision["selected"]
            complexity = decision.get("complexity") or {}

            is_correct = selected == expected

            if decision.get("fallback_applied"):
                fallback_count += 1

            row = {
                "id": case["id"],
                "query": case["query"],
                "expected": expected,
                "proposed": decision["proposed"],
                "selected": selected,
                "confidence": decision["confidence"],
                "complexity_level": complexity.get("level"),
                "fallback_applied": decision.get("fallback_applied"),
                "policy_reason": decision.get("policy_reason"),
                "correct": is_correct,
                "estimated_routed_cost": estimate_cost(selected),
                "estimated_baseline_cost": estimate_cost(BASELINE_MODEL),
                "error": None,
            }

            print(
                f"Expected={expected} | "
                f"Proposed={decision['proposed']} | "
                f"Selected={selected} | "
                f"Confidence={decision['confidence']:.2f} | "
                f"Complexity={complexity.get('level')} | "
                f"Correct={'YES' if is_correct else 'NO'}"
            )

        except Exception as exc:
            errors += 1

            row = {
                "id": case["id"],
                "query": case["query"],
                "expected": case["expected"],
                "proposed": None,
                "selected": None,
                "confidence": None,
                "complexity_level": None,
                "fallback_applied": None,
                "policy_reason": None,
                "correct": False,
                "estimated_routed_cost": None,
                "estimated_baseline_cost": None,
                "error": str(exc),
            }

            print(f"ERROR: {exc}")

        rows.append(row)

    # Errors count as incorrect in the primary accuracy metric.
    correct = sum(
        row["correct"]
        for row in rows
    )

    total = len(dataset)

    accuracy = (
        correct / total * 100
        if total
        else 0.0
    )

    successful_rows = [
        row for row in rows
        if row["selected"] in TIERS
    ]

    successful_accuracy = (
        sum(row["correct"] for row in successful_rows)
        / len(successful_rows) * 100
        if successful_rows
        else 0.0
    )

    routed_cost = sum(
        row["estimated_routed_cost"] or 0.0
        for row in rows
    )

    baseline_cost = sum(
        row["estimated_baseline_cost"] or 0.0
        for row in rows
    )

    estimated_savings = baseline_cost - routed_cost

    savings_percentage = (
        estimated_savings / baseline_cost * 100
        if baseline_cost
        else 0.0
    )

    tier_metrics = calculate_tier_metrics(successful_rows)
    matrix = calculate_confusion_matrix(successful_rows)

    summary = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "dataset": run_name,
        "total_cases": total,
        "successful_decisions": len(successful_rows),
        "errors": errors,
        "correct_cases": correct,
        "routing_accuracy_percent": round(accuracy, 2),
        "successful_decisions_accuracy_percent": round(
            successful_accuracy, 2
        ),
        "fallback_count": fallback_count,
        "fallback_rate_percent": round(
            fallback_count / total * 100, 2
        ) if total else 0.0,
        "per_tier_metrics": tier_metrics,
        "confusion_matrix_expected_rows_selected_columns": matrix,
        "estimated_routed_cost_usd": round(routed_cost, 10),
        "estimated_baseline_cost_usd": round(baseline_cost, 10),
        "estimated_savings_usd": round(estimated_savings, 10),
        "estimated_savings_percent": round(savings_percentage, 2),
        "cost_assumptions": {
            "input_tokens_per_query": ESTIMATED_INPUT_TOKENS,
            "output_tokens_per_query": ESTIMATED_OUTPUT_TOKENS,
            "baseline_model": BASELINE_MODEL,
            "note": (
                "Estimated from configured rates. "
                "Not actual AWS charges or proof of answer quality."
            ),
        },
        "metric_definition": (
            "Expected tiers are manually assigned. These metrics measure "
            "routing-tier agreement, not answer quality or calibrated confidence."
        ),
        "results": rows,
    }

    with json_path.open("w", encoding="utf-8") as file:
        json.dump(summary, file, indent=2, ensure_ascii=False)

    fieldnames = list(rows[0].keys()) if rows else []

    with csv_path.open(
        "w",
        newline="",
        encoding="utf-8-sig",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames,
        )

        writer.writeheader()
        writer.writerows(rows)

    print("\n" + "-" * 68)
    print(f"Dataset                 : {run_name}")
    print(f"Total cases             : {total}")
    print(f"Errors                  : {errors}")
    print(f"Correct selections      : {correct}")
    print(f"Routing accuracy        : {accuracy:.2f}%")
    print(f"Successful-only accuracy: {successful_accuracy:.2f}%")
    print(f"Fallback count          : {fallback_count}")
    print(f"Estimated routed cost   : ${routed_cost:.8f}")
    print(f"Estimated baseline cost : ${baseline_cost:.8f}")
    print(f"Estimated savings       : ${estimated_savings:.8f}")
    print(f"Estimated savings %     : {savings_percentage:.2f}%")

    print("\nPer-tier precision and recall:")

    for tier, metric in tier_metrics.items():
        print(
            f"{tier:10} "
            f"Precision={metric['precision_percent']:.2f}% | "
            f"Recall={metric['recall_percent']:.2f}%"
        )

    print("\nConfusion matrix (expected rows, selected columns):")
    print(f"{'Expected':12} {'Cheap':>10} {'Medium':>10} {'Powerful':>10}")

    for expected in TIERS:
        values = matrix[expected]

        print(
            f"{expected:12} "
            f"{values['cheap']:>10} "
            f"{values['medium']:>10} "
            f"{values['powerful']:>10}"
        )

    print(f"\nJSON report: {json_path}")
    print(f"CSV report : {csv_path}")

    return summary


# ============================================================
# ROUTING STRATEGY COMPARISON (SIMULATION ONLY)
# ============================================================

def simulate_strategy(rows, strategy_name):
    """
    Compare routing policies without calling Bedrock inference.

    A: Nova Lite for every request.
    B: Current CostGuard-selected tier.
    C: Downgrade selected Powerful requests to Medium.

    C is an experiment only. Tier-label agreement is not answer quality.
    """
    simulated_rows = []
    for row in rows:
        if row.get("selected") not in TIERS:
            continue

        if strategy_name == "A_baseline_nova_lite":
            selected = BASELINE_MODEL
        elif strategy_name == "B_current_costguard":
            selected = row["selected"]
        elif strategy_name == "C_powerful_to_medium":
            selected = "medium" if row["selected"] == "powerful" else row["selected"]
        else:
            raise ValueError(f"Unknown strategy: {strategy_name}")

        simulated_rows.append({
            "id": row["id"],
            "expected": row["expected"],
            "selected": selected,
            "correct": selected == row["expected"],
            "estimated_cost_usd": estimate_cost(selected),
        })

    total = len(simulated_rows)
    correct = sum(r["correct"] for r in simulated_rows)
    cost = sum(r["estimated_cost_usd"] for r in simulated_rows)
    baseline = sum(estimate_cost(BASELINE_MODEL) for _ in simulated_rows)
    savings = baseline - cost

    return {
        "strategy": strategy_name,
        "total_cases": total,
        "correct_cases": correct,
        "routing_accuracy_percent": round(100 * correct / total, 2) if total else 0.0,
        "estimated_cost_usd": round(cost, 10),
        "nova_lite_baseline_cost_usd": round(baseline, 10),
        "estimated_savings_usd": round(savings, 10),
        "estimated_savings_percent": round(100 * savings / baseline, 2) if baseline else 0.0,
        "note": "Simulation with fixed token assumptions/configured rates; no Bedrock inference or answer-quality measurement.",
    }


def compare_strategies(original_summary, holdout_summary):
    """Print and save A/B/C strategy simulation results."""
    comparison = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "simulation_only": True,
        "datasets": {},
    }
    datasets = {
        "original": original_summary["results"],
        "holdout": holdout_summary["results"],
    }
    strategies = (
        "A_baseline_nova_lite",
        "B_current_costguard",
        "C_powerful_to_medium",
    )

    for dataset_name, rows in datasets.items():
        results = [simulate_strategy(rows, strategy) for strategy in strategies]
        comparison["datasets"][dataset_name] = results

        print("\n" + "=" * 92)
        print(f"STRATEGY COMPARISON: {dataset_name.upper()} (SIMULATION ONLY)")
        print("=" * 92)
        print(f"{'Strategy':30} {'Accuracy':>10} {'Cost USD':>14} {'Savings %':>12}")
        for result in results:
            print(
                f"{result['strategy']:30} "
                f"{result['routing_accuracy_percent']:>9.2f}% "
                f"${result['estimated_cost_usd']:>13.8f} "
                f"{result['estimated_savings_percent']:>11.2f}%"
            )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    report_path = OUTPUT_DIR / f"strategy_comparison_{timestamp}.json"
    with report_path.open("w", encoding="utf-8") as file:
        json.dump(comparison, file, indent=2, ensure_ascii=False)

    print(f"\nStrategy comparison JSON: {report_path}")
    print("Reminder: C's tier-label accuracy does not prove Medium answers complex tasks adequately.")
    return comparison


# ============================================================
# RUN ORIGINAL AND HOLDOUT DATASETS
# ============================================================

def run_evaluations():
    original_results = evaluate(TEST_CASES, "original")
    holdout_results = evaluate(HOLDOUT_CASES, "holdout")
    compare_strategies(original_results, holdout_results)

    print("\n" + "=" * 68)
    print("FINAL COMPARISON")
    print("=" * 68)
    print(
        "Original accuracy: "
        f"{original_results['routing_accuracy_percent']:.2f}%"
    )
    print(
        "Holdout accuracy : "
        f"{holdout_results['routing_accuracy_percent']:.2f}%"
    )
    print("\\nImportant:")
    print("- Routing accuracy measures agreement with expected model tiers.")
    print("- It does not measure generated-answer quality.")
    print("- Estimated savings are not actual AWS billing savings.")
    print(
        "- If you tune policy using holdout failures, treat those examples "
        "as development data and create a fresh final holdout."
    )


if __name__ == "__main__":
    run_evaluations()