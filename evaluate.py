import csv
import json
from datetime import datetime
from pathlib import Path

from decider import CostGuardDecider


# ==================================================
# ORIGINAL EVALUATION DATASET
# ==================================================

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


# ==================================================
# SEPARATE HOLDOUT DATASET
# ==================================================

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
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def evaluate(dataset, run_name):
    """Evaluate routing decisions and save separate JSON/CSV reports."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    json_path = OUTPUT_DIR / f"{run_name}_{timestamp}.json"
    csv_path = OUTPUT_DIR / f"{run_name}_{timestamp}.csv"

    print("\n" + "=" * 65)
    print(f"COSTGUARD ROUTING EVALUATION: {run_name.upper()}")
    print("=" * 65)
    print("\nLoading routing decider...")

    decider = CostGuardDecider()

    results = []
    correct = 0
    errors = 0
    fallback_count = 0
    total = len(dataset)

    for index, case in enumerate(dataset, start=1):
        print(f"\n[{index}/{total}] {case['query']}")

        try:
            decision = decider.choose_model(case["query"])

            selected = decision["selected"]
            expected = case["expected"]
            is_correct = selected == expected

            if is_correct:
                correct += 1

            if decision["fallback_applied"]:
                fallback_count += 1

            row = {
                "id": case["id"],
                "query": case["query"],
                "expected": expected,
                "proposed": decision["proposed"],
                "selected": selected,
                "confidence": decision["confidence"],
                "fallback_applied": decision["fallback_applied"],
                "policy_reason": decision.get("policy_reason", "N/A"),
                "correct": is_correct,
                "error": None,
            }

            print(
                f"Expected: {expected} | "
                f"Proposed: {decision['proposed']} | "
                f"Selected: {selected} | "
                f"Confidence: {decision['confidence']:.2f} | "
                f"Fallback: {decision['fallback_applied']} | "
                f"Match: {'YES' if is_correct else 'NO'}"
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
                "fallback_applied": None,
                "policy_reason": None,
                "correct": False,
                "error": str(exc),
            }

            print(f"ERROR: {exc}")

        results.append(row)

    accuracy = (correct / total * 100) if total else 0.0

    summary = {
        "timestamp": datetime.now().isoformat(),
        "dataset": run_name,
        "total_cases": total,
        "successful_cases": total - errors,
        "errors": errors,
        "correct_cases": correct,
        "routing_accuracy_percent": round(accuracy, 2),
        "fallback_count": fallback_count,
        "metric_definition": (
            "Selected tier compared with manually assigned expected tier. "
            "This does not measure answer quality or actual savings."
        ),
        "results": results,
    }

    with open(json_path, "w", encoding="utf-8") as file:
        json.dump(summary, file, indent=2, ensure_ascii=False)

    fieldnames = [
        "id",
        "query",
        "expected",
        "proposed",
        "selected",
        "confidence",
        "fallback_applied",
        "policy_reason",
        "correct",
        "error",
    ]

    with open(csv_path, "w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)

    print("\n" + "-" * 65)
    print(f"Dataset                : {run_name}")
    print(f"Total test cases       : {total}")
    print(f"Successful decisions   : {total - errors}")
    print(f"Errors                 : {errors}")
    print(f"Correct tier selections: {correct}")
    print(f"Routing accuracy       : {accuracy:.2f}%")
    print(f"Fallbacks applied      : {fallback_count}")
    print(f"JSON report            : {json_path}")
    print(f"CSV report             : {csv_path}")

    return summary


def run_evaluations():
    """Run the original and holdout datasets separately."""
    original_results = evaluate(TEST_CASES, "original")
    holdout_results = evaluate(HOLDOUT_CASES, "holdout")

    print("\n" + "=" * 65)
    print("FINAL COMPARISON")
    print("=" * 65)
    print(
        "Original dataset accuracy: "
        f"{original_results['routing_accuracy_percent']:.2f}%"
    )
    print(
        "Holdout dataset accuracy:  "
        f"{holdout_results['routing_accuracy_percent']:.2f}%"
    )
    print("\nNote:")
    print("- These metrics measure routing-tier agreement only.")
    print("- They do not measure answer quality.")
    print("- They do not prove actual AWS cost savings.")
    print("- Review the manually assigned expected tiers.")


if __name__ == "__main__":
    run_evaluations()
