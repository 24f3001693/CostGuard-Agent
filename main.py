import json
import time
from datetime import datetime
from pathlib import Path

from decider import CostGuardDecider
from bedrock_client import BedrockClient
from config import (
    MODELS,
    BASELINE_MODEL,
    ESTIMATED_INPUT_TOKENS,
    ESTIMATED_OUTPUT_TOKENS,
)

LOG_FILE = Path("decisions_log.json")


def load_logs():
    """Load existing decision logs, returning an empty list if unavailable."""
    if not LOG_FILE.exists():
        return []

    try:
        with LOG_FILE.open("r", encoding="utf-8") as file:
            data = json.load(file)
            return data if isinstance(data, list) else []
    except (json.JSONDecodeError, OSError):
        print("Warning: Could not read existing logs; starting a new log list.")
        return []


def save_log(entry):
    """Append one entry to the JSON log file."""
    logs = load_logs()
    logs.append(entry)

    with LOG_FILE.open("w", encoding="utf-8") as file:
        json.dump(logs, file, indent=2, ensure_ascii=False)


def estimate_cost(
    model_type,
    input_tokens=ESTIMATED_INPUT_TOKENS,
    output_tokens=ESTIMATED_OUTPUT_TOKENS,
):
    """
    Estimate inference cost using configured prices and token counts.
    This is not an actual AWS bill or a guarantee of provider pricing.
    """
    if model_type not in MODELS:
        raise ValueError(f"Unknown model tier: {model_type}")

    model = MODELS[model_type]
    input_cost = (input_tokens / 1000) * model["input_cost"]
    output_cost = (output_tokens / 1000) * model["output_cost"]
    return input_cost + output_cost


def print_banner():
    print("=" * 60)
    print("          CostGuard Agent - Smart Model Router")
    print("     Powered by Strands Decider 2B + Amazon Bedrock")
    print("=" * 60)
    print("Commands: type question | 'stats' | 'savings' | 'exit'\n")


def show_stats(model_usage, total_queries, failed_requests,
               estimated_cost, baseline_cost):
    print("\n" + "-" * 45)
    print("SESSION STATS")
    print("-" * 45)

    for model_type, count in model_usage.items():
        percentage = count / total_queries * 100 if total_queries > 0 else 0
        print(f"  {model_type:10} -> {count:3} times ({percentage:.1f}%)")

    print(f"\n  Successful responses       : {total_queries}")
    print(f"  Failed requests            : {failed_requests}")
    print(f"  Estimated routed cost      : ${estimated_cost:.8f}")
    print(f"  Estimated baseline cost    : ${baseline_cost:.8f}")

    estimated_savings = baseline_cost - estimated_cost
    if baseline_cost > 0:
        savings_percentage = estimated_savings / baseline_cost * 100
        print(
            f"  Estimated savings          : ${estimated_savings:.8f} "
            f"({savings_percentage:.1f}%)"
        )
    else:
        print("  Estimated savings          : $0.00000000 (baseline cost is zero)")

    print("-" * 45 + "\n")


def main():
    print_banner()
    decider = CostGuardDecider()

    # Keep mock mode enabled until AWS Bedrock access is verified.
    # Change to use_mock=False only after access and model permissions work.
    bedrock = BedrockClient(use_mock=True)

    total_queries = 0
    failed_requests = 0
    model_usage = {"cheap": 0, "medium": 0, "powerful": 0}
    total_estimated_cost = 0.0
    total_baseline_estimated_cost = 0.0

    while True:
        try:
            query = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nInput closed. Finishing session...")
            break

        if not query:
            continue

        command = query.lower()
        if command in {"exit", "quit"}:
            break
        if command == "stats":
            show_stats(
                model_usage,
                total_queries,
                failed_requests,
                total_estimated_cost,
                total_baseline_estimated_cost,
            )
            continue
        if command == "savings":
            estimated_savings = (
                total_baseline_estimated_cost - total_estimated_cost
            )
            percentage = (
                estimated_savings / total_baseline_estimated_cost * 100
                if total_baseline_estimated_cost > 0
                else 0.0
            )
            print(
                f"\nEstimated savings this session: "
                f"${estimated_savings:.8f} ({percentage:.1f}%)\n"
            )
            print(
                "These are estimates based on configured prices and token counts, "
                "not actual billed savings."
            )
            continue

        start = time.time()

        # 1. Choose a model tier.
        decision = decider.choose_model(query)
        selected = decision["selected"]

        if selected not in model_usage:
            print(f"Error: Decider returned an unknown tier: {selected}")
            continue

        # 2. Generate a response and collect provider usage when available.
        result = bedrock.generate_result(query, model_type=selected)
        elapsed = time.time() - start
        response_mode = result.get("mode", "unknown")
        usage = result.get("usage") or {}
        error = result.get("error")

        if error:
            failed_requests += 1
            print(f"-> Decider: {selected.upper()} | "
                  f"Confidence: {decision['confidence']:.2f}")
            print(f"\nGeneration failed: {error}\n")

            save_log({
                "timestamp": datetime.now().isoformat(),
                "query": query,
                "proposed_model": decision["proposed"],
                "selected_model": selected,
                "confidence": decision["confidence"],
                "reason": decision["reason"],
                "policy_reason": decision.get("policy_reason"),
                "fallback_applied": decision["fallback_applied"],
                "probabilities": decision["probabilities"],
                "response_mode": response_mode,
                "status": "error",
                "error": str(error),
                "estimated_cost": None,
                "baseline_estimated_cost": None,
                "estimated_savings": None,
                "cost_basis": "not_estimated_due_to_generation_error",
                "time_seconds": round(elapsed, 2),
            })
            continue

        # Provider-reported token usage is used when present. Mock mode and
        # missing usage metadata use configured assumed token counts instead.
        provider_input = usage.get("input_tokens")
        provider_output = usage.get("output_tokens")
        has_provider_usage = (
            isinstance(provider_input, int)
            and isinstance(provider_output, int)
            and response_mode == "bedrock"
        )

        if has_provider_usage:
            cost_input_tokens = provider_input
            cost_output_tokens = provider_output
            cost_basis = (
                "provider_reported_routed_usage_with_configured_prices; "
                "baseline applies the same token counts as an estimate"
            )
        else:
            cost_input_tokens = ESTIMATED_INPUT_TOKENS
            cost_output_tokens = ESTIMATED_OUTPUT_TOKENS
            cost_basis = "assumed_token_counts_with_configured_prices"

        routed_estimated_cost = estimate_cost(
            selected, cost_input_tokens, cost_output_tokens
        )
        baseline_estimated_cost = estimate_cost(
            BASELINE_MODEL, cost_input_tokens, cost_output_tokens
        )
        estimated_savings = baseline_estimated_cost - routed_estimated_cost

        total_queries += 1
        model_usage[selected] += 1
        total_estimated_cost += routed_estimated_cost
        total_baseline_estimated_cost += baseline_estimated_cost

        print(
            f"-> Decider: {selected.upper()} | "
            f"Confidence: {decision['confidence']:.2f}"
        )
        print(
            f"   Estimated cost: ${routed_estimated_cost:.8f} | "
            f"Estimated savings vs baseline: ${estimated_savings:.8f}"
        )
        if has_provider_usage:
            print(
                f"   Provider-reported tokens: input={provider_input}, "
                f"output={provider_output}"
            )
        else:
            print(
                "   Token usage: assumed demo values "
                f"(input={cost_input_tokens}, output={cost_output_tokens})"
            )

        print(f"\nAgent ({MODELS[selected]['name']}) [{response_mode}]:")
        print(result.get("text", "No response text returned."))
        print(f"\n(Time: {elapsed:.1f}s)\n")

        # 3. Persist the decision, usage metadata, and clearly labelled estimates.
        save_log({
            "timestamp": datetime.now().isoformat(),
            "query": query,
            "proposed_model": decision["proposed"],
            "selected_model": selected,
            "confidence": decision["confidence"],
            "reason": decision["reason"],
            "policy_reason": decision.get("policy_reason"),
            "fallback_applied": decision["fallback_applied"],
            "probabilities": decision["probabilities"],
            "response_mode": response_mode,
            "status": "success",
            "provider_usage_input_tokens": provider_input,
            "provider_usage_output_tokens": provider_output,
            "cost_input_tokens": cost_input_tokens,
            "cost_output_tokens": cost_output_tokens,
            "estimated_cost": routed_estimated_cost,
            "baseline_model": BASELINE_MODEL,
            "baseline_estimated_cost": baseline_estimated_cost,
            "estimated_savings": estimated_savings,
            "cost_basis": cost_basis,
            "time_seconds": round(elapsed, 2),
        })

    print("\n" + "=" * 60)
    show_stats(
        model_usage,
        total_queries,
        failed_requests,
        total_estimated_cost,
        total_baseline_estimated_cost,
    )
    print("Thank you for using CostGuard Agent!")
    print(f"Full log saved in: {LOG_FILE}")


if __name__ == "__main__":
    main()
