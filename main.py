
"""
CostGuard Agent — Interactive CLI

Routes queries using Strands Decider 2B and generates responses
through Amazon Bedrock or an explicitly enabled mock implementation.

Decision pattern: Model Routing
"""

import json
import time
from datetime import datetime, timezone
from pathlib import Path

from decider import CostGuardDecider
from bedrock_client import BedrockClient
from config import (
    MODELS,
    BASELINE_MODEL,
    ESTIMATED_INPUT_TOKENS,
    ESTIMATED_OUTPUT_TOKENS,
)
from config import USE_MOCK


LOG_FILE = Path("decisions_log.json")


def load_logs():
    """Load existing JSON logs safely."""
    if not LOG_FILE.exists():
        return []

    try:
        with LOG_FILE.open("r", encoding="utf-8") as file:
            data = json.load(file)

        return data if isinstance(data, list) else []

    except (json.JSONDecodeError, OSError):
        print(
            "Warning: Existing decision log could not be read. "
            "A new log will be started."
        )
        return []


def save_log(entry):
    """Persist a decision record without crashing the session on I/O errors."""
    logs = load_logs()
    logs.append(entry)

    try:
        with LOG_FILE.open("w", encoding="utf-8") as file:
            json.dump(
                logs,
                file,
                indent=2,
                ensure_ascii=False,
            )
    except OSError as exc:
        print(f"Warning: Could not save decision log: {exc}")


def estimate_cost(
    model_type,
    input_tokens=ESTIMATED_INPUT_TOKENS,
    output_tokens=ESTIMATED_OUTPUT_TOKENS,
):
    """
    Estimate model inference cost using configured rates.

    This is an estimate, not an AWS invoice or guaranteed saving.
    The configured prices must match the selected model and AWS region.
    """
    if model_type not in MODELS:
        raise ValueError(f"Unknown model tier: {model_type}")

    if input_tokens < 0 or output_tokens < 0:
        raise ValueError("Token counts cannot be negative.")

    model = MODELS[model_type]

    input_cost = (
        input_tokens / 1000
    ) * model["input_cost"]

    output_cost = (
        output_tokens / 1000
    ) * model["output_cost"]

    return input_cost + output_cost


def print_banner():
    """Display application name and supported commands."""
    print("=" * 64)
    print("             CostGuard Agent")
    print("     Strands Decider 2B + Amazon Bedrock")
    print("=" * 64)
    print("Commands: stats | savings | help | exit")
    print(
        "\nNote: Cost and savings figures are estimates. "
        "Mock responses are not real Bedrock inference.\n"
    )


def show_help():
    """Display available commands."""
    print(
        "\nAvailable commands:\n"
        "  stats    - Show session statistics\n"
        "  savings  - Show estimated savings versus baseline\n"
        "  help     - Show these commands\n"
        "  exit     - End the session\n"
    )


def show_stats(
    model_usage,
    total_queries,
    failed_requests,
    estimated_cost,
    baseline_cost,
):
    """Print session metrics."""
    print("\n" + "-" * 50)
    print("SESSION STATS")
    print("-" * 50)

    for model_type, count in model_usage.items():
        percentage = (
            count / total_queries * 100
            if total_queries
            else 0.0
        )
        print(
            f"  {model_type:10} -> {count:3} "
            f"times ({percentage:.1f}%)"
        )

    print(f"\n  Successful responses     : {total_queries}")
    print(f"  Failed requests          : {failed_requests}")
    print(f"  Estimated routed cost    : ${estimated_cost:.8f}")
    print(f"  Estimated baseline cost  : ${baseline_cost:.8f}")

    estimated_savings = baseline_cost - estimated_cost

    savings_percentage = (
        estimated_savings / baseline_cost * 100
        if baseline_cost > 0
        else 0.0
    )

    print(
        f"  Estimated savings        : ${estimated_savings:.8f} "
        f"({savings_percentage:.1f}%)"
    )

    print("-" * 50 + "\n")


def create_log_entry(query, decision, response_mode, status, **extra):
    """Build a consistent, auditable decision log entry."""
    complexity = decision.get("complexity") or {}

    return {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "query": query,
        "proposed_model": decision.get("proposed"),
        "selected_model": decision.get("selected"),
        "confidence": decision.get("confidence"),
        "reason": decision.get("reason"),
        "policy_reason": decision.get("policy_reason"),
        "fallback_applied": decision.get("fallback_applied"),
        "probabilities": decision.get("probabilities"),

        # New routing-policy audit fields
        "complexity_level": complexity.get("level"),
        "complexity_signals": {
            "complex_matches": complexity.get("complex_matches", []),
            "medium_matches": complexity.get("medium_matches", []),
            "simple_matches": complexity.get("simple_matches", []),
            "word_count": complexity.get("word_count"),
            "action_count": complexity.get("action_count"),
        },

        "response_mode": response_mode,
        "status": status,
        **extra,
    }


def main():
    """Run the interactive CostGuard session."""
    print_banner()

    # Load the decision engine once.
    try:
        decider = CostGuardDecider()
    except Exception as exc:
        print(f"Could not initialize Strands Decider 2B: {exc}")
        print(
            "Check your environment, installed dependencies, "
            "model availability, and configuration."
        )
        return

    # Keep mock mode enabled until AWS Bedrock inference is verified.
    # Switch to use_mock=False only when real access works.
    bedrock = BedrockClient(use_mock=USE_MOCK)
    model_usage = {
        "cheap": 0,
        "medium": 0,
        "powerful": 0,
    }

    total_queries = 0
    failed_requests = 0
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

        command = query.casefold()

        if command in {"exit", "quit"}:
            break

        if command == "help":
            show_help()
            continue

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
                total_baseline_estimated_cost
                - total_estimated_cost
            )

            percentage = (
                estimated_savings / total_baseline_estimated_cost * 100
                if total_baseline_estimated_cost > 0
                else 0.0
            )

            print(
                f"\nEstimated savings versus baseline: "
                f"${estimated_savings:.8f} ({percentage:.1f}%)"
            )
            print(
                "Estimate only; this is not actual billed AWS savings.\n"
            )
            continue

        start_time = time.perf_counter()

        # Step 1: Select a model tier.
        try:
            decision = decider.choose_model(query)
        except Exception as exc:
            failed_requests += 1
            print(f"\nRouting decision failed: {exc}\n")

            save_log({
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "query": query,
                "status": "routing_error",
                "error": str(exc),
            })
            continue

        selected = decision["selected"]

        if selected not in model_usage:
            failed_requests += 1
            print(f"Error: Unsupported model tier: {selected}\n")
            continue

        # Step 2: Generate a response.
        try:
            result = bedrock.generate_result(
                query,
                model_type=selected,
            )
        except Exception as exc:
            result = {
                "text": "",
                "mode": "unknown",
                "usage": {},
                "error": str(exc),
            }

        elapsed = time.perf_counter() - start_time

        response_mode = result.get("mode", "unknown")
        usage = result.get("usage") or {}
        error = result.get("error")

        if error:
            failed_requests += 1

            print(
                f"-> Tier: {selected.upper()} | "
                f"Confidence: {decision['confidence']:.2f}"
            )
            print(f"Generation failed: {error}\n")

            save_log(
                create_log_entry(
                    query,
                    decision,
                    response_mode,
                    "generation_error",
                    error=str(error),
                    estimated_cost=None,
                    baseline_estimated_cost=None,
                    estimated_savings=None,
                    time_seconds=round(elapsed, 3),
                )
            )
            continue

        # Step 3: Use provider token usage when available.
        provider_input = usage.get("input_tokens")
        provider_output = usage.get("output_tokens")

        has_provider_usage = (
            response_mode == "bedrock"
            and isinstance(provider_input, int)
            and isinstance(provider_output, int)
            and not isinstance(provider_input, bool)
            and not isinstance(provider_output, bool)
        )

        if has_provider_usage:
            cost_input_tokens = provider_input
            cost_output_tokens = provider_output
            cost_basis = (
                "provider_reported_token_usage_with_configured_rates"
            )
        else:
            cost_input_tokens = ESTIMATED_INPUT_TOKENS
            cost_output_tokens = ESTIMATED_OUTPUT_TOKENS
            cost_basis = "assumed_token_counts_with_configured_rates"

        # Step 4: Calculate routed and baseline estimates.
        try:
            routed_cost = estimate_cost(
                selected,
                cost_input_tokens,
                cost_output_tokens,
            )

            baseline_cost = estimate_cost(
                BASELINE_MODEL,
                cost_input_tokens,
                cost_output_tokens,
            )
        except (KeyError, TypeError, ValueError) as exc:
            failed_requests += 1
            print(f"Cost calculation failed: {exc}\n")
            continue

        estimated_savings = baseline_cost - routed_cost

        # Step 5: Update session statistics.
        total_queries += 1
        model_usage[selected] += 1
        total_estimated_cost += routed_cost
        total_baseline_estimated_cost += baseline_cost

        # Step 6: Display results.
        print(
            f"-> Tier: {selected.upper()} | "
            f"Confidence: {decision['confidence']:.2f}"
        )
        print(
            f"   Estimated routed cost: ${routed_cost:.8f}"
        )
        print(
            f"   Estimated savings vs baseline: "
            f"${estimated_savings:.8f}"
        )

        if has_provider_usage:
            print(
                f"   Provider-reported tokens: "
                f"input={provider_input}, output={provider_output}"
            )
        else:
            print(
                "   Token usage: assumed demo values "
                f"(input={cost_input_tokens}, "
                f"output={cost_output_tokens})"
            )

        model_name = MODELS[selected].get("name", selected)

        print(
            f"\nAgent ({model_name}) [{response_mode}]:"
        )
        print(result.get("text") or "No response text returned.")
        print(f"\nElapsed time: {elapsed:.2f}s\n")

        # Step 7: Persist the decision and estimate.
        save_log(
            create_log_entry(
                query,
                decision,
                response_mode,
                "success",
                provider_usage_input_tokens=provider_input,
                provider_usage_output_tokens=provider_output,
                cost_input_tokens=cost_input_tokens,
                cost_output_tokens=cost_output_tokens,
                estimated_cost=routed_cost,
                baseline_model=BASELINE_MODEL,
                baseline_estimated_cost=baseline_cost,
                estimated_savings=estimated_savings,
                cost_basis=cost_basis,
                time_seconds=round(elapsed, 3),
            )
        )

    print("\n" + "=" * 64)
    show_stats(
        model_usage,
        total_queries,
        failed_requests,
        total_estimated_cost,
        total_baseline_estimated_cost,
    )
    print(f"Decision log: {LOG_FILE.resolve()}")
    print("Thank you for using CostGuard Agent!")


if __name__ == "__main__":
    main()
