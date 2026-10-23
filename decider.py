"""
CostGuard Agent — Decision Engine

Strands Decider 2B proposes a model tier.
A deterministic, confidence-aware policy evaluates the proposal.

Decision pattern: Model Routing
"""

import re
from typing import Any

from strands_decider.infer import load_engine
from strands_decider.schema import ChoiceQuestion

from config import (
    DECIDER_MODEL,
    MIN_CONFIDENCE,
    LOW_CONFIDENCE_THRESHOLD,
    MEDIUM_CONFIDENCE_THRESHOLD,
)


class CostGuardDecider:
    VALID_TIERS = {"cheap", "medium", "powerful"}

    TIER_RANK = {
        "cheap": 0,
        "medium": 1,
        "powerful": 2,
    }

    GREETINGS = {
        "hi",
        "hii",
        "hello",
        "hey",
        "good morning",
        "good afternoon",
        "good evening",
        "good night",
    }

    # Phrases indicating potentially complex work.
    # These are policy signals, not a complete understanding of language.
    COMPLEX_PATTERNS = (
        r"\broot cause analysis\b",
        r"\btime and space complexity of (?:three|multiple|several)\b",
        r"\bcomplexity trade[- ]?offs?\b",
        r"\bcompare .* algorithms .*complexity trade[- ]?offs?\b",
        r"\bgraph traversal algorithms\b.*\bcomplexity trade[- ]?offs?\b",
        r"\bdistributed systems?\b",
        r"\bdistributed application\b",
        r"\bmultiple services\b",
        r"\bproduction outage\b",
        r"\bintermittent (?:production )?(?:failure|failures|error|errors|outage)\b",
        r"\brace conditions?\b",
        r"\bdeadlocks?\b",
        r"\bfault tolerance\b",
        r"\bgraceful degradation\b",
        r"\bsecurity threat model\b",
        r"\bconcurrent transactions\b",
        r"\barchitectural design\b",
        r"\bsystem design\b",
        r"\bdesign .* fault[- ]tolerant\b",
        r"\bscalable distributed\b",
        r"\bmultiple constraints\b",
        r"\bmulti[- ]step reasoning\b",
        r"\berror recovery\b",
        r"\bverification plan\b",
        r"\bdependency management\b",
        r"\bproduction[- ]ready\b",
        r"\bedge[- ]case handling\b",
        r"\bbinary search tree\b.*\b(edge[- ]case handling|edge cases|insert|search|delete)\b",
        r"\bdata[- ]processing pipeline\b",
        r"\bdistributed transaction\b",
        r"\bexactly[- ]once\b",
        r"\bidempotency\b",
    )

    # Moderate task indicators.
    MEDIUM_PATTERNS = (
        r"\bcompare\b",
        r"\bdifference between\b",
        r"\badvantages and disadvantages\b",
        r"\bpros and cons\b",
        r"\bexplain\b",
        r"\bsummarize\b",
        r"\bsummarise\b",
        r"\bhow does\b",
        r"\bhow do i\b",
        r"\bwrite (?:a |an )?(?:python )?(?:function|program|script)\b",
        r"\bdebug\b",
        r"\bimplement\b",
        r"\bsql\b",
        r"\bdatabase\b",
        r"\balgorithm\b",
        r"\bdata structure\b",
        r"\bhttp\b",
        r"\bauthentication\b",
    )

    # Short, straightforward factual requests.
    SIMPLE_PATTERNS = (
        r"\bcapital of\b",
        r"\bboiling point of\b",
        r"\bwhat is the value of\b",
        r"\bhow many\b",
        r"\bwhat does .* mean\b",
        r"\bdefine\b",
        r"\bmultiply\b",
        r"\bmultiplied by\b",
        r"\bcalculate\b",
        r"\bwhat is \d+\s*[\+\-\*/]\s*\d+\b",
    )

    def __init__(self):
        print("Loading Strands Decider 2B...")

        self.engine = load_engine(
            DECIDER_MODEL,
            device="cpu",
        )

        self.questions = {
            "model": ChoiceQuestion(
                instructions=(
                    "Select the least expensive model tier that can "
                    "reasonably handle the user's complete request. "
                    "Consider reasoning difficulty, number of constraints, "
                    "uncertainty, and the consequences of an inadequate "
                    "response. Do not classify by query length alone."
                ),
                criteria={
                    "cheap": (
                        "Simple greetings, straightforward facts, "
                        "basic arithmetic, and short direct answers."
                    ),
                    "medium": (
                        "Normal explanations, ordinary comparisons, "
                        "summaries, routine coding, and moderate reasoning."
                    ),
                    "powerful": (
                        "Difficult debugging, complex coding, algorithm "
                        "analysis, multiple interacting constraints, "
                        "system design, and multi-step reasoning."
                    ),
                },
            )
        }

        print("Decider ready!\n")

    @staticmethod
    def _normalize_query(user_query: str) -> str:
        if not isinstance(user_query, str):
            raise TypeError("User query must be a string.")

        query = user_query.strip()

        if not query:
            raise ValueError("User query cannot be empty.")

        return query

    @staticmethod
    def _matches(query: str, patterns: tuple[str, ...]) -> list[str]:
        matches = []

        for pattern in patterns:
            if re.search(pattern, query, flags=re.IGNORECASE):
                matches.append(pattern)

        return matches

    def _assess_complexity(self, query: str) -> dict[str, Any]:
        """
        Assess observable task signals.

        This is a deterministic heuristic, not a semantic model.
        The Decider's proposal remains the primary routing signal.
        """
        complex_matches = self._matches(
            query,
            self.COMPLEX_PATTERNS,
        )

        medium_matches = self._matches(
            query,
            self.MEDIUM_PATTERNS,
        )

        simple_matches = self._matches(
            query,
            self.SIMPLE_PATTERNS,
        )

        word_count = len(query.split())

        # Multiple explicit requirements may indicate additional work.
        action_count = len(
            re.findall(
                r"\b(?:design|implement|compare|analyze|analyse|debug|"
                r"evaluate|optimize|optimise|verify|build|explain)\b",
                query,
                flags=re.IGNORECASE,
            )
        )

        # Do not use word count alone to classify complexity.
        if complex_matches:
            level = "complex"
        elif medium_matches:
            level = "moderate"
        elif simple_matches and word_count <= 15:
            level = "simple"
        elif action_count >= 3:
            level = "complex"
        else:
            level = "uncertain"

        return {
            "level": level,
            "complex_matches": complex_matches,
            "medium_matches": medium_matches,
            "simple_matches": simple_matches,
            "word_count": word_count,
            "action_count": action_count,
        }

    def choose_model(self, user_query: str) -> dict[str, Any]:
        query = self._normalize_query(user_query)
        normalized = query.casefold()
        word_count = len(query.split())

        # Obtain the initial proposal from Strands Decider 2B.
        state = f"""
User request:
{query}

Choose the most cost-effective suitable tier.

cheap:
Simple facts, greetings, and basic calculations.

medium:
Explanations, ordinary comparisons, summaries, and moderate reasoning.

powerful:
Complex coding, difficult debugging, algorithm analysis,
system design, multiple interacting constraints, and multi-step reasoning.

Consider the complete task, not only individual keywords or query length.
"""

        result = self.engine.ask(state, self.questions)
        answer = result.answers["model"]

        proposed = answer.choice
        confidence = float(answer.confidence)
        probabilities = answer.probabilities

        if proposed not in self.VALID_TIERS:
            raise ValueError(
                f"Unknown tier returned by Decider: {proposed!r}"
            )

        if not 0.0 <= confidence <= 1.0:
            raise ValueError(
                f"Invalid confidence returned by Decider: {confidence}"
            )

        complexity = self._assess_complexity(normalized)

        selected = proposed
        policy_reasons = []

        # Policy 1: Exact greetings are always routed cheaply.
        if normalized in self.GREETINGS:
            selected = "cheap"
            policy_reasons.append("exact_greeting")
        else:
            # Policy 2: Strong, explicit complexity signals take priority.
            if complexity["level"] == "complex":
                selected = "powerful"
                policy_reasons.append("explicit_complexity_signal")

            # Policy 3: Short, clearly simple requests can use Cheap.
            elif (
                complexity["level"] == "simple"
                and word_count <= 15
                and proposed == "cheap"
            ):
                selected = "cheap"
                policy_reasons.append("short_simple_request")
            elif complexity["level"] == "simple" and proposed == "medium":
                selected = "medium"
                policy_reasons.append("medium_retained_for_explanatory_request")
            elif (
                complexity["level"] == "simple"
                and proposed == "powerful"
                and confidence < MEDIUM_CONFIDENCE_THRESHOLD
            ):
                selected = "medium"
                policy_reasons.append("low_confidence_powerful_downshifted_for_simple_request")

            # Policy 4: Moderate requests should not be promoted to Powerful
            # solely because the Decider has low confidence.
            elif complexity["level"] == "moderate":
                if proposed == "cheap":
                    selected = "medium"
                    policy_reasons.append("moderate_task_minimum_medium")
                elif proposed == "powerful" and confidence < MEDIUM_CONFIDENCE_THRESHOLD:
                    selected = "medium"
                    policy_reasons.append(
                        "low_confidence_powerful_downshifted_for_moderate_task"
                    )
                else:
                    selected = "medium"
                    policy_reasons.append("moderate_task_routed_to_medium")

            # Policy 5: For uncertain requests, avoid an automatic Powerful
            # upgrade based on low confidence alone. Keep a Powerful proposal
            # only when confidence is at least the configured threshold.
            elif complexity["level"] == "uncertain":
                if proposed == "cheap":
                    if confidence < LOW_CONFIDENCE_THRESHOLD and word_count > 15:
                        selected = "medium"
                        policy_reasons.append(
                            "low_confidence_cheap_upgraded_for_uncertain_request"
                        )
                    else:
                        selected = "cheap"
                        policy_reasons.append("uncertain_request_kept_cheap")
                elif proposed == "powerful" and confidence < MEDIUM_CONFIDENCE_THRESHOLD:
                    selected = "medium"
                    policy_reasons.append(
                        "low_confidence_powerful_downshifted_for_uncertain_request"
                    )
                else:
                    selected = proposed
                    policy_reasons.append("uncertain_request_kept_at_proposed_tier")

        fallback_applied = selected != proposed

        if not policy_reasons:
            policy_reasons.append("no_override_required")

        policy_reason = "; ".join(policy_reasons)

        reason = (
            f"proposed={proposed}; "
            f"selected={selected}; "
            f"confidence={confidence:.3f}; "
            f"complexity={complexity['level']}; "
            f"policy={policy_reason}"
        )

        return {
            "selected": selected,
            "proposed": proposed,
            "confidence": confidence,
            "reason": reason,
            "fallback_applied": fallback_applied,
            "probabilities": probabilities,
            "policy_reason": policy_reason,
            "complexity": complexity,
        }