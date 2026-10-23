
# decider.py

from strands_decider.infer import load_engine
from strands_decider.schema import ChoiceQuestion

from config import DECIDER_MODEL, MIN_CONFIDENCE


class CostGuardDecider:
    """
    CostGuard's decision layer.

    Selects a model tier using Strands Decider 2B and applies
    an explicit fallback policy when confidence is low.
    """

    VALID_TIERS = {"cheap", "medium", "powerful"}

    def __init__(self):
        print("Loading Strands Decider 2B...")

        self.engine = load_engine(
            DECIDER_MODEL,
            device="cpu",
        )

        print("Decider ready!\n")

    def choose_model(self, user_query: str) -> dict:
        """
        Select a model tier using Strands Decider 2B,
        followed by a transparent policy validation layer.
        """

        query = user_query.strip()

        if not query:
            raise ValueError("User query cannot be empty.")

        state = f"""
    User Query:
    {query}

    Choose the most cost-effective model tier that is suitable
    for the task.

    cheap:
    Greetings, basic facts, short answers, simple calculations.

    medium:
    Explanations, comparisons, and moderate reasoning.

    powerful:
    Complex coding, multi-step reasoning, and difficult analysis.
    """

        questions = {
            "model": ChoiceQuestion(
                instructions=(
                    "Which model tier is most appropriate "
                    "for this query?"
                ),
                criteria={
                    "cheap": (
                        "Simple greetings, basic facts, "
                        "or straightforward tasks."
                    ),
                    "medium": (
                        "Normal explanations and moderate reasoning."
                    ),
                    "powerful": (
                        "Complex coding and multi-step reasoning."
                    ),
                },
            )
        }

        result = self.engine.ask(state, questions)
        answer = result.answers["model"]

        proposed = answer.choice
        confidence = float(answer.confidence)

        probabilities = answer.probabilities

        if proposed not in self.VALID_TIERS:
            raise ValueError(
                f"Unknown tier returned by Decider: {proposed}"
            )

        if not 0.0 <= confidence <= 1.0:
            raise ValueError(
                f"Invalid confidence: {confidence}"
            )

        selected = proposed
        fallback_applied = False
        policy_reason = "No additional policy applied."

        normalized_query = query.lower()

        # --------------------------------------------------
        # Policy 1: Clear, simple greetings
        # --------------------------------------------------

        greetings = {
            "hi",
            "hii",
            "hello",
            "hey",
            "good morning",
            "good afternoon",
            "good evening",
        }

        if normalized_query in greetings:
            selected = "cheap"

            if selected != proposed:
                fallback_applied = True

            policy_reason = "Exact greeting matched."

        # --------------------------------------------------
        # Policy 2: Explicitly complex tasks
        # --------------------------------------------------

        complex_indicators = [
        "multi-step",
        "edge-case handling",
        "time and space complexity",
        "compare their trade-offs",
        "distributed application",
        "data-processing pipeline",
        "insert, search, and delete",
        "error recovery",
        "production outage",
        "multiple services",
        "intermittent production",
        "verification plan",
        "fault tolerance",
        "race conditions",
        "deadlocks",
        "graceful degradation",
        "dependency management",
        ]

        matched_indicators = [
            indicator
            for indicator in complex_indicators
            if indicator in normalized_query
        ]

        if matched_indicators:
            selected = "powerful"

            if selected != proposed:
                fallback_applied = True

            policy_reason = (
                "Complex-task policy matched: "
                + ", ".join(matched_indicators)
            )

        # --------------------------------------------------
        # Policy 3: Low-confidence cheap decisions
        # --------------------------------------------------

        elif confidence < MIN_CONFIDENCE and selected == "cheap":
            simple_factual_indicators = [
            "capital of",
            "what is the capital",
            "who is the president",
            "what is",
            "how many",
            "multiply",
            "multiplied by",
            "addition",
            "subtraction",
            ]

            query_lower = query.lower().strip()

            is_simple_factual = (
                len(query_lower.split()) <= 12
                and any(
                    indicator in query_lower
                    for indicator in simple_factual_indicators
                )
            )

            if is_simple_factual:
                policy_reason = (
                    "Simple factual query retained on cheap tier "
                    "despite low confidence."
                )
            else:
                selected = "medium"
                fallback_applied = True
                policy_reason = (
                    "Low-confidence cheap decision upgraded to medium."
                )


        reason = (
            f"Proposed={proposed}; final={selected}; "
            f"confidence={confidence:.2f}; {policy_reason}"
        )

        return {
            "selected": selected,
            "proposed": proposed,
            "confidence": confidence,
            "reason": reason,
            "fallback_applied": fallback_applied,
            "probabilities": probabilities,
            "policy_reason": policy_reason,
        }


