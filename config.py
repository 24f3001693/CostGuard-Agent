
# config.py

import os


# ============================================================
# STRANDS DECIDER CONFIGURATION
# ============================================================

DECIDER_MODEL = "StrandsAgents/strands-decider-2B-hobson-v19"

# AWS configuration
AWS_REGION = os.getenv("AWS_REGION", "us-east-1")

# Confidence threshold
MIN_CONFIDENCE = 0.55


# ============================================================
# MODEL CONFIGURATION
# ============================================================
#
# IMPORTANT:
# Prices below are illustrative placeholders.
# Replace them with verified pricing for your AWS Region.
#
# Do not claim actual savings until real token usage and
# verified prices are used.
# ============================================================

MODELS = {
    "cheap": {
        "id": "amazon.nova-micro-v1:0",
        "name": "Nova Micro",
        "input_cost": 0.000035,
        "output_cost": 0.00014,
        "description": "Simple questions and short responses",
    },

    "medium": {
        "id": "amazon.nova-lite-v1:0",
        "name": "Nova Lite",
        "input_cost": 0.00006,
        "output_cost": 0.00024,
        "description": "Explanations and moderate reasoning",
    },

    "powerful": {
        # Configure a genuinely different, more capable model
        # after verifying its availability and pricing.
        #
        # Example:
        # set COSTGUARD_POWERFUL_MODEL=<verified-model-id>
        #
        # Until then, this tier is explicitly marked as
        # using the same model as the medium tier.

        "id": os.getenv(
            "COSTGUARD_POWERFUL_MODEL",
            "amazon.nova-lite-v1:0",
        ),
        "name": "Powerful Tier",
        "input_cost": 0.00006,
        "output_cost": 0.00024,
        "description": "Complex coding and multi-step reasoning",
    },
}


# ============================================================
# COST BASELINE
# ============================================================

# Compare routing against consistently using the medium tier.
# This can be changed to "powerful" when evaluating against
# a powerful-model baseline.

BASELINE_MODEL = "medium"


# ============================================================
# DEMO COST ESTIMATION
# ============================================================

# Used only when actual token usage is unavailable.
# These are assumptions, not measured token counts.

ESTIMATED_INPUT_TOKENS = 150
ESTIMATED_OUTPUT_TOKENS = 200

