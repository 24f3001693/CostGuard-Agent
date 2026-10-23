import os

USE_MOCK = os.getenv(
    "COSTGUARD_USE_MOCK",
    "true"
).lower() == "true"

# ============================================================
# STRANDS DECIDER
# ============================================================

DECIDER_MODEL = "StrandsAgents/strands-decider-2B-hobson-v19"

AWS_REGION = os.getenv("AWS_REGION", "us-east-1")

# Confidence thresholds.
# These are policy thresholds, not calibrated probabilities.
MIN_CONFIDENCE = 0.55
LOW_CONFIDENCE_THRESHOLD = 0.35
MEDIUM_CONFIDENCE_THRESHOLD = 0.55

# ============================================================
# AMAZON BEDROCK MODELS
# Costs are USD per 1,000 tokens.
# Verify current AWS rates before relying on estimates.
# ============================================================

MODELS = {
    "cheap": {
        "id": "amazon.nova-micro-v1:0",
        "name": "Amazon Nova Micro",
        "input_cost": 0.000035,
        "output_cost": 0.00014,
        "description": "Simple facts and basic calculations",
    },
    "medium": {
        "id": "amazon.nova-lite-v1:0",
        "name": "Amazon Nova Lite",
        "input_cost": 0.00006,
        "output_cost": 0.00024,
        "description": "Explanations and moderate reasoning",
    },
    "powerful": {
        "id": "amazon.nova-pro-v1:0",
        "name": "Amazon Nova Pro",
        "input_cost": 0.0008,
        "output_cost": 0.0032,
        "description": "Complex reasoning and difficult tasks",
    },
}

# Compare routing with using Nova Lite for every request.
BASELINE_MODEL = "medium"

# Used when actual provider token usage is unavailable.
ESTIMATED_INPUT_TOKENS = 150
ESTIMATED_OUTPUT_TOKENS = 200