# CostGuard Agent

**Intelligent Model Routing using Strands Decider 2B + Amazon Bedrock**

CostGuard Agent automatically selects the most cost-effective foundation model for every user query.  
It uses **Strands Decider 2B** (a specialized decision model) to classify query complexity and routes the request to the cheapest suitable model on Amazon Bedrock.

## Problem

Most applications send every request to an expensive frontier model.  
This wastes money on simple queries (greetings, factual questions, short answers).

## Solution

CostGuard Agent uses Strands Decider 2B to classify each query into three tiers:

| Tier       | Model          | Use Case                          |
|------------|----------------|-----------------------------------|
| cheap      | Nova Micro     | Greetings, simple facts           |
| medium     | Nova Lite      | Normal questions & explanations   |
| powerful   | Configurable model; currently defaults to Nova Lite | Coding, complex reasoning |

## Key Features

- **Smart Routing** powered by Strands Decider 2B
- **Confidence-aware** decisions with fallback
- **Estimated Cost Comparison** against a configurable baseline; estimates are not actual AWS billing savings.
- **Decision Logging** (JSON log of every choice)
- **Session Stats** (`stats` command)
- Works in **Mock mode** when Bedrock access is pending

## Architecture


                    User Query
                        |
                        v
                Strands Decider 2B
                        |
                        v
                Routing Decision
            (Tier + Confidence Score)
                        |
                        v
                Confidence & Policy
                    Evaluation
                        |
            +----------+----------+
            |          |          |
            v          v          v
            Cheap      Medium    Powerful
        Nova Micro   Nova Lite  Configured
            |          |          |
            +----------+----------+
                        |
                        v
                Selected Model
                        |
                        v
                    Agent Response
                        |
                        v
            Decision Log & Cost Estimate


### Decision Flow

1. Receive a user query.
2. Use Strands Decider 2B to evaluate query complexity and propose a routing tier.
3. Apply the confidence threshold and routing policies.
4. Select the configured model for the resulting tier.
5. Generate a response using Amazon Bedrock when real inference is enabled, or use the clearly labelled mock implementation during development.
6. Log the decision and calculate an estimated cost using configured model prices and token assumptions.

**Current limitation:** Real Amazon Bedrock inference has not yet been verified because the account's model invocation is returning `ValidationException: Operation not allowed`. Mock-mode results do not demonstrate successful Bedrock inference.

## Installation and Setup

### Prerequisites

* Python 3.10 or a compatible version supported by the project's dependencies.
* Git.
* AWS credentials and Amazon Bedrock model access for real inference.
* Internet access for installing dependencies and downloading the Strands Decider 2B model when required.

### 1. Clone the Repository

git clone https://github.com/24f3001693/CostGuard-Agent.git
cd CostGuard-Agent

### 2. Create a Virtual Environment

**Windows PowerShell:**

python -m venv venv
.\venv\Scripts\Activate.ps1

**Linux/macOS:**

python3 -m venv venv
source venv/bin/activate

### 3. Install Dependencies

python -m pip install --upgrade pip
pip install -r requirements.txt

Follow the AWS CLI and Amazon Bedrock documentation to configure credentials and select a region supported by the configured models.

## How to Run

### Run in Mock Mode

Mock mode is useful for development and demonstrations when Amazon Bedrock inference is unavailable.

**Windows PowerShell:**

$env:COSTGUARD_USE_MOCK="true"
python main.py

Mock responses are simulated and must not be interpreted as real Amazon Bedrock inference results.

### Run with Amazon Bedrock

Set the environment variable to disable mock mode:

$env:COSTGUARD_USE_MOCK="false"
$env:AWS_REGION="us-east-1"
python main.py

Configure valid AWS credentials and ensure the account is authorized to invoke the selected models.

**Important:** Real Amazon Bedrock inference has not yet been verified for this project. The current environment returns `ValidationException: Operation not allowed`. Successful execution in mock mode does not establish successful Bedrock integration.

## Evaluation Results

CostGuard includes an evaluation script for testing routing decisions against labelled query datasets.

Run:

python evaluate.py

The evaluation reports routing accuracy, per-tier precision and recall, fallback counts, a confusion matrix, and estimated model costs.

### Current Local Evaluation

| Metric                            | Original Dataset | Holdout Dataset |
| --------------------------------- | ---------------: | --------------: |
| Test cases                        |               12 |              12 |
| Correct routing selections        |               12 |              12 |
| Routing accuracy                  |             100% |            100% |
| Errors                            |                0 |               0 |
| Fallback count                    |                2 |               1 |
| Estimated routed cost             |      $0.00340100 |     $0.00485450 |
| Estimated Nova Lite baseline cost |      $0.00068400 |     $0.00068400 |

These results describe agreement with the expected routing tiers for the current labelled datasets. They do not measure generated-answer quality.

The estimated costs are based on configured prices and token assumptions. They are not actual AWS charges or verified cost savings. In these test runs, the current routing strategy costs more than the configured Nova Lite baseline.

## Limitations

* Real Amazon Bedrock inference has not yet been verified because model invocation currently returns `ValidationException: Operation not allowed`.
* Mock mode simulates responses and cannot prove that live model invocation works.
* Routing accuracy measures agreement with labelled model tiers, not the correctness or quality of generated answers.
* Cost estimates depend on configured model prices and token assumptions. Actual costs may differ.
* The current evaluation datasets are small and do not establish performance across all possible queries.
* Actual cost savings and production readiness require further validation with authorized Bedrock inference, representative workloads, and measured token usage.

