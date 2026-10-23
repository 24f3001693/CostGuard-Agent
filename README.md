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
