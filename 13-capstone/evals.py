"""
Eval suite: retrieval quality, routing accuracy, response quality.

Usage:
    python evals.py                # Run all evals
    python evals.py retrieval      # Retrieval evals only
    python evals.py routing        # Routing evals only
    python evals.py response       # Response evals only
"""

import argparse
import json
import sys
import time

from pydantic import BaseModel

from config import openai_client, CHAT_MODEL
from knowledge import search as kb_search
from agents import route_query, run_support_agent, AgentEvent

# ---------------------------------------------------------------------------
# Eval case definitions
# ---------------------------------------------------------------------------

RETRIEVAL_CASES = [
    {
        "query": "How much does the Pro plan cost?",
        "expected_source": "pricing-and-plans.md",
        "expected_content": "49",
    },
    {
        "query": "What is the API rate limit?",
        "expected_source": "api-reference.md",
        "expected_content": "rate limit",
    },
    {
        "query": "How do I add team members?",
        "expected_source": "account-management.md",
        "expected_content": "invite",
    },
    {
        "query": "Can I export my data?",
        "expected_source": "account-management.md",
        "expected_content": "export",
    },
    {
        "query": "What authentication methods does the API support?",
        "expected_source": "api-reference.md",
        "expected_content": "api key",
    },
]

ROUTING_CASES = [
    {"message": "I want a refund for last month", "expected": "billing"},
    {"message": "The API returns 500 when I POST to /contacts", "expected": "technical"},
    {"message": "What time zone are your offices in?", "expected": "general"},
    {"message": "I'm going to sue you if this isn't fixed today", "expected": "escalation"},
    {"message": "How do I upgrade to Enterprise?", "expected": "billing"},
    {"message": "My webhook isn't receiving events", "expected": "technical"},
    {"message": "Do you have a referral program?", "expected": "general"},
    {"message": "Delete all my data immediately, GDPR request", "expected": "escalation"},
]

RESPONSE_CASES = [
    {
        "message": "How much does the Pro plan cost per month?",
        "reference": "The Pro plan costs $49 per month per user (or $39/month billed annually).",
        "grounding_doc": "pricing-and-plans.md",
    },
    {
        "message": "What's the API rate limit on the Pro plan?",
        "reference": "The Pro plan has a rate limit of 1,000 requests per minute with a burst of 50 per second.",
        "grounding_doc": "api-reference.md",
    },
    {
        "message": "How do I set up SSO?",
        "reference": "SSO is available on the Enterprise plan. Go to Settings > Security > SSO, select your provider (SAML or OIDC), and follow the configuration steps.",
        "grounding_doc": "account-management.md",
    },
]


# ---------------------------------------------------------------------------
# Eval scoring models (structured output for LLM-as-judge)
# ---------------------------------------------------------------------------

class ResponseScore(BaseModel):
    accuracy: int       # 0-5: how factually correct is the response?
    helpfulness: int    # 0-5: how helpful is the response to the customer?
    grounded: bool      # is it grounded in the knowledge base (not made up)?
    hallucination: bool # does it state facts not in the knowledge base?
    reasoning: str


# ---------------------------------------------------------------------------
# Eval functions
# ---------------------------------------------------------------------------

def eval_retrieval(query: str, expected_source: str, expected_content: str) -> dict:
    """Evaluate retrieval quality for a single query.

    Checks:
    1. Is the expected source file in the top results?
    2. Does any top result contain the expected content?

    Returns: {"pass": bool, "top_source": str|None, "expected_source": str, "details": str}
    """
    # TODO 1: Search with kb_search(query, top_k=5).
    #
    # Check if expected_source appears in any result's "source" field.
    # Check if expected_content (case-insensitive) appears in any result's "text".
    # A case passes if BOTH conditions are true.
    #
    # Return {"pass": bool, "top_source": results[0]["source"] if results else None,
    #         "expected_source": expected_source, "details": "...explanation..."}
    raise NotImplementedError


def eval_routing(message: str, expected_category: str) -> dict:
    """Evaluate routing accuracy for a single message.

    Returns: {"pass": bool, "predicted": str, "expected": str,
              "confidence": float, "reasoning": str}
    """
    # TODO 2: Call route_query(message).
    # Compare decision.category to expected_category.
    # Return pass/fail with details.
    raise NotImplementedError


def eval_response(message: str, reference: str, grounding_doc: str) -> dict:
    """Evaluate response quality using LLM-as-judge.

    Runs the full support agent, collects the response, then asks a judge LLM
    to score it against the reference answer.

    Returns: {"accuracy": int, "helpfulness": int, "grounded": bool,
              "hallucination": bool, "reasoning": str}
    """
    # TODO 3: Two steps.
    #
    # Step A — get the agent's response:
    #   Iterate over run_support_agent(message) events.
    #   Concatenate all "token" events into a response string.
    #
    # Step B — judge the response:
    #   Call openai_client.beta.chat.completions.parse() with:
    #     model=CHAT_MODEL
    #     messages=[
    #       {"role": "system", "content": judge_prompt},
    #       {"role": "user", "content": f"Customer: {message}\nAgent: {response}\nReference: {reference}\nSource doc: {grounding_doc}"},
    #     ]
    #     response_format=ResponseScore
    #
    #   The judge prompt should instruct the model to:
    #     - Score accuracy 0-5 (does the response match the reference facts?)
    #     - Score helpfulness 0-5 (would a customer find this useful?)
    #     - Assess grounding (is the info from the knowledge base?)
    #     - Detect hallucination (does it state things not in the docs?)
    #
    #   Return the ResponseScore fields as a dict.
    raise NotImplementedError


# ---------------------------------------------------------------------------
# Suite runner
# ---------------------------------------------------------------------------

def run_eval_suite(categories: list[str] | None = None) -> dict:
    """Run eval categories and print a report.

    Returns: {"retrieval": {...}, "routing": {...}, "response": {...}}
    """
    if categories is None:
        categories = ["retrieval", "routing", "response"]

    results = {}

    # TODO 4: For each category in categories, run all cases and collect results.
    #
    # Retrieval:
    #   Run eval_retrieval() for each case in RETRIEVAL_CASES.
    #   Print each result (pass/fail, query, details).
    #   Print summary: "RETRIEVAL: X/Y passed (Z%)"
    #
    # Routing:
    #   Run eval_routing() for each case in ROUTING_CASES.
    #   Print each result (pass/fail, message, predicted vs expected).
    #   Print summary: "ROUTING: X/Y passed (Z%)"
    #
    # Response:
    #   Run eval_response() for each case in RESPONSE_CASES.
    #   Print each result (scores, reasoning).
    #   Print summary: "RESPONSE: avg accuracy=X.X, avg helpfulness=X.X, hallucinations=N/M"
    #
    # Store all results in the results dict and return it.
    raise NotImplementedError


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="NovaCRM Support Agent — Eval Suite")
    parser.add_argument(
        "categories",
        nargs="*",
        default=["retrieval", "routing", "response"],
        choices=["retrieval", "routing", "response"],
    )
    args = parser.parse_args()

    print("=" * 60)
    print("  NovaCRM Support Agent — Eval Suite")
    print("=" * 60)
    print()

    results = run_eval_suite(args.categories)

    print()
    print("=" * 60)
    print("  Done.")
    print("=" * 60)


if __name__ == "__main__":
    main()
