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
    results = kb_search(query, top_k=5)

    source_match = any(r["source"] == expected_source for r in results)
    top_source = results[0]["source"] if results else None

    content_match = any(
        expected_content.lower() in r["text"].lower()
        for r in results
    )

    passed = source_match and content_match
    details = f"source_match={source_match}, content_match={content_match}"

    return {
        "pass": passed,
        "top_source": top_source,
        "expected_source": expected_source,
        "details": details,
    }

def eval_routing(message: str, expected_category: str) -> dict:
    """Evaluate routing accuracy for a single message.

    Returns: {"pass": bool, "predicted": str, "expected": str,
              "confidence": float, "reasoning": str}
    """
    decision = route_query(message)
    passed = decision.category == expected_category

    return {
        "pass": passed,
        "predicted": decision.category,
        "expected": expected_category,
        "confidence": decision.confidence,
        "reasoning": decision.reasoning,
    }

def eval_response(message: str, reference: str, grounding_doc: str) -> dict:
    """Evaluate response quality using LLM-as-judge.

    Runs the full support agent, collects the response, then asks a judge LLM
    to score it against the reference answer.

    Returns: {"accuracy": int, "helpfulness": int, "grounded": bool,
              "hallucination": bool, "reasoning": str}
    """
    response_parts = []
    for event in run_support_agent(message):
        if event.type == "token":
            response_parts.append(event.data["content"])

    response = "".join(response_parts)

    judge_prompt = (
        "You are evaluating an AI support agent's response to a customer message.\n"
        "Score the response on these dimensions:\n"
        "- accuracy (0-5): does it match the reference facts?\n"
        "- helpfulness (0-5): would a customer find it useful?\n"
        "- grounded (true/false): is all info traceable to the knowledge base?\n"
        "- hallucination (true/false): does it state facts not in the docs?\n"
        "Be strict. A response missing key info from the reference is not accurate."
    )

    judge_input = (
        f"Customer: {message}\n"
        f"Agent: {response}\n"
        f"Reference: {reference}\n"
        f"Source doc: {grounding_doc}"
    )

    judge_response = openai_client.beta.chat.completions.parse(
        model=CHAT_MODEL,
        messages=[
            {"role": "system", "content": judge_prompt},
            {"role": "user", "content": judge_input},
        ],
        response_format=ResponseScore,
    )
    score = judge_response.choices[0].message.parsed

    return {
        "accuracy": score.accuracy,
        "helpfulness": score.helpfulness,
        "grounded": score.grounded,
        "hallucination": score.hallucination,
        "reasoning": score.reasoning,
    }
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

    if "retrieval" in categories:
        print("RETRIEVAL")
        print("-" * 60)
        retrieval_results = []
        for case in RETRIEVAL_CASES:
            r = eval_retrieval(case["query"], case["expected_source"], case["expected_content"])
            retrieval_results.append(r)
            mark = "PASS" if r["pass"] else "FAIL"
            print(f"  [{mark}] {case['query']}")
            print(f"         {r['details']}")

        passed = sum(1 for r in retrieval_results if r["pass"])
        total = len(retrieval_results)
        pct = 100 * passed / total if total else 0
        print(f"RETRIEVAL: {passed}/{total} passed ({pct:.0f}%)")
        print()
        results["retrieval"] = retrieval_results

    if "routing" in categories:
        print("ROUTING")
        print("-" * 60)
        routing_results = []
        for case in ROUTING_CASES:
            r = eval_routing(case["message"], case["expected"])
            routing_results.append(r)
            mark = "PASS" if r["pass"] else "FAIL"
            print(f"  [{mark}] {case['message']}")
            print(f"         predicted={r['predicted']} expected={r['expected']} (conf={r['confidence']:.2f})")

        passed = sum(1 for r in routing_results if r["pass"])
        total = len(routing_results)
        pct = 100 * passed / total if total else 0
        print(f"ROUTING: {passed}/{total} passed ({pct:.0f}%)")
        print()
        results["routing"] = routing_results

    if "response" in categories:
        print("RESPONSE")
        print("-" * 60)
        response_results = []
        for case in RESPONSE_CASES:
            r = eval_response(case["message"], case["reference"], case["grounding_doc"])
            response_results.append(r)
            print(f"  {case['message']}")
            print(f"    accuracy={r['accuracy']}/5 helpfulness={r['helpfulness']}/5  " f"grounded={r['grounded']} hallucination={r['hallucination']}")
            print(f"    {r['reasoning']}")

        total = len(response_results)
        avg_acc = sum(r["accuracy"] for r in response_results) / total if total else 0
        avg_help = sum(r["helpfulness"] for r in response_results) / total if total else 0
        halluc = sum(1 for r in response_results if r["hallucination"])
        print(f"RESPONSE: avg accuracy={avg_acc:.1f}  avg helpfulness={avg_help:.1f}  " f"hallucinations={halluc}/{total}")
        print()
        results["response"] = response_results

    return results



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
