# eval.py
import json
import os
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()
client = OpenAI()

SYSTEM_PROMPT = (
    "You are a tech support classifier. Given a user message, respond with JSON:\n"
    '{"category": "<one of: billing, bug_report, feature_request, how_to, account_access>",\n'
    ' "priority": "<low|medium|high>",\n'
    ' "response": "<a helpful 1-2 sentence reply>"}'
)


def load_test_set(filepath: str = "test.jsonl") -> list[dict]:
    """Load the test set from JSONL."""
    # TODO: read test.jsonl, extract user messages and expected outputs
    pass


def run_model(model: str, user_message: str, use_system_prompt: bool = True) -> dict:
    """
    Run a single test case through a model.
    Returns {"raw_output": ..., "parsed": ..., "tokens_used": ...}
    """
    # TODO: call the model, parse JSON from response, count tokens
    pass


def check_format_compliance(output: str) -> bool:
    """Check if the output is valid JSON with required fields."""
    # TODO: try json.loads, check for category, priority, response keys
    pass


def judge_quality(user_message: str, response: str) -> int:
    """Use GPT-4o as a judge to rate response quality 1-5."""
    # TODO: prompt GPT-4o to rate the response
    pass


def evaluate(test_set: list[dict], model: str, use_system_prompt: bool = True) -> dict:
    """
    Run full evaluation of a model on the test set.
    Returns metrics: accuracy, format_compliance, avg_quality, avg_tokens.
    """
    # TODO: run each test case, collect metrics, return summary
    pass


if __name__ == "__main__":
    # Load fine-tuned model ID
    with open("model_id.txt") as f:
        fine_tuned_model = f.read().strip()

    test_set = load_test_set()
    print(f"Loaded {len(test_set)} test examples\n")

    print("Evaluating base model (gpt-4o-mini)...")
    base_results = evaluate(test_set, "gpt-4o-mini", use_system_prompt=True)

    print(f"\nEvaluating fine-tuned model ({fine_tuned_model})...")
    ft_results = evaluate(test_set, fine_tuned_model, use_system_prompt=False)

    # Print comparison
    print("\n" + "=" * 60)
    print(f"{'Metric':<25} {'Base Model':<18} {'Fine-tuned':<18}")
    print("=" * 60)
    for metric in ["accuracy", "format_compliance", "avg_quality", "avg_tokens"]:
        base_val = base_results[metric]
        ft_val = ft_results[metric]
        print(f"{metric:<25} {base_val:<18.2f} {ft_val:<18.2f}")
    print("=" * 60)
