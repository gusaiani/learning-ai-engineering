import json
from anthropic import Anthropic
from dotenv import load_dotenv

load_dotenv()
client = Anthropic()
MODEL = "claude-haiku-4-5-20251001"  # fast and cheap for testing


def run_prompt(system: str, user: str) -> str:
    """Call the API with a system prompt and user message, return text."""
    # TODO: implement
    pass


def score(predicted: str, expected: str) -> bool:
    """Return True if predicted matches expected (case-insensitive strip)."""
    # TODO: implement
    pass


def run_harness(test_cases: list[dict], variants: dict[str, str]) -> None:
    """
    Run all variants against all test cases and print a comparison table.

    test_cases: [{"input": "...", "expected": "..."}]
    variants:   {"variant_name": "system_prompt_string"}
    """
    # TODO: implement
    # Hint: collect results in a dict[variant_name][i] = {"output": ..., "pass": bool}
    # Then use rich.table.Table to print
    pass


if __name__ == "__main__":
    # Define your test cases inline or load from tests.json
    test_cases = [
        {"input": "Love this product, works perfectly!", "expected": "positive"},
        {"input": "It's okay, nothing special.",          "expected": "neutral"},
        {"input": "Broke after two days. Very disappointed.", "expected": "negative"},
        {"input": "Fast shipping, great packaging.",      "expected": "positive"},
        {"input": "Not what I expected but usable.",      "expected": "neutral"},
        {"input": "Worst purchase I've ever made.",       "expected": "negative"},
    ]

    variants = {
        "zero-shot": "Classify the sentiment of the text. Reply with exactly one word: positive, neutral, or negative.",
        "few-shot": """Classify the sentiment of the text. Reply with exactly one word: positive, neutral, or negative.

Examples:
Text: "Amazing quality!" → positive
Text: "Does the job." → neutral
Text: "Total waste of money." → negative""",
        # TODO: add a third variant using chain-of-thought
    }

    run_harness(test_cases, variants)
