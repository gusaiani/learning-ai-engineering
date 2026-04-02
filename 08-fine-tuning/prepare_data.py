# prepare_data.py
import json
import random
import os
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()
client = OpenAI()

CATEGORIES = ["billing", "bug_report", "feature_request", "how_to", "account_access"]
EXAMPLES_PER_CATEGORY = 40  # 200 total
SYSTEM_PROMPT = (
    "You are a tech support classifier. Given a user message, respond with JSON:\n"
    '{"category": "<one of: billing, bug_report, feature_request, how_to, account_access>",\n'
    ' "priority": "<low|medium|high>",\n'
    ' "response": "<a helpful 1-2 sentence reply>"}'
)


def generate_synthetic_tickets(category: str, count: int) -> list[dict]:
    """
    Use GPT-4o-mini to generate realistic support tickets for a given category.
    Returns a list of {"user_message": ..., "assistant_response": ...} dicts.
    """
    # TODO: prompt GPT-4o-mini to generate diverse, realistic tickets
    # Each ticket should have a natural user message and a structured JSON response
    pass


def build_jsonl_example(user_message: str, assistant_response: str) -> dict:
    """Convert a single example into OpenAI chat fine-tuning format."""
    # TODO: return {"messages": [system, user, assistant]}
    pass


def validate_dataset(examples: list[dict]) -> dict:
    """
    Validate the dataset:
    - Check JSONL format
    - Count tokens per example
    - Check category balance
    Returns stats dict.
    """
    # TODO: validate format, count tokens, check balance
    pass


def split_and_save(examples: list[dict], train_ratio: float = 0.8):
    """Shuffle, split into train/test, save as JSONL files."""
    # TODO: shuffle, split, write train.jsonl and test.jsonl
    pass


if __name__ == "__main__":
    print("Generating synthetic training data...")
    all_examples = []
    for category in CATEGORIES:
        print(f"  Generating {EXAMPLES_PER_CATEGORY} examples for '{category}'...")
        tickets = generate_synthetic_tickets(category, EXAMPLES_PER_CATEGORY)
        for ticket in tickets:
            example = build_jsonl_example(ticket["user_message"], ticket["assistant_response"])
            all_examples.append(example)

    print(f"\nValidating {len(all_examples)} examples...")
    stats = validate_dataset(all_examples)
    print(json.dumps(stats, indent=2))

    print("\nSplitting and saving...")
    split_and_save(all_examples)
    print("Done! Created train.jsonl and test.jsonl")
