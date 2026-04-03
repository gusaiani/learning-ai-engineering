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
    prompt = (
        f"Generate {count} realistic tech support tickets for the category '{category}'.\n"
        "Each ticket should have a natural, varied user message and a structured JSON response.\n"
        "The JSON response must have: category, priority (low/medium/high), and response (1-2 sentences).\n\n"
        "Return a JSON array of objects with 'user_message' and 'assistant_response' keys.\n"
        "The assistant_response should be a JSON string matching the system prompt format.\n"
        "Make the user messages diverse — different tones, lengths, and specific details."
    )

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        response_format={"type": "json_object"},
    )

    raw = response.choices[0].message.content
    data = json.loads(raw)

    # The model may wrap the array in a key like "tickets"
    if isinstance(data, dict):
        data = list(data.values())[0]

    return data


def build_jsonl_example(user_message: str, assistant_response: str) -> dict:
    """Convert a single example into OpenAI chat fine-tuning format."""
    return {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_message},
            {"role": "assistant", "content": assistant_response}
        ]
    }


def validate_dataset(examples: list[dict]) -> dict:
    """
    Validate the dataset:
    - Check JSONL format
    - Count tokens per example
    - Check category balance
    Returns stats dict.
    """
    category_counts = {}
    format_errors = 0

    for ex in examples:
        msgs = ex.get("messages", [])

        # Check structure: must have exactly 3 messages (system, user, assistant)
        if len(msgs) != 3:
            format_errors += 1
            continue

        roles = [m["role"] for m in msgs]
        if roles != ["system", "user", "assistant"]:
            format_errors += 1
            continue

        # Extract category from assistant response
        try:
            parsed = json.loads(msgs[2]["content"])
            cat = parsed.get("category", "unknown")
            category_counts[cat] = category_counts.get(cat, 0) + 1
        except (json.JSONDecodeError, TypeError):
            format_errors += 1

    return {
        "total_examples": len(examples),
        "format_errors": format_errors,
        "category_distribution": category_counts,
    }


def split_and_save(examples: list[dict], train_ratio: float = 0.8):
    """Shuffle, split into train/test, save as JSONL files."""
    random.shuffle(examples)

    split_idx = int(len(examples) * train_ratio)
    train_set = examples[:split_idx]
    test_set = examples[split_idx:]

    for filename, dataset in [("train.jsonl", train_set), ("test.jsonl", test_set)]:
        with open(filename, "w") as f:
            for example in dataset:
                f.write(json.dumps(example) + "\n")

    print(f"  Train: {len(train_set)} examples → train.jsonl")
    print(f"  Test: {len(test_set)} examples → test.jsonl")


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
