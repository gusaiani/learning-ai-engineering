import json
from pathlib import Path
from openai import OpenAI
from dotenv import load_dotenv
from rich.console import Console
from rich.table import Table


load_dotenv(Path(__file__).parent.parent / ".env")
client = OpenAI()
console = Console()
model = "gpt-4o-mini"  # fast and cheap for testing


def run_prompt(system: str, user: str) -> str:
    """Call the API with a system prompt and user message, return text."""
    response = client.chat.completions.create(
        model=model,
        max_tokens=1024,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
    )

    return response.choices[0].message.content

def score(predicted: str, expected: str) -> bool:
    """Return True if predicted matches expected (case-insensitive strip)."""
    return predicted.strip().lower() == expected.strip().lower()

def run_harness(test_cases: list[dict], variants: dict[str, str]) -> None:
    """
    Run all variants against all test cases and print a comparison table.

    test_cases: [{"input": "...", "expected": "..."}]
    variants:   {"variant_name": "system_prompt_string"}
    """
    results = {name: [] for name in variants}
    for name, system in variants.items():
        for case in test_cases:
            output = run_prompt(system, case["input"])
            passed = score(output, case["expected"])
            results[name].append({"output": output, "pass": passed})

    table = Table()
    table.add_column("Input")
    for name in variants:
        table.add_column(name)

    for i, case in enumerate(test_cases):
        row = [case["input"]]
        for name in variants:
            r = results[name][i]
            cell = "✅" if r["pass"] else f"❌ {r['output']}"
            row.append(cell)
        table.add_row(*row)

    accuracy_row = ["Accuracy"]
    for name in variants:
        passed = sum(r["pass"] for r in results[name])
        total = len(results[name])
        accuracy_row.append(f"{passed}/{total}")
    table.add_row(*accuracy_row)

    console.print(table)


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
        "cot": """Classify the sentiment of the text. Reply with exactly one word: positive, neutral, or negative.

Think step by step before answering. First identify the emotional tone, then give your final answer.""",
    }

    run_harness(test_cases, variants)