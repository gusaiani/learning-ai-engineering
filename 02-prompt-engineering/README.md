# Module 02 — Prompt Engineering

**Goal:** Learn how to write reliable, high-quality prompts — and build a harness to test them systematically.

**Time:** ~1 day

---

## Setup & running

```bash
pip install anthropic python-dotenv rich

# Run the harness
python harness.py
```

---

## What you'll learn

- Chain-of-thought (CoT) prompting and when it helps
- Few-shot prompting: how examples shape model behavior
- Output formatting: getting consistent structure from prose models
- Prompt reliability: same prompt, different runs — how stable is it?
- How to compare prompt variants side-by-side systematically

---

## Concepts

### Chain-of-thought prompting

Telling the model to reason step-by-step before answering dramatically improves accuracy on multi-step tasks.

```python
# Without CoT
"Is 17 a prime number? Answer yes or no."

# With CoT
"Is 17 a prime number? Think step by step, then answer yes or no."
```

Add `"Let's think step by step."` as a suffix or instruct it in the system prompt. For hard tasks, CoT is almost always worth it.

### Few-shot prompting

Examples in the prompt teach the model the format, tone, and reasoning pattern you want.

```python
system = """Classify customer feedback as positive, neutral, or negative.

Examples:
Input: "Love this product, works perfectly!"
Output: positive

Input: "It's okay, nothing special."
Output: neutral

Input: "Broke after two days. Very disappointed."
Output: negative"""
```

Rule of thumb: 3–5 examples cover most cases. More is only needed if the task is very nuanced.

### Output formatting

LLMs are flexible but inconsistent. Force structure:

```python
# Unreliable
"List three pros and cons of remote work."

# Reliable
"""List three pros and cons of remote work.
Respond in this exact JSON format:
{"pros": ["...", "...", "..."], "cons": ["...", "...", "..."]}"""
```

XML tags also work well and are what Claude is natively trained on:
```
<pros>...</pros>
<cons>...</cons>
```

### Prompt reliability

A good prompt should produce consistent outputs across multiple runs — especially for classification and extraction tasks. Temperature=0 helps, but doesn't guarantee it. Test it.

---

## Project: Prompt Testing Harness

Build a CLI tool that:

1. Defines a set of **test cases** (input + expected output or evaluation criteria)
2. Runs **multiple prompt variants** against all test cases
3. **Scores** each result (exact match for classification, LLM-as-judge for open-ended)
4. Prints a **comparison table** showing which prompt performed best

### Example use case

You're building a sentiment classifier. You have three prompt variants:

- **v1:** Zero-shot — just ask
- **v2:** Few-shot — add 3 examples
- **v3:** CoT + few-shot — add examples and ask it to reason first

Run all three against 10 labeled test cases. See which wins.

### Requirements

```
- Load test cases from a JSON file (input + expected_output)
- Define 2–3 prompt variants in code
- Run each variant against every test case
- Score: exact match (for classification), or keyword match
- Print a rich table: rows = test cases, columns = prompt variants
- Show pass/fail per cell and overall accuracy per variant
```

### Starter code

```python
# harness.py
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
```

### Your task

1. Implement `run_prompt()` — call the API, pass the user input as the user message
2. Implement `score()` — compare predicted vs expected
3. Implement `run_harness()` — loop over variants × test cases, collect results, print a table
4. Add a third CoT variant and see if it improves accuracy
5. Add an accuracy row at the bottom of the table

### Hint — rich table

```python
from rich.table import Table
from rich.console import Console

console = Console()
table = Table(title="Prompt Comparison")
table.add_column("Input", style="dim")
table.add_column("Expected")
# add one column per variant...
table.add_row("some input", "positive", "✅ positive", "❌ negative")
console.print(table)
```

---

## Stretch goals

- Load test cases from `tests.json` instead of hardcoding them
- Add LLM-as-judge scoring for open-ended outputs (not just exact match)
- Run each variant N times and report consistency (% same answer across runs)
- Add a `/export` that saves results to CSV

---

## Key questions to answer before moving on

1. When does chain-of-thought hurt instead of help?
2. How many few-shot examples is too many?
3. Why is exact-match a poor metric for most real-world tasks?
4. What's the difference between prompt brittleness and model randomness?

---

## Resources

- [Anthropic prompt engineering guide](https://docs.anthropic.com/en/docs/build-with-claude/prompt-engineering/overview)
- [Claude: be clear and direct](https://docs.anthropic.com/en/docs/build-with-claude/prompt-engineering/be-clear-and-direct)
- [Chain-of-thought prompting paper](https://arxiv.org/abs/2201.11903)

---

**When done:** Mark Module 02 as shipped in the root README, commit, and move to [Module 03](../03-embeddings-semantic-search/).
