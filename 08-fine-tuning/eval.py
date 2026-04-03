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
    examples = []
    with open(filepath) as f:
        for line in f:
            data = json.loads(line)
            msgs = data["messages"]
            examples.append({
                "user_message": msgs[1]["content"],
                "expected": json.loads(msgs[2]["content"])
            })
    return examples

def run_model(model: str, user_message: str, use_system_prompt: bool = True) -> dict:
    """
    Run a single test case through a model.
    Returns {"raw_output": ..., "parsed": ..., "tokens_used": ...}
    """
    messages = []
    if use_system_prompt:
        messages.append({"role": "system", "content": SYSTEM_PROMPT})
    messages.append({"role": "user", "content": user_message})

    response = client.chat.completions.create(
        model=model,
        messages=messages,
    )

    raw = response.choices[0].message.content
    tokens = response.usage.total_tokens

    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        parsed = None

    return {"raw_output": raw, "parsed": parsed, "tokens_used": tokens}

def check_format_compliance(output: str) -> bool:
    """Check if the output is valid JSON with required fields."""
    try:
        data = json.loads(output)
    except (json.JSONDecodeError, TypeError):
        return False

    required_keys = {"category", "priority", "response"}
    return required_keys.issubset(data.keys())

def judge_quality(user_message: str, response: str) -> int:
    """Use GPT-4o as a judge to rate response quality 1-5."""
    prompt = (
        "Rate this tech support response on a scale of 1-5.\n"
        "1 = wrong or unhelpful, 5 = perfect.\n"
        "Reply with ONLY the number.\n\n"
        f"User message: {user_message}\n"
        f"Response: {response}"
    )

    result = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=1,
    )

    try:
        return int(result.choices[0].message.content.strip())
    except ValueError:
        return 3

def evaluate(test_set: list[dict], model: str, use_system_prompt: bool = True) -> dict:
    """
    Run full evaluation of a model on the test set.
    Returns metrics: accuracy, format_compliance, avg_quality, avg_tokens.
    """
    correct = 0
    compliant = 0
    total_quality = 0
    total_tokens = 0

    for i, example in enumerate(test_set):
        print(f"  [{i + 1}/{len(test_set)}]", end="\r")

        result = run_model(model, example["user_message"], use_system_prompt)

        if check_format_compliance(result["raw_output"]):
            compliant += 1

        if result["parsed"] and result["parsed"].get("category") == example["expected"]["category"]:
            correct += 1

        quality_text = result["raw_output"] if not result["parsed"] else result["parsed"].get("response", "")
        total_quality += judge_quality(example["user_message"], quality_text)
        total_tokens += result["tokens_used"]

    n = len(test_set)

    return {
        "accuracy": correct / n,
        "format_compliance": compliant / n,
        "avg_quality": total_quality / n,
        "avg_tokens": total_tokens / n,
    }



if __name__ == "__main__":
    # Load fine-tuned model ID
    with open("model_id.txt") as f:
        fine_tuned_model = f.read().strip()

    test_set = load_test_set()
    print(f"Loaded {len(test_set)} test examples\n")

    print("Evaluating base model (gpt-4o-mini)...")
    base_results = evaluate(test_set, "gpt-4o-mini", use_system_prompt=True)

    print(f"\nEvaluating fine-tuned model ({fine_tuned_model})...")
    ft_results = evaluate(test_set, fine_tuned_model, use_system_prompt=True)

    # Print comparison
    print("\n" + "=" * 60)
    print(f"{'Metric':<25} {'Base Model':<18} {'Fine-tuned':<18}")
    print("=" * 60)
    for metric in ["accuracy", "format_compliance", "avg_quality", "avg_tokens"]:
        base_val = base_results[metric]
        ft_val = ft_results[metric]
        print(f"{metric:<25} {base_val:<18.2f} {ft_val:<18.2f}")
    print("=" * 60)
