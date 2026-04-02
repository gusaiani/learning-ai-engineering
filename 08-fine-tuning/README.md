# Module 08 — Fine-tuning

**Goal:** Fine-tune GPT-4o-mini on a custom dataset using the OpenAI fine-tuning API. Understand when fine-tuning beats prompting, how to prepare training data, and how to evaluate the result.

**Time:** ~2 days

---

## Setup & running

```bash
pip install openai tiktoken python-dotenv pandas

# Prepare your dataset
python prepare_data.py

# Launch fine-tuning job
python finetune.py

# Evaluate the fine-tuned model vs. base model
python eval.py
```

---

## What you'll learn

- When to fine-tune vs. when to prompt-engineer or RAG
- How to prepare and validate a JSONL training dataset
- OpenAI's fine-tuning API workflow: upload → create job → monitor → use
- Token counting and cost estimation before you start
- Evaluating fine-tuned models against base models with the same test set
- Hyperparameter basics: epochs, learning rate multiplier, batch size

---

## Concepts

### When to fine-tune (and when NOT to)

Fine-tuning is the right tool when:
- You need a **specific output format** consistently (e.g., always respond as JSON with certain fields)
- You want to **teach a style or tone** that's hard to capture in a prompt (e.g., mimic your company's support voice)
- You need to **reduce token usage** by replacing long system prompts with learned behavior
- You have **hundreds of high-quality examples** of input→output pairs

Fine-tuning is NOT the right tool when:
- You need **up-to-date knowledge** → use RAG instead
- You have **fewer than 50 examples** → use few-shot prompting
- You want to **add new factual knowledge** → fine-tuning memorizes patterns, not facts
- A good prompt already works → don't over-engineer

### The fine-tuning pipeline

```
1. Collect examples (input/output pairs)
2. Format as JSONL (OpenAI chat format)
3. Validate: token counts, format checks, dedup
4. Upload file to OpenAI
5. Create fine-tuning job
6. Monitor progress (training loss, validation loss)
7. Test the fine-tuned model
8. Compare against base model on held-out test set
```

### JSONL format

Each line is a JSON object with a `messages` array:

```json
{"messages": [{"role": "system", "content": "You are a concise tech support agent."}, {"role": "user", "content": "My app crashes on startup"}, {"role": "assistant", "content": "Clear the app cache: Settings → Apps → [App] → Clear Cache. If that fails, reinstall."}]}
```

### Token counting and cost

Before fine-tuning, count tokens to estimate cost:

```python
import tiktoken

enc = tiktoken.encoding_for_model("gpt-4o-mini")
tokens = sum(len(enc.encode(msg["content"])) for msg in example["messages"])
```

OpenAI charges per token for training (~$0.003/1K tokens for GPT-4o-mini). A dataset with 200 examples averaging 500 tokens each, trained for 3 epochs = 200 × 500 × 3 = 300K tokens ≈ $0.90.

### Evaluating fine-tuned models

Always hold out a test set (10-20% of your data). Compare:

| Metric | How to measure |
|--------|---------------|
| Format compliance | Does the output match the expected structure? (regex or schema check) |
| Accuracy | For classification tasks, simple accuracy on test set |
| Quality | LLM-as-judge: have GPT-4o rate outputs 1-5 on relevance, correctness, style |
| Consistency | Run the same input 5 times — how stable are the outputs? |
| Token usage | Compare prompt+completion tokens vs. base model with long system prompt |

---

## Project: Fine-tune a Tech Support Classifier

Build a pipeline that fine-tunes GPT-4o-mini to classify tech support tickets into categories and generate structured responses — then evaluate it against the base model.

### Requirements

```
- Dataset preparation (prepare_data.py):
  1. Generate 200 synthetic tech support tickets using GPT-4o-mini
     - Categories: billing, bug_report, feature_request, how_to, account_access
     - Each ticket has: user message + structured assistant response (JSON with category, priority, response)
  2. Validate the dataset: check JSONL format, token counts, balance across categories
  3. Split into train (80%) and test (20%) sets
  4. Save as train.jsonl and test.jsonl

- Fine-tuning pipeline (finetune.py):
  1. Upload train.jsonl to OpenAI
  2. Create fine-tuning job with gpt-4o-mini-2024-07-18
  3. Poll for job completion, print training metrics
  4. Save the fine-tuned model ID to a file

- Evaluation (eval.py):
  1. Run test set through BOTH base model (with detailed system prompt) and fine-tuned model
  2. Compare:
     - Category accuracy (does it pick the right category?)
     - JSON format compliance (does it always return valid JSON?)
     - Response quality (LLM-as-judge score 1-5)
     - Average token usage (fine-tuned should use fewer tokens — no long system prompt)
  3. Print a comparison table
```

### Starter code

```python
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
```

```python
# finetune.py
import json
import time
import os
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()
client = OpenAI()


def upload_training_file(filepath: str) -> str:
    """Upload a JSONL file to OpenAI and return the file ID."""
    # TODO: use client.files.create()
    pass


def create_fine_tuning_job(file_id: str) -> str:
    """Create a fine-tuning job and return the job ID."""
    # TODO: use client.fine_tuning.jobs.create()
    # Model: gpt-4o-mini-2024-07-18
    # Hyperparameters: n_epochs=3
    pass


def monitor_job(job_id: str) -> str:
    """Poll until the job completes. Return the fine-tuned model ID."""
    # TODO: poll client.fine_tuning.jobs.retrieve() every 30 seconds
    # Print status updates and training metrics
    # Return the fine-tuned model name when done
    pass


if __name__ == "__main__":
    print("Uploading training file...")
    file_id = upload_training_file("train.jsonl")
    print(f"File uploaded: {file_id}")

    print("\nCreating fine-tuning job...")
    job_id = create_fine_tuning_job(file_id)
    print(f"Job created: {job_id}")

    print("\nMonitoring job progress...")
    model_id = monitor_job(job_id)
    print(f"\nFine-tuned model ready: {model_id}")

    # Save model ID for eval
    with open("model_id.txt", "w") as f:
        f.write(model_id)
    print("Model ID saved to model_id.txt")
```

```python
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
```

### Your task

1. **`generate_synthetic_tickets()`** — Use GPT-4o-mini to generate diverse training examples
2. **`build_jsonl_example()`** — Format examples into OpenAI chat fine-tuning format
3. **`validate_dataset()`** — Check format, count tokens, verify category balance
4. **`split_and_save()`** — Shuffle, split 80/20, write JSONL files
5. **`upload_training_file()`** — Upload JSONL to OpenAI
6. **`create_fine_tuning_job()`** — Start the fine-tuning job
7. **`monitor_job()`** — Poll until completion, print metrics
8. **`load_test_set()`** — Load test JSONL and extract user messages + expected outputs
9. **`run_model()`** — Run a single inference, parse output, count tokens
10. **`check_format_compliance()`** — Validate JSON structure
11. **`judge_quality()`** — LLM-as-judge scoring
12. **`evaluate()`** — Full evaluation loop with aggregated metrics

### Hints

<details>
<summary>Generating synthetic data</summary>

```python
def generate_synthetic_tickets(category, count):
    prompt = (
        f"Generate {count} realistic tech support tickets for the '{category}' category.\n"
        "Each ticket should have a natural, varied user message.\n"
        "Return as a JSON array of objects with 'user_message', 'category', 'priority', and 'response' fields.\n"
        "Make the messages diverse — different tones, lengths, levels of detail."
    )
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        response_format={"type": "json_object"},
        temperature=1.0,
    )
    data = json.loads(response.choices[0].message.content)
    # The model may nest the array under a key
    tickets = data if isinstance(data, list) else list(data.values())[0]
    return [
        {"user_message": t["user_message"],
         "assistant_response": json.dumps({"category": t["category"], "priority": t["priority"], "response": t["response"]})}
        for t in tickets
    ]
```
</details>

<details>
<summary>Uploading and fine-tuning</summary>

```python
def upload_training_file(filepath):
    with open(filepath, "rb") as f:
        response = client.files.create(file=f, purpose="fine-tune")
    return response.id

def create_fine_tuning_job(file_id):
    job = client.fine_tuning.jobs.create(
        training_file=file_id,
        model="gpt-4o-mini-2024-07-18",
        hyperparameters={"n_epochs": 3},
    )
    return job.id
```
</details>

<details>
<summary>Monitoring the job</summary>

```python
def monitor_job(job_id):
    while True:
        job = client.fine_tuning.jobs.retrieve(job_id)
        print(f"  Status: {job.status}")
        if job.status == "succeeded":
            return job.fine_tuned_model
        if job.status == "failed":
            raise RuntimeError(f"Fine-tuning failed: {job.error}")
        time.sleep(30)
```
</details>

---

## Stretch goals

- **Custom validation loss tracking** — Plot training and validation loss curves using matplotlib
- **Hyperparameter sweep** — Try different epoch counts (1, 2, 3, 5) and learning rate multipliers, compare results
- **LoRA locally** — Use Hugging Face PEFT + a small open model (Llama 3.1 8B) to fine-tune locally on the same dataset
- **Data augmentation** — Paraphrase training examples to 2x your dataset, measure if more data helps
- **A/B comparison UI** — Build a simple Gradio app that shows base vs. fine-tuned responses side by side
- **Cost analysis** — Calculate exact cost per training run, cost per inference, and break-even point vs. using a long system prompt

---

## Key questions to answer before moving on

1. How many examples do you need before fine-tuning outperforms a well-crafted system prompt?
2. What happens if your training data has label noise (e.g., 10% of categories are wrong)?
3. How do you detect overfitting during fine-tuning? What do the training curves look like?
4. When would you fine-tune a small model (GPT-4o-mini) vs. prompt a large model (GPT-4o)?
5. How does fine-tuning interact with RAG — can you combine them? When would you?

---

## Resources

- [OpenAI Fine-tuning Guide](https://platform.openai.com/docs/guides/fine-tuning)
- [OpenAI Fine-tuning API Reference](https://platform.openai.com/docs/api-reference/fine-tuning)
- [tiktoken — OpenAI's tokenizer](https://github.com/openai/tiktoken)
- [Hugging Face PEFT — Parameter Efficient Fine-Tuning](https://huggingface.co/docs/peft)
- [LoRA: Low-Rank Adaptation of Large Language Models](https://arxiv.org/abs/2106.09685)

---

**When done:** Mark Module 08 as shipped in the root README, commit, and move to [Module 09](../09-multi-agent/).
