# Module 06 — Evals: Measuring LLM Quality

**Goal:** Build an eval suite that systematically measures your RAG system's quality — so you can improve it with confidence instead of vibes.

**Time:** ~2 days

---

## Setup & running

```bash
pip install anthropic openai python-dotenv numpy rich

# Run the full eval suite against your RAG system
python eval_suite.py

# Run a specific eval category only
python eval_suite.py --category faithfulness

# Compare two runs side by side
python eval_suite.py --compare results/run_001.json results/run_002.json
```

---

## What you'll learn

- Why evals are the most important skill in AI engineering (not prompting)
- How to design test cases that catch real failures
- The three pillars of RAG evaluation: retrieval, faithfulness, and answer quality
- LLM-as-judge: using one model to grade another
- How to detect regressions before they reach users
- When to use automated evals vs. human evaluation

---

## Concepts

### Why evals matter

Every time you change a prompt, swap a model, adjust chunking, or tweak retrieval — you need to know if things got better or worse. Without evals, you're guessing. With evals, you're engineering.

The workflow:

```
Make a change → Run evals → See scores → Decide to keep or revert
```

This is the difference between "AI tinkering" and "AI engineering." Companies hiring AI engineers care about this more than anything else.

### What to evaluate in a RAG system

RAG has three failure modes, and you need evals for each:

| Failure mode | What went wrong | Eval type |
|---|---|---|
| Bad retrieval | Wrong chunks were pulled from the index | **Retrieval eval** |
| Hallucination | Model ignored the context and made things up | **Faithfulness eval** |
| Poor answer | Answer is correct but unhelpful, verbose, or misformatted | **Answer quality eval** |

### Eval dataset structure

Each test case has:

```python
{
    "question": "What is the capital of France?",
    "expected_answer": "Paris",
    "expected_chunks": ["france_geography.pdf chunk 3"],  # optional: for retrieval eval
    "category": "factual",
    "difficulty": "easy"
}
```

You want 20–50 test cases covering:
- **Easy factual questions** (sanity checks — these should always pass)
- **Multi-hop questions** (require combining info from multiple chunks)
- **Adversarial questions** (trick the model into hallucinating)
- **Edge cases** (questions with no answer in the corpus)

### LLM-as-judge

The key insight: you can use an LLM to grade another LLM's output. This is faster and cheaper than human evaluation, and surprisingly reliable when done right.

```python
judge_prompt = """You are an eval judge. Given a question, a reference answer,
and a candidate answer, rate the candidate on a scale of 1-5.

Question: {question}
Reference answer: {expected}
Candidate answer: {actual}

Rate on:
- Correctness (1-5): Is the answer factually correct?
- Completeness (1-5): Does it cover the key points?
- Faithfulness (1-5): Does it stick to the provided context without hallucinating?

Return JSON: {"correctness": N, "completeness": N, "faithfulness": N, "explanation": "..."}
"""
```

Important: the judge model should be at least as capable as the model being judged. Use Claude Sonnet or GPT-4o as the judge, even if your RAG uses a cheaper model.

### Scoring approaches

| Approach | When to use | Example |
|---|---|---|
| **Exact match** | Factual, short answers | `actual.strip().lower() == expected.strip().lower()` |
| **Contains** | Answer must include a key term | `"Paris" in actual` |
| **Semantic similarity** | Meaning matters, not wording | Cosine similarity of embeddings > 0.85 |
| **LLM-as-judge** | Complex, nuanced answers | Judge prompt returning structured scores |
| **Regex** | Structured output format | Pattern matching on JSON, dates, etc. |

### Regression testing

The real power of evals is catching regressions. Save every run's results to a JSON file. Before deploying a change, compare the new run against the baseline:

```
Baseline (run_001):  correctness=4.2  faithfulness=4.5  completeness=3.8
New run  (run_002):  correctness=4.3  faithfulness=3.9  completeness=4.0
                                       ↑ regression!
```

A 0.6 drop in faithfulness means your change introduced hallucinations. Revert it.

---

## Project: RAG Eval Suite

Build a CLI tool that evaluates your Module 04 RAG system across three dimensions, saves results as JSON, and can compare runs to detect regressions.

### Requirements

```
- Build a test dataset of 15+ question/answer pairs covering 4 categories:
  factual, multi-hop, adversarial, no-answer
- Implement 3 scoring functions:
  1. exact_match — case-insensitive string comparison
  2. semantic_similarity — cosine similarity of embeddings (reuse Module 03 skills)
  3. llm_judge — Claude or GPT rates correctness, completeness, and faithfulness (1-5)
- Implement a RAG evaluator that:
  - Runs each test case through your RAG system (or a mock of it)
  - Scores the output using all applicable scoring functions
  - Tracks per-category and overall scores
- Save results to a JSON file with timestamp, scores, and per-case details
- Implement a comparison mode that loads two result files and highlights regressions
- Display results in a rich table with color-coded pass/fail
```

### Starter code

```python
# eval_suite.py
import os
import json
import time
from pathlib import Path
from dataclasses import dataclass, asdict

import numpy as np
from openai import OpenAI
from anthropic import Anthropic
from dotenv import load_dotenv
from rich.console import Console
from rich.table import Table
from rich.panel import Panel

load_dotenv(Path(__file__).parent.parent / ".env")

openai_client = OpenAI()
anthropic_client = Anthropic()
console = Console()

RESULTS_DIR = Path(__file__).parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)


# ── Test dataset ──────────────────────────────────────────────────


@dataclass
class TestCase:
    question: str
    expected_answer: str
    category: str  # factual, multi_hop, adversarial, no_answer
    difficulty: str  # easy, medium, hard


@dataclass
class EvalResult:
    test_case: TestCase
    actual_answer: str
    exact_match: bool
    semantic_similarity: float
    llm_judge_scores: dict  # {"correctness": N, "completeness": N, "faithfulness": N}
    llm_judge_explanation: str


# TODO: Create your test dataset — at least 15 cases across all 4 categories
# These should be based on the PDFs you used in Module 04
TEST_CASES = [
    # TestCase(
    #     question="...",
    #     expected_answer="...",
    #     category="factual",
    #     difficulty="easy",
    # ),
]


# ── Scoring functions ─────────────────────────────────────────────


def score_exact_match(expected: str, actual: str) -> bool:
    """Case-insensitive exact match after stripping whitespace."""
    # TODO: implement — straightforward string comparison
    pass


def get_embedding(text: str) -> list[float]:
    """Get embedding vector for a text string using OpenAI."""
    # TODO: reuse the embedding approach from Module 03
    # Use text-embedding-3-small
    pass


def score_semantic_similarity(expected: str, actual: str) -> float:
    """Cosine similarity between embeddings of expected and actual answers."""
    # TODO: embed both strings, compute cosine similarity
    # Return a float between -1 and 1
    pass


def score_llm_judge(question: str, expected: str, actual: str) -> dict:
    """
    Use Claude as a judge to rate the answer on correctness,
    completeness, and faithfulness (each 1-5).

    Returns: {"correctness": N, "completeness": N, "faithfulness": N, "explanation": "..."}
    """
    # TODO: send a judge prompt to Claude
    # Parse the JSON response
    # Handle cases where the model doesn't return valid JSON
    pass


# ── RAG system interface ─────────────────────────────────────────


def ask_rag(question: str) -> str:
    """
    Send a question to your RAG system and get the answer.

    Option A: Import and call your Module 04 rag.py directly
    Option B: Implement a simple mock RAG for testing the eval framework
    """
    # TODO: either import your RAG system or create a mock
    # For the mock approach, you can use Claude directly with a system prompt
    # that simulates RAG behavior
    pass


# ── Eval runner ───────────────────────────────────────────────────


def run_eval(test_cases: list[TestCase], verbose: bool = False) -> list[EvalResult]:
    """
    Run the full eval suite:
    1. For each test case, get the RAG answer
    2. Score it with all three methods
    3. Collect and return results
    """
    # TODO: iterate through test cases, score each one
    # Print progress as you go
    # Return list of EvalResult
    pass


def compute_summary(results: list[EvalResult]) -> dict:
    """
    Compute aggregate scores:
    - Overall averages for each metric
    - Per-category breakdowns
    - Pass rate (exact match)
    """
    # TODO: aggregate the results into a summary dict
    pass


def save_results(results: list[EvalResult], summary: dict) -> Path:
    """Save results and summary to a timestamped JSON file."""
    # TODO: serialize to JSON, save to RESULTS_DIR
    # Return the path to the saved file
    pass


# ── Display ───────────────────────────────────────────────────────


def display_results(results: list[EvalResult], summary: dict):
    """Show a rich table with per-case results and a summary panel."""
    # TODO: build a Rich table with columns:
    # Category | Question (truncated) | Exact Match | Similarity | Judge Avg | Pass/Fail
    # Color-code: green for pass, red for fail
    # Show summary panel at the end
    pass


def compare_runs(file_a: Path, file_b: Path):
    """
    Load two result files and display a comparison:
    - Show metric deltas (↑ improvement, ↓ regression)
    - Highlight any category where faithfulness dropped
    """
    # TODO: load both JSON files, compute deltas, display comparison table
    pass


# ── CLI ──────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="RAG Eval Suite")
    parser.add_argument("--category", help="Run only a specific category")
    parser.add_argument("--verbose", "-v", action="store_true", help="Show detailed output")
    parser.add_argument(
        "--compare", nargs=2, metavar="FILE",
        help="Compare two result files"
    )
    args = parser.parse_args()

    if args.compare:
        compare_runs(Path(args.compare[0]), Path(args.compare[1]))
    else:
        cases = TEST_CASES
        if args.category:
            cases = [c for c in cases if c.category == args.category]
            console.print(f"Filtering to category: [bold]{args.category}[/bold] ({len(cases)} cases)")

        if not cases:
            console.print("[red]No test cases found.[/red] Add test cases to TEST_CASES list.")
            sys.exit(1)

        console.print(Panel(
            f"Running [bold]{len(cases)}[/bold] eval cases",
            title="🧪 RAG Eval Suite"
        ))

        results = run_eval(cases, verbose=args.verbose)
        summary = compute_summary(results)

        display_results(results, summary)

        saved_path = save_results(results, summary)
        console.print(f"\n[dim]Results saved to {saved_path}[/dim]")
```

### Your task

1. **Test dataset** — Write 15+ test cases based on your Module 04 PDFs (or create sample questions if you don't have the PDFs handy)
2. **`score_exact_match()`** — Simple case-insensitive comparison
3. **`get_embedding()` + `score_semantic_similarity()`** — Cosine similarity of answer embeddings
4. **`score_llm_judge()`** — Claude rates correctness, completeness, faithfulness on a 1–5 scale
5. **`ask_rag()`** — Either import your real RAG or build a mock
6. **`run_eval()`** — Loop through test cases, score each, collect results
7. **`compute_summary()`** — Aggregate scores overall and per-category
8. **`save_results()`** — Persist to JSON in `results/`
9. **`display_results()`** — Rich table with color-coded output
10. **`compare_runs()`** — Load two JSON files, show deltas and regressions

### Hints

<details>
<summary>score_llm_judge — the judge prompt</summary>

```python
judge_response = anthropic_client.messages.create(
    model="claude-sonnet-4-20250514",
    max_tokens=500,
    messages=[{
        "role": "user",
        "content": f"""You are an eval judge. Rate the candidate answer.

Question: {question}
Reference answer: {expected}
Candidate answer: {actual}

Rate each dimension 1-5:
- correctness: Is the answer factually correct?
- completeness: Does it cover the key points from the reference?
- faithfulness: Does it avoid adding information not in the reference?

Return ONLY valid JSON:
{{"correctness": N, "completeness": N, "faithfulness": N, "explanation": "brief reason"}}"""
    }]
)
return json.loads(judge_response.content[0].text)
```
</details>

<details>
<summary>compute_summary — aggregation pattern</summary>

```python
summary = {
    "total_cases": len(results),
    "exact_match_rate": sum(r.exact_match for r in results) / len(results),
    "avg_semantic_similarity": np.mean([r.semantic_similarity for r in results]),
    "avg_correctness": np.mean([r.llm_judge_scores["correctness"] for r in results]),
    "avg_completeness": np.mean([r.llm_judge_scores["completeness"] for r in results]),
    "avg_faithfulness": np.mean([r.llm_judge_scores["faithfulness"] for r in results]),
    "by_category": {}
}

for cat in set(r.test_case.category for r in results):
    cat_results = [r for r in results if r.test_case.category == cat]
    summary["by_category"][cat] = {
        "count": len(cat_results),
        "avg_correctness": np.mean([r.llm_judge_scores["correctness"] for r in cat_results]),
        # ... etc
    }
```
</details>

<details>
<summary>compare_runs — detecting regressions</summary>

```python
def compare_runs(file_a, file_b):
    a = json.loads(file_a.read_text())
    b = json.loads(file_b.read_text())

    metrics = ["avg_correctness", "avg_completeness", "avg_faithfulness"]
    for m in metrics:
        delta = b["summary"][m] - a["summary"][m]
        arrow = "↑" if delta > 0 else "↓" if delta < 0 else "="
        color = "green" if delta > 0 else "red" if delta < 0 else "white"
        console.print(f"  {m}: {a['summary'][m]:.2f} → {b['summary'][m]:.2f} [{color}]{arrow} {delta:+.2f}[/]")
```
</details>

---

## Stretch goals

- **Eval-driven prompt optimization** — run evals, identify the weakest category, tweak the RAG prompt, re-run, and verify improvement
- **Cost tracking** — log token usage per eval run and report cost (input/output tokens × price per token)
- **Confidence intervals** — run each test case 3 times and report mean ± std to account for LLM non-determinism
- **Human-in-the-loop** — for the LLM judge's lowest-scoring cases, prompt the user to provide a manual rating
- **CI integration** — write a script that runs evals and exits with code 1 if any metric drops below a threshold
- **Retrieval eval** — if your RAG returns source chunks, compare them against expected chunks using overlap metrics (precision@k, recall@k)

---

## Key questions to answer before moving on

1. Why is LLM-as-judge useful even though it's imperfect? When would you use human eval instead?
2. How many test cases do you need for reliable evals? What's the trade-off between coverage and cost?
3. What's the difference between correctness and faithfulness? Can an answer be correct but unfaithful?
4. How would you eval a system where there's no single "right" answer (e.g., creative writing, summarization)?
5. How do you prevent your eval dataset from "leaking" into your prompts — i.e., overfitting to the test set?

---

## Resources

- [Hamel Husain — Your AI Product Needs Evals](https://hamel.dev/blog/posts/evals/)
- [Anthropic — Evaluating Claude](https://docs.anthropic.com/en/docs/build-with-claude/develop-tests)
- [OpenAI — Evals framework](https://github.com/openai/evals)
- [Braintrust — LLM Eval guide](https://www.braintrustdata.com/docs)
- [Eugene Yan — LLM Patterns (evals section)](https://eugeneyan.com/writing/llm-patterns/)

---

**When done:** Mark Module 06 as shipped in the root README, commit, and move to [Module 07](../07-streaming-production/).
