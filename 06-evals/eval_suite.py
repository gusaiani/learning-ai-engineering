# eval_suite.py
import os
import sys
import json
import time
from pathlib import Path
from dataclasses import dataclass, asdict

import numpy as np
from openai import OpenAI
from dotenv import load_dotenv
from rich.console import Console
from rich.table import Table
from rich.panel import Panel

load_dotenv(Path(__file__).parent.parent / ".env")

openai_client = OpenAI()
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


# Test dataset based on Graham & Dodd's Security Analysis (Chapters 9 & 11)
TEST_CASES = [
    # ── Factual (easy) ────────────────────────────────────────────
    TestCase(
        question="Under the New York statute, what type of bonds are eligible in the public-utility group?",
        expected_answer="Only bonds secured by mortgage are eligible in the public-utility group.",
        category="factual",
        difficulty="easy",
    ),
    TestCase(
        question="What does the New York statute require about the relationship between capital stock and mortgage debt for public-utility bonds?",
        expected_answer="The capital stock shall be equal to at least two-thirds of the mortgage debt.",
        category="factual",
        difficulty="easy",
    ),
    TestCase(
        question="What is the maximum percentage of mortgaged property value that mortgage debt can represent under the New York law?",
        expected_answer="60%",
        category="factual",
        difficulty="easy",
    ),
    TestCase(
        question="How many times must a railroad earn its fixed charges to meet the New York statutory requirement?",
        expected_answer="1.5 times in five out of the six years immediately preceding, and also in the latest year.",
        category="factual",
        difficulty="medium",
    ),
    TestCase(
        question="What earnings coverage is required for gas, electric, and telephone bonds?",
        expected_answer="Average earnings for the past five years must equal twice the average total-interest charges, and the same coverage must be shown in the latest year.",
        category="factual",
        difficulty="medium",
    ),

    # ── Multi-hop ─────────────────────────────────────────────────
    TestCase(
        question="How do income bonds compare to debentures in terms of investment safety, and why does the New York statute's treatment of them seem contradictory?",
        expected_answer="Income bonds are weaker than debentures because interest payment depends on earnings or directors' discretion. Yet the New York statute accepts railroad income bonds on the same basis as debentures, which Graham considers objectionable, while simultaneously excluding all unsecured public-utility issues, which he considers too severe.",
        category="multi_hop",
        difficulty="hard",
    ),
    TestCase(
        question="What is the relationship between stock equity, net assets, and bonded debt, and how do the two New York statute requirements relate to each other?",
        expected_answer="Stock equity equals net assets minus bonded debt. The two requirements — mortgage debt not exceeding 60% of property value, and capital stock equaling at least two-thirds of mortgage debt — are broadly equivalent. However, when a company has substantial unsecured debt, it might meet the first but not the second, so the second provides additional protection.",
        category="multi_hop",
        difficulty="hard",
    ),
    TestCase(
        question="Why does Graham argue that the distinction between first mortgage bonds and debentures is less important than the overall ability of the company to pay, and what does he suggest instead?",
        expected_answer="If a company has only one bond issue, it makes little difference whether it is a first mortgage or debenture, provided the debenture is protected against future senior issues. The investor's chief reliance in both cases is the ability of the company to meet all its obligations. Graham argues against attaching predominant weight to specific security.",
        category="multi_hop",
        difficulty="hard",
    ),
    TestCase(
        question="How does the treatment of income bonds as part of stock equity affect the analysis of fixed-interest bonds ahead of them? Use the Colorado Fuel and Iron example.",
        expected_answer="Junior income bonds of long maturity are so close to preferred stock that their market value can be considered part of the stock equity protecting fixed-interest bonds. In the Colorado Fuel and Iron case, the income bond and stock equity totaled $13,355,000 protecting $4,483,000 of first mortgage 5s, making the position much stronger than if the junior lien carried fixed interest.",
        category="multi_hop",
        difficulty="hard",
    ),

    # ── Adversarial ───────────────────────────────────────────────
    TestCase(
        question="What does Graham say about using the P/E ratio to evaluate bonds?",
        expected_answer="The provided context does not contain information about P/E ratios in the context of bond evaluation.",
        category="adversarial",
        difficulty="medium",
    ),
    TestCase(
        question="According to Graham, what is the ideal stock-to-bond ratio for tech companies?",
        expected_answer="The context does not discuss tech companies. Graham discusses stock equity to bonded debt ratios for railroads, public utilities, and industrials, but not technology companies specifically.",
        category="adversarial",
        difficulty="medium",
    ),
    TestCase(
        question="Does Graham recommend that all investors avoid debenture bonds entirely?",
        expected_answer="No. Graham argues against the categorical exclusion of unsecured bonds. He considers the New York statute's exclusion of all unsecured public-utility issues to be out of date and illogical. He does not favor the establishment of standards that favor secured bonds over debentures per se.",
        category="adversarial",
        difficulty="medium",
    ),

    # ── No-answer ─────────────────────────────────────────────────
    TestCase(
        question="What are Graham's views on cryptocurrency as an investment vehicle?",
        expected_answer="The context does not contain any information about cryptocurrency.",
        category="no_answer",
        difficulty="easy",
    ),
    TestCase(
        question="What specific companies does Graham recommend buying bonds from in 2024?",
        expected_answer="The context does not contain recommendations for specific companies to buy bonds from in 2024. The text is from a historical work on security analysis principles.",
        category="no_answer",
        difficulty="easy",
    ),
    TestCase(
        question="What does Graham say about ESG investing and environmental factors in bond selection?",
        expected_answer="The context does not contain any discussion of ESG investing or environmental factors.",
        category="no_answer",
        difficulty="easy",
    ),
]


# ── Scoring functions ─────────────────────────────────────────────


def score_exact_match(expected: str, actual: str) -> bool:
    """Case-insensitive exact match after stripping whitespace."""
    return expected.strip().lower() == actual.strip().lower()


def get_embedding(text: str) -> list[float]:
    """Get embedding vector for a text string using OpenAI."""
    response = openai_client.embeddings.create(
        model="text-embedding-3-small",
        input=text,
    )
    return response.data[0].embedding


def score_semantic_similarity(expected: str, actual: str) -> float:
    """Cosine similarity between embeddings of expected and actual answers."""
    vec_a = np.array(get_embedding(expected))
    vec_b = np.array(get_embedding(actual))
    return float(np.dot(vec_a, vec_b) / (np.linalg.norm(vec_a) * np.linalg.norm(vec_b)))


def score_llm_judge(question: str, expected: str, actual: str) -> dict:
    """
    Use GPT as a judge to rate the answer on correctness,
    completeness, and faithfulness (each 1-5).

    Returns: {"correctness": N, "completeness": N, "faithfulness": N, "explanation": "..."}
    """
    prompt = (
        f"You are an eval judge. Rate the candidate answer.\n\n"
        f"Question: {question}\n"
        f"Reference answer: {expected}\n"
        f"Candidate answer: {actual}\n\n"
        f"Rate each dimension 1-5:\n"
        f"- correctness: Is the answer factually correct?\n"
        f"- completeness: Does it cover the key points from the reference?\n"
        f"- faithfulness: Does it avoid adding information not in the reference?\n\n"
        f"Return ONLY valid JSON:\n"
        f'{{"correctness": N, "completeness": N, "faithfulness": N, "explanation": "brief reason"}}'
    )

    response = openai_client.chat.completions.create(
        model="gpt-4o-mini",
        max_tokens=500,
        messages=[{"role": "user", "content": prompt}],
    )

    raw = response.choices[0].message.content

    try:
        result = json.loads(raw)
    except json.JSONDecodeError:
        result = {
            "correctness": 1,
            "completeness": 1,
            "faithfulness": 1,
            "explanation": "Failed to parse judge response",
        }

    return result

# ── RAG system interface ─────────────────────────────────────────


def ask_rag(question: str) -> str:
    """Send a question to GPT-4o-mini (mock RAG — no retrieval, just LLM knowledge)."""
    system_msg = (
        "You are a Q&A system answering questions about Graham & Dodd's "
        "Security Analysis (Chapters 9 & 11). Answer concisely based only "
        "on what you know from the text. If the question is outside the "
        "scope of those chapters, say so clearly."
    )

    response = openai_client.chat.completions.create(
        model="gpt-4o-mini",
        max_tokens=500,
        messages=[
            {"role": "system", "content": system_msg},
            {"role": "user", "content": question}
        ]
    )

    return response.choices[0].message.content


# ── Eval runner ───────────────────────────────────────────────────


def run_eval(test_cases: list[TestCase]) -> list[EvalResult]:
    """
    Run the full eval suite:
    1. For each test case, get the RAG answer
    2. Score it with all three methods
    3. Collect and return results
    """
    results = []

    for i, tc in enumerate(test_cases, 1):
        console.print(f"[dim]({i}/{len(test_cases)})[/dim] {tc.category}: {tc.question[:60]}…")

        actual = ask_rag(tc.question)
        exact = score_exact_match(tc.expected_answer, actual)
        similarity = score_semantic_similarity(tc.expected_answer, actual)
        judge = score_llm_judge(tc.question, tc.expected_answer, actual)

        result = EvalResult(
            test_case=tc,
            actual_answer=actual,
            exact_match=exact,
            semantic_similarity=similarity,
            llm_judge_scores={
                "correctness": judge.get("correctness", 1),
                "completeness": judge.get("completeness", 1),
                "faithfulness": judge.get("faithfulness", 1),
            },
            llm_judge_explanation=judge.get("explanation", "")
        )
        results.append(result)

    return results



def compute_summary(results: list[EvalResult]) -> dict:
    """
    Compute aggregate scores:
    - Overall averages for each metric
    - Per-category breakdowns
    - Pass rate (exact match)
    """
    summary = {
        "total_cases": len(results),
        "exact_match_rate": sum(r.exact_match for r in results) / len(results),
        "avg_semantic_similarity": float(np.mean([r.semantic_similarity for r in results])),
        "avg_correctness": float(np.mean([r.llm_judge_scores["correctness"] for r in results])),
        "avg_completeness": float(np.mean([r.llm_judge_scores["completeness"] for r in results])),
        "avg_faithfulness": float(np.mean([r.llm_judge_scores["faithfulness"] for r in results])),
        "by_category": {},
    }

    for cat in set(r.test_case.category for r in results):
        cat_results = [r for r in results if r.test_case.category == cat]
        summary["by_category"][cat] = {
            "count": len(cat_results),
            "exact_match_rate": sum(r.exact_match for r in cat_results) / len(cat_results),
            "avg_similarity": float(np.mean([r.semantic_similarity for r in cat_results])),
            "avg_correctness": float(np.mean([r.llm_judge_scores["correctness"] for r in cat_results])),
            "avg_completeness": float(np.mean([r.llm_judge_scores["completeness"] for r in cat_results])),
            "avg_faithfulness": float(np.mean([r.llm_judge_scores["faithfulness"] for r in cat_results])),
        }

    return summary


def save_results(results: list[EvalResult], summary: dict) -> Path:
    """Save results and summary to a timestamped JSON file."""
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    filename = RESULTS_DIR / f"run_{timestamp}.json"

    data = {
        "timestamp": timestamp,
        "summary": summary,
        "results": [
            {
                "question": r.test_case.question,
                "category": r.test_case.category,
                "difficulty": r.test_case.difficulty,
                "expected_answer": r.test_case.expected_answer,
                "actual_answer": r.actual_answer,
                "exact_match": r.exact_match,
                "semantic_similarity": r.semantic_similarity,
                "llm_judge_scores": r.llm_judge_scores,
                "llm_judge_explanation": r.llm_judge_explanation,
            }
            for r in results
        ],
    }

    filename.write_text(json.dumps(data, indent=2))
    return filename


# ── Display ───────────────────────────────────────────────────────


def display_results(results: list[EvalResult], summary: dict):
    """Show a rich table with per-case results and a summary panel."""
    table = Table(title="Eval Results")
    table.add_column("Category", style="cyan")
    table.add_column("Question", max_width=50)
    table.add_column("Exact", justify="center")
    table.add_column("Similarity", justify="center")
    table.add_column("Judge Avg", justify="center")
    table.add_column("Pass/Fail", justify="center")

    for r in results:
        judge_avg = np.mean(list(r.llm_judge_scores.values()))
        passed = judge_avg >= 3.0 and r.semantic_similarity >= 0.7

        exact_str = "[green]✓[/green]" if r.exact_match else "[red]✗[/red]"
        sim_color = "green" if r.semantic_similarity >= 0.7 else "red"
        judge_color = "green" if judge_avg >= 3.0 else "red"
        pass_str = "[green]PASS[/green]" if passed else "[red]FAIL[/red]"

        table.add_row(
            r.test_case.category,
            r.test_case.question[:50] + "…",
            exact_str,
            f"[{sim_color}]{r.semantic_similarity:.2f}[/{sim_color}]",
            f"[{judge_color}]{judge_avg:.1f}[/{judge_color}]",
            pass_str,
        )

    console.print(table)

    console.print(Panel(
        f"Cases: {summary['total_cases']}  |  "
        f"Exact match: {summary['exact_match_rate']:.0%}  |  "
        f"Avg similarity: {summary['avg_semantic_similarity']:.2f}  |  "
        f"Correctness: {summary['avg_correctness']:.1f}  |  "
        f"Completeness: {summary['avg_completeness']:.1f}  |  "
        f"Faithfulness: {summary['avg_faithfulness']:.1f}",
        title="Summary",
    ))


def compare_runs(file_a: Path, file_b: Path):
    """
    Load two result files and display a comparison:
    - Show metric deltas (↑ improvement, ↓ regression)
    - Highlight any category where faithfulness dropped
    """
    a = json.loads(file_a.read_text())
    b = json.loads(file_b.read_text())

    console.print(Panel(
        f"[dim]{file_a.name}[/dim] vs [bold]{file_b.name}[/bold]",
        title="Run Comparison",
    ))

    metrics = [
        ("exact_match_rate", "Exact Match Rate"),
        ("avg_semantic_similarity", "Avg Similarity"),
        ("avg_correctness", "Avg Correctness"),
        ("avg_completeness", "Avg Completeness"),
        ("avg_faithfulness", "Avg Faithfulness"),
    ]

    table = Table()
    table.add_column("Metric")
    table.add_column("Run A", justify="center")
    table.add_column("Run B", justify="center")
    table.add_column("Delta", justify="center")

    for key, label in metrics:
        val_a = a["summary"][key]
        val_b = b["summary"][key]
        delta = val_b - val_a
        arrow = "↑" if delta > 0 else "↓" if delta < 0 else "="
        color = "green" if delta > 0 else "red" if delta < 0 else "white"

        table.add_row(
            label,
            f"{val_a:.2f}",
            f"{val_b:.2f}",
            f"[{color}]{arrow} {delta:+.2f}[/{color}]",
        )

    console.print(table)


# ── CLI ──────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="RAG Eval Suite")
    parser.add_argument("--category", help="Run only a specific category")
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

        results = run_eval(cases)
        summary = compute_summary(results)

        display_results(results, summary)

        saved_path = save_results(results, summary)
        console.print(f"\n[dim]Results saved to {saved_path}[/dim]")
