"""
Multi-agent Research Pipeline

Usage:
    python pipeline.py "Your research question here"

Architecture:
    User query → Coordinator → [Researcher x3 in parallel] → Critic → Synthesizer → Report
"""

import asyncio
import json
import sys
import time
from dataclasses import dataclass, field
from openai import AsyncOpenAI
from dotenv import load_dotenv

load_dotenv()

# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class Agent:
    """Defines an agent with a role, system prompt, and model."""
    name: str
    role: str  # coordinator | researcher | critic | synthesizer
    system_prompt: str
    model: str = "gpt-5.4"


@dataclass
class Message:
    """Structured message passed between agents."""
    from_agent: str
    to_agent: str
    type: str  # task_assignment | research_result | critique | revision | final_report
    content: str
    metadata: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "from_agent": self.from_agent,
            "to_agent": self.to_agent,
            "type": self.type,
            "content": self.content,
            "metadata": self.metadata,
        }


class TokenTracker:
    """Tracks token usage across all agents."""

    def __init__(self):
        self.usage: dict[str, dict[str, int]] = {}

    def add(self, agent_name: str, input_tokens: int, output_tokens: int):
        if agent_name not in self.usage:
            self.usage[agent_name] = {"input": 0, "output": 0}
        self.usage[agent_name]["input"] += input_tokens
        self.usage[agent_name]["output"] += output_tokens

    def summary(self):
        print("\n--- Token Usage ---")
        total_in = 0
        total_out = 0
        for agent, tokens in self.usage.items():
            print(f"  {agent:<20} input={tokens['input']:>6}  output={tokens['output']:>6}")
            total_in += tokens["input"]
            total_out += tokens["output"]
        print(f"  {'TOTAL':<20} input={total_in:>6}  output={total_out:>6}")
        print("-----------------\n")

# ---------------------------------------------------------------------------
# System prompts
# ---------------------------------------------------------------------------

COORDINATOR_PROMPT = """You are a research coordinator. Your job is to decompose a research question into exactly 3 focused subtasks.

Given a research question, return a JSON object with this structure:
{"subtasks": ["subtask 1", "subtask 2", "subtask 3"]}

Each subtask should:
- Cover a distinct angle of the research question
- Be specific enough for a single researcher to investigate
- Together, the 3 subtasks should comprehensively cover the original question

Return ONLY the JSON object, nothing else."""

RESEARCHER_PROMPT = """You are a research specialist. You investigate one specific subtask in depth.

Given a research subtask, provide:
1. Key findings (3-5 bullet points)
2. Supporting evidence or reasoning
3. Any caveats or uncertainties

Be thorough but concise. Prioritize depth over breadth.
Flag anything you're uncertain about with [UNCERTAIN]."""

CRITIC_PROMPT = """You are a research critic. You review combined research findings for quality.

Evaluate the research for:
- Gaps: important angles not covered
- Contradictions: claims that conflict with each other
- Unsupported claims: assertions without reasoning
- Redundancy: repeated points across researchers

Return a JSON object:
{"approved": true/false, "issues": ["issue 1", "issue 2", ...], "summary": "overall assessment"}

Be specific and actionable. If the research is solid, approve it.
Return ONLY the JSON object, nothing else."""

SYNTHESIZER_PROMPT = """You are a research synthesizer. You combine approved research into a clear, structured final report.

Rules:
- Do NOT add new claims beyond what the researchers found
- Organize by theme, not by researcher
- Use clear headings and bullet points
- Note any remaining uncertainties flagged by researchers
- Keep it concise but comprehensive

Output format:
# Research Report: [Topic]

## Executive Summary
[2-3 sentences]

## Key Findings
[Organized by theme with bullet points]

## Uncertainties & Limitations
[What we're less sure about]

## Conclusion
[1-2 sentences]"""


# ---------------------------------------------------------------------------
# Core agent function
# ---------------------------------------------------------------------------

# - Takes an Agent, a user message string, and the AsyncOpenAI client
# - Calls client.chat.completions.create() with the agent's model and system prompt
# - Returns a tuple of (response_content: str, input_tokens: int, output_tokens: int)
# - If the agent is coordinator or critic, use response_format={"type": "json_object"}
async def call_agent(
    agent: Agent,
    user_message: str,
    client: AsyncOpenAI,
    tracker: TokenTracker,
) -> str:
    kwargs = {
        "model": agent.model,
        "messages": [
            {"role": "system", "content": agent.system_prompt},
            {"role": "user", "content": user_message},
        ],
    }

    if agent.role in {"coordinator", "critic"}:
        kwargs["response_format"] = {"type": "json_object"}

    response = await client.chat.completions.create(**kwargs)

    usage = response.usage
    tracker.add(agent.name, usage.prompt_tokens, usage.completion_tokens)

    return response.choices[0].message.content


# ---------------------------------------------------------------------------
# Agent runners
# ---------------------------------------------------------------------------

# - Create a Coordinator agent (use gpt-4o model)
# - Call it with the user's research question
# - Parse the JSON response to extract the list of subtasks
# - Return the list of subtask strings
async def decompose_query(query: str, client: AsyncOpenAI, tracker: TokenTracker) -> list[str]:
    coordinator = Agent(
        name="coordinator",
        role="coordinator",
        system_prompt=COORDINATOR_PROMPT,
        model="gpt-4o"
    )

    result = await call_agent(coordinator, query, client, tracker)
    data = json.loads(result)
    return data["subtasks"]


# - Create a Researcher agent (use gpt-5.4 model)
# - Call it with the subtask
# - Return a Message with type="research_result"
# - Wrap in asyncio.wait_for() with a 30-second timeout
# - On timeout, return a Message indicating the researcher timed out
async def run_researcher(
    subtask: str,
    researcher_id: int,
    client: AsyncOpenAI,
    tracker: TokenTracker,
) -> Message:
    agent_name = f"researcher_{researcher_id}"
    researcher = Agent(
        name=agent_name,
        role="researcher",
        system_prompt=RESEARCHER_PROMPT,
        model="gpt-5.4",
    )

    try:
        result = await asyncio.wait_for(
            call_agent(researcher, subtask, client, tracker),
            timeout=30        
        )
    except asyncio.TimeoutError:
        result = f"[TIMEOUT] Researcher {researcher_id} timed out on: {subtask}"

    return Message(
        from_agent=agent_name,
        to_agent="coordinator",
        type="research_result",
        content=result,
        metadata={"subtask": subtask},
    )


# - Takes a list of subtasks
# - Runs all researchers in parallel using asyncio.gather()
# - Returns list of Messages
async def run_all_researchers(
    subtasks: list[str],
    client: AsyncOpenAI,
    tracker: TokenTracker,
) -> list[Message]:
    tasks = [
        run_researcher(subtask, i + 1, client, tracker) for i, subtask in enumerate(subtasks)
    ]
    return await asyncio.gather(*tasks)

# - Create a Critic agent (use gpt-5.4 model)
# - Combine all research results into a single string
# - Call the critic with the combined research
# - Parse JSON response for approved/issues
# - Return a Message with type="critique" and metadata={"approved": bool}
async def run_critic(
    research_results: list[Message],
    client: AsyncOpenAI,
    tracker: TokenTracker,
) -> Message:
    critic = Agent(
        name="critic",
        role="critic",
        system_prompt=CRITIC_PROMPT,
        model="gpt-5.4",
    )

    combined = "\n\n".join(
        f"### {msg.from_agent}\nSubtask: {msg.metadata['subtask']}\n\n{msg.content}" for msg in research_results
    )

    result = await call_agent(critic, combined, client, tracker)
    data = json.loads(result)

    return Message(
        from_agent="critic",
        to_agent="coordinator",
        type="critique",
        content=result,
        metadata={"approved": data["approved"], "issues": data["issues"]},
    )


# - Create a Synthesizer agent (use gpt-4o model)
# - Combine all approved research into a single input
# - Call the synthesizer
# - Return a Message with type="final_report"
async def run_synthesizer(
    research_results: list[Message],
    query: str,
    client: AsyncOpenAI,
    tracker: TokenTracker,
) -> Message:
    synthesizer = Agent(
        name="synthesizer",
        role="synthesizer",
        system_prompt=SYNTHESIZER_PROMPT,
        model="gpt-4o",
    )

    combined = (
        f"Original question: {query}\n\n"
        + "\n\n".join(
            f"### {msg.from_agent}\nSubtask: {msg.metadata['subtask']}\n\n{msg.content}"
            for msg in research_results
        )
    )

    result = await call_agent(synthesizer, combined, client, tracker)

    return Message(
        from_agent="synthesizer",
        to_agent="user",
        type="final_report",
        content=result,
    )


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

MAX_REVISION_ROUNDS = 2

# TODO: implement run_pipeline()
# Orchestrate the full flow:
# 1. Decompose the query into subtasks (coordinator)
# 2. Run all researchers in parallel
# 3. Run the critic on combined results
# 4. If critic disapproves and rounds < MAX_REVISION_ROUNDS:
#    - Log the issues
#    - Run researchers again with the critic's feedback appended to their subtask
#    - Run the critic again
# 5. Run the synthesizer on the final approved research
# 6. Print the final report
# 7. Print token usage summary
async def run_pipeline(query: str):
    client = AsyncOpenAI()
    tracker = TokenTracker()

    print(f"\n{'='*60}")
    print(f"Research query: {query}")
    print(f"{'='*60}\n")

    start_time = time.time()

    # Step 1: Decompose
    print("[Coordinator] Decomposing query into subtasks…")
    subtasks = await decompose_query(query, client, tracker)
    for i, subtask in enumerate(subtasks, 1):
        print(f"  Subtasks {i}: {subtask}")

    # Step 2: Research (parallel)
    print("\n[Researchers] Running in parallel…")
    results = await run_all_researchers(subtasks, client, tracker)
    for msg in results:
        print(f"  {msg.from_agent} completed: {msg.metadata['subtask'][:50]}…")

    # Step 3-4: Critique loop
    for round_num in range(1, MAX_REVISION_ROUNDS + 1):
        print(f"\n[Critic] Review round {round_num}…")
        critique = await run_critic(results, client, tracker)
        approved = critique.metadata["approved"]

        if approved:
            print("  Research approved!")
            break

        # Log issues and re-run researchers with feedback
        issues = critique.metadata["issues"]
        print(f"  Issues found: {issues}")
        print("  Re-running researchers with feedback…")

        revised_subtasks = [
            f"{subtask}\n\nCritic feedback: {', '.join(issues)}"
            for subtask in subtasks
        ]
        results = await run_all_researchers(revised_subtasks, client, tracker)
    else:
        print("  Max revision rounds reached, proceeding anyway.")

    # Step 5: Synthesize
    print("\n[Synthesizer] Producing final report…")
    report = await run_synthesizer(results, query, client, tracker)
    print(report.content)

    # Step 6: Output
    elapsed = time.time() - start_time
    print(f"\n{'='*60}")
    print(f"Pipeline completed in {elapsed:.1f}s")
    print(f"{'='*60}\n")

    # Step 7: Token summary
    tracker.summary()


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main():
    if len(sys.argv) < 2:
        print("Usage: python pipeline.py \"Your research question here\"")
        sys.exit(1)

    query = sys.argv[1]
    asyncio.run(run_pipeline(query))


if __name__ == "__main__":
    main()
