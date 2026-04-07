# Module 09 — Multi-agent Systems

**Goal:** Build a multi-agent research pipeline where specialized agents collaborate to research a topic, synthesize findings, and produce a structured report — all orchestrated by a coordinator agent.

**Time:** ~2 days

---

## Setup & running

```bash
pip install openai python-dotenv

# Run the multi-agent pipeline
python pipeline.py "What are the key differences between RAG and fine-tuning for enterprise use cases?"
```

---

## What you'll learn

- Why multi-agent architectures exist and when they beat single-agent systems
- Agent roles: coordinator, specialist, critic, synthesizer
- Orchestration patterns: sequential, parallel, and hierarchical
- Structured handoffs between agents (message passing, shared state)
- Error handling and retry logic in multi-agent flows
- How to keep costs under control when multiple agents are calling LLMs

---

## Concepts

### Why multi-agent?

A single LLM call can do a lot. A single agent (LLM + tools) can do more. But some tasks have natural decomposition points:

- **Research** requires gathering from multiple angles, then synthesizing
- **Code review** benefits from separate "find bugs" and "check style" passes
- **Content creation** improves when one agent drafts and another critiques

Multi-agent systems split work across specialized agents. Each agent has a focused system prompt, its own tools, and a narrow responsibility. The coordinator decides what to delegate and when the job is done.

The key insight: **a focused agent with a narrow prompt outperforms a general agent with a kitchen-sink prompt.** Specialization reduces hallucination and improves output quality.

### Orchestration patterns

**Sequential (pipeline):**
```
Researcher → Analyst → Writer → Editor
```
Each agent's output feeds the next. Simple, predictable, easy to debug. Use this when each step genuinely depends on the previous one.

**Parallel (fan-out / fan-in):**
```
Coordinator → [Researcher A, Researcher B, Researcher C] → Synthesizer
```
Multiple agents work simultaneously on independent subtasks. The coordinator fans out work and a synthesizer fans results back in. Use this when subtasks are independent — cuts wall-clock time proportionally.

**Hierarchical (manager / worker):**
```
Manager
├── Team Lead A → [Worker 1, Worker 2]
└── Team Lead B → [Worker 3, Worker 4]
```
Nested delegation. The manager breaks the problem down, team leads break it further. Use this for complex tasks that need multiple levels of decomposition.

### Agent roles in this project

You'll implement four agent roles:

| Role | Responsibility | System prompt focus |
|------|---------------|-------------------|
| **Coordinator** | Breaks the query into research subtasks, delegates, decides when done | Task decomposition, delegation, quality gates |
| **Researcher** | Investigates one specific angle of the topic | Depth over breadth, cite reasoning, flag uncertainty |
| **Critic** | Reviews research for gaps, contradictions, unsupported claims | Skeptical, specific, actionable feedback |
| **Synthesizer** | Combines all research into a coherent final report | Structure, clarity, no new claims beyond what researchers found |

### Message passing between agents

Agents communicate through structured messages. Each message has:

```python
{
    "from_agent": "researcher_1",
    "to_agent": "coordinator",
    "type": "research_result",       # research_result | critique | task_assignment | final_report
    "content": "...",
    "metadata": {"subtask": "...", "confidence": 0.85}
}
```

This structure lets you:
- Log every handoff for debugging
- Retry failed steps without re-running the whole pipeline
- Trace how the final output was built

### Parallel execution with asyncio

When researchers work on independent subtasks, run them in parallel:

```python
import asyncio

async def run_researchers(subtasks: list[str], client) -> list[dict]:
    tasks = [research(subtask, client) for subtask in subtasks]
    return await asyncio.gather(*tasks)
```

This is where you see real time savings. Three researchers running in parallel take ~1x the time instead of ~3x.

### Cost control

Multi-agent = multiple LLM calls = costs add up fast. Strategies:

- **Use cheap models for simple roles.** The critic doesn't need GPT-4o — GPT-4o-mini can spot gaps just fine.
- **Limit rounds.** Set a max number of research→critique→revision cycles (e.g., 2).
- **Track tokens.** Log input/output tokens per agent per run.
- **Short-circuit.** If the coordinator sees the research is sufficient after the first round, skip the critique cycle.

### Error handling

Multi-agent pipelines have more failure points. Handle them:

- **Retry with backoff** for rate limits and transient API errors
- **Timeout per agent** — don't let one slow researcher block everything
- **Fallback** — if a researcher fails after retries, the coordinator notes the gap and continues
- **Structured errors** — agents return error messages in the same message format, so the coordinator can reason about them

---

## Project: Multi-agent Research Pipeline

Build a CLI tool that takes a research question and produces a structured report using four collaborating agents.

### Architecture

```
User query
    │
    ▼
Coordinator ──→ breaks query into 3 subtasks
    │
    ▼
[Researcher 1, Researcher 2, Researcher 3]  ← parallel via asyncio
    │
    ▼
Coordinator ──→ collects results
    │
    ▼
Critic ──→ reviews combined research, flags issues
    │
    ▼
Coordinator ──→ decides: revise or proceed?
    │            (max 2 revision rounds)
    ▼
Synthesizer ──→ produces final structured report
    │
    ▼
Final report (printed to stdout)
```

### Requirements

1. **Coordinator agent** that decomposes a query into 3 research subtasks
2. **Researcher agents** that run in parallel (asyncio) and return structured findings
3. **Critic agent** that reviews the combined research and returns actionable feedback
4. **Synthesizer agent** that produces the final report from approved research
5. **Structured message passing** between all agents (dict-based, logged to console)
6. **Configurable models** — coordinator and synthesizer use GPT-4o, researchers and critic use GPT-4o-mini
7. **Token tracking** — print total tokens used (input + output) per agent and grand total at the end
8. **Max 2 revision rounds** — if the critic still has issues after 2 rounds, proceed anyway
9. **Timeout handling** — individual researcher timeout of 30 seconds
10. **CLI interface** — takes the research question as a command-line argument

### Starter files

- `pipeline.py` — main orchestrator with all agent functions stubbed out

### Your task

1. Start with the `Agent` dataclass and `call_agent` function — get a single LLM call working
2. Implement the `Coordinator.decompose` method — given a query, return 3 subtasks
3. Implement `run_researcher` — call the LLM with the researcher system prompt
4. Make researchers run in parallel with `asyncio.gather`
5. Implement `run_critic` — review combined research, return structured feedback
6. Implement the coordinator's revision loop (max 2 rounds)
7. Implement `run_synthesizer` — produce the final report
8. Add token tracking across all calls
9. Wire everything together in `main()`
10. Test with different research questions

### Hints

<details>
<summary>Hint for step 1</summary>
The `call_agent` function should take an `Agent` (with system prompt and model) and a user message. Use `client.chat.completions.create()`. Return both the response content and token usage.
</details>

<details>
<summary>Hint for step 2</summary>
The coordinator's system prompt should instruct it to return exactly 3 subtasks as a JSON array of strings. Use `response_format={"type": "json_object"}` to enforce JSON output.
</details>

<details>
<summary>Hint for step 4</summary>
Use `asyncio.gather(*tasks)` with `asyncio.wait_for(task, timeout=30)` for individual timeouts. Wrap each researcher call in a try/except to handle timeouts gracefully.
</details>

<details>
<summary>Hint for step 6</summary>
The critic returns a JSON object with `{"approved": bool, "issues": [...]}`. If not approved and rounds < max, send the issues back to the researchers for a targeted follow-up.
</details>

<details>
<summary>Hint for step 8</summary>
Create a simple `TokenTracker` class with `add(agent_name, input_tokens, output_tokens)` and `summary()` methods. Call `add()` inside `call_agent` every time you get a response.
</details>

---

## Stretch goals

- **Add web search** — give researchers a tool (function calling) to search the web via a search API, so they produce grounded results instead of relying on training data
- **Streaming output** — stream the synthesizer's final report to stdout in real-time
- **Conversation memory** — let the user ask follow-up questions that feed back into the pipeline with previous context

---

## Key questions

- When does a multi-agent system outperform a single agent with a long prompt? When does it not?
- How do you decide which model to assign to each agent role?
- What are the failure modes unique to multi-agent systems that don't exist in single-agent setups?
- How would you test a multi-agent pipeline? What would your eval look like?
- If you had to add a fifth agent role, what would it be and why?

---

## Resources

- [OpenAI Chat Completions API](https://platform.openai.com/docs/guides/chat-completions)
- [Python asyncio docs](https://docs.python.org/3/library/asyncio.html)
- [Andrew Ng on Agentic Design Patterns](https://www.deeplearning.ai/the-batch/how-agents-can-improve-llm-performance/)
- [LangGraph multi-agent docs](https://langchain-ai.github.io/langgraph/) (for reference — you're building from scratch first)
