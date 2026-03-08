# Module 01 — LLM API Fundamentals

**Goal:** Go from zero to making real API calls, understanding the message format, and building a working CLI chat app.

**Time:** ~1 day

---

## What you'll learn

- How the messages API works (roles, turns, context window)
- System prompts and how to use them effectively
- Temperature, max_tokens, and when each setting matters
- Structured outputs (JSON mode)
- Basic error handling and retries

---

## Concepts

### The messages format

Every LLM API uses a list of messages with roles:

```python
messages = [
    {"role": "user",      "content": "What is the capital of France?"},
    {"role": "assistant", "content": "Paris."},
    {"role": "user",      "content": "And Germany?"},
]
```

The model sees the full list every call. There is no memory — you pass the history manually. This is the most important thing to understand about LLMs.

### System prompts

A system prompt sets the rules before the conversation starts. It defines persona, format, constraints. Always use one in production.

```python
system = "You are a concise assistant. Answer in one sentence unless asked for detail."
```

### Key parameters

| Parameter | What it does | Typical value |
|-----------|-------------|---------------|
| `model` | Which model to use | `claude-sonnet-4-6` |
| `max_tokens` | Hard cap on output length | 1024–4096 |
| `temperature` | Randomness (0 = deterministic, 1 = creative) | 0 for factual, 0.7 for creative |
| `system` | System prompt | Always set this |

### Structured outputs

Force JSON output by telling the model in the system prompt and using a prefix:

```python
system = "Always respond in valid JSON. No prose outside the JSON object."
```

For critical parsing, use Anthropic's tool use or OpenAI's `response_format={"type": "json_object"}`.

---

## Project: CLI Chat App

Build a command-line chat application with:

1. Multi-turn conversation (maintains history)
2. A configurable system prompt
3. Ability to switch between models
4. Structured output mode (user can ask for JSON responses)
5. Token usage tracking per turn

### Requirements

```
Features:
- Interactive REPL loop
- /system <prompt> — change system prompt mid-conversation
- /clear — reset conversation history
- /model <name> — switch model
- /json — toggle JSON output mode
- /usage — show total tokens used this session
- /quit — exit
```

### Starter code

```python
# chat.py
import os
from anthropic import Anthropic
from dotenv import load_dotenv

load_dotenv()

client = Anthropic()

def chat():
    messages = []
    system = "You are a helpful assistant."
    model = "claude-sonnet-4-6"
    json_mode = False
    total_input_tokens = 0
    total_output_tokens = 0

    print(f"Chat started. Model: {model}")
    print("Commands: /system, /clear, /model, /json, /usage, /quit\n")

    while True:
        user_input = input("You: ").strip()
        if not user_input:
            continue

        # Handle commands
        if user_input.startswith("/"):
            parts = user_input.split(" ", 1)
            cmd = parts[0]
            arg = parts[1] if len(parts) > 1 else ""

            if cmd == "/quit":
                print("Bye.")
                break
            elif cmd == "/clear":
                messages = []
                print("History cleared.")
                continue
            elif cmd == "/system":
                system = arg
                print(f"System prompt updated: {system}")
                continue
            elif cmd == "/model":
                model = arg
                print(f"Model switched to: {model}")
                continue
            elif cmd == "/json":
                json_mode = not json_mode
                print(f"JSON mode: {'on' if json_mode else 'off'}")
                continue
            elif cmd == "/usage":
                print(f"Tokens used — Input: {total_input_tokens}, Output: {total_output_tokens}")
                continue
            else:
                print(f"Unknown command: {cmd}")
                continue

        # Add user message
        messages.append({"role": "user", "content": user_input})

        # Build system prompt
        active_system = system
        if json_mode:
            active_system += " Always respond in valid JSON."

        # Call the API
        # TODO: implement this
        response = None  # replace with actual API call

        # TODO: extract text, update history, print response, track tokens

if __name__ == "__main__":
    chat()
```

### Your task

1. Fill in the API call using `client.messages.create()`
2. Extract the response text from the response object
3. Append the assistant's response to `messages`
4. Print the response
5. Track tokens from `response.usage`

### Solution hint

```python
response = client.messages.create(
    model=model,
    max_tokens=1024,
    system=active_system,
    messages=messages,
)
text = response.content[0].text
messages.append({"role": "assistant", "content": text})
print(f"\nAssistant: {text}\n")
total_input_tokens += response.usage.input_tokens
total_output_tokens += response.usage.output_tokens
```

---

## Stretch goals

- Add color to the terminal output (use `rich` library)
- Save/load conversation history to a JSON file
- Add a `/cost` command that estimates API cost based on token usage

---

## Key questions to answer before moving on

1. What happens to the context window as a conversation gets longer?
2. Why does temperature=0 not guarantee identical outputs every time?
3. What's the difference between `max_tokens` and the context window limit?
4. When would you choose structured JSON output vs. free text?

---

## Resources

- [Anthropic API docs — Messages](https://docs.anthropic.com/en/api/messages)
- [Anthropic Python SDK](https://github.com/anthropics/anthropic-sdk-python)
- [Claude models overview](https://docs.anthropic.com/en/docs/about-claude/models)

---

**When done:** Mark Module 01 as shipped in the root README, commit your code, and move to [Module 02](../02-prompt-engineering/).
