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

    client.messages.create(model="", max_tokens="", system="", messages="")
        # TODO: Call the API and fill in the rest
        # Hint: use client.messages.create(model=..., max_tokens=..., system=..., messages=...)
        # Then extract text, append to messages, print, and track tokens


if __name__ == "__main__":
    chat()
