import json
from datetime import datetime
from anthropic import Anthropic
from dotenv import load_dotenv
from rich.console import Console
from rich.markdown import Markdown
from rich.prompt import Prompt

load_dotenv()

client = Anthropic()
console = Console()


def chat():
    messages = []
    system = "You are a helpful assistant."
    model = "claude-sonnet-4-6"
    json_mode = False
    total_input_tokens = 0
    total_output_tokens = 0
    context_tokens = 0  # input tokens from the most recent call = current context size

    context_limits = {
        "claude-opus-4-6":          200_000,
        "claude-sonnet-4-6":        200_000,
        "claude-haiku-4-5-20251001": 200_000,
    }

    # Pricing per 1M tokens (input, output) in USD
    pricing = {
        "claude-opus-4-6":           (15.00, 75.00),
        "claude-sonnet-4-6":          (3.00, 15.00),
        "claude-haiku-4-5-20251001":  (0.80,  4.00),
    }

    console.print(f"[bold green]Chat started.[/bold green] Model: [cyan]{model}[/cyan]")
    console.print("[dim]Commands: /system, /clear, /model, /json, /usage, /context, /cost, /save, /load, /quit[/dim]\n")

    while True:
        user_input = Prompt.ask("[bold blue]You[/bold blue]").strip()
        if not user_input:
            continue

        # Handle commands
        if user_input.startswith("/"):
            parts = user_input.split(" ", 1)
            cmd = parts[0]
            arg = parts[1] if len(parts) > 1 else ""

            if cmd == "/quit":
                console.print("[bold red]Bye.[/bold red]")
                break
            elif cmd == "/clear":
                messages = []
                context_tokens = 0
                console.print("[yellow]History cleared.[/yellow]")
                continue
            elif cmd == "/system":
                system = arg
                console.print(f"[yellow]System prompt updated:[/yellow] {system}")
                continue
            elif cmd == "/model":
                model = arg
                console.print(f"[yellow]Model switched to:[/yellow] [cyan]{model}[/cyan]")
                continue
            elif cmd == "/json":
                json_mode = not json_mode
                state = "[green]on[/green]" if json_mode else "[red]off[/red]"
                console.print(f"[yellow]JSON mode:[/yellow] {state}")
                continue
            elif cmd == "/usage":
                console.print(f"[dim]Tokens used — Input: [cyan]{total_input_tokens}[/cyan], Output: [cyan]{total_output_tokens}[/cyan][/dim]")
                continue
            elif cmd == "/context":
                limit = context_limits.get(model, 200_000)
                pct = (context_tokens / limit) * 100 if context_tokens else 0
                console.print(f"[dim]Context: [cyan]{context_tokens:,}[/cyan] / [cyan]{limit:,}[/cyan] tokens ([yellow]{pct:.1f}%[/yellow] full)[/dim]")
                continue
            elif cmd == "/cost":
                in_price, out_price = pricing.get(model, (None, None))
                if in_price is None:
                    console.print(f"[yellow]No pricing data for[/yellow] [cyan]{model}[/cyan]")
                else:
                    in_cost = (total_input_tokens / 1_000_000) * in_price
                    out_cost = (total_output_tokens / 1_000_000) * out_price
                    total_cost = in_cost + out_cost
                    console.print(
                        f"[bold]Cost estimate[/bold] ([cyan]{model}[/cyan])\n"
                        f"  Input:  [dim]{total_input_tokens:,} tokens[/dim] × ${in_price}/1M = [green]${in_cost:.6f}[/green]\n"
                        f"  Output: [dim]{total_output_tokens:,} tokens[/dim] × ${out_price}/1M = [green]${out_cost:.6f}[/green]\n"
                        f"  [bold]Total:  [green]${total_cost:.6f}[/green][/bold]"
                    )
                continue
            elif cmd == "/save":
                filename = arg or f"chat_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
                data = {"model": model, "system": system, "messages": messages}
                with open(filename, "w") as f:
                    json.dump(data, f, indent=2)
                console.print(f"[green]Saved to[/green] [cyan]{filename}[/cyan] [dim]({len(messages)} messages)[/dim]")
                continue
            elif cmd == "/load":
                if not arg:
                    console.print("[red]Usage:[/red] /load <filename>")
                    continue
                try:
                    with open(arg) as f:
                        data = json.load(f)
                    messages = data.get("messages", [])
                    model = data.get("model", model)
                    system = data.get("system", system)
                    console.print(f"[green]Loaded[/green] [cyan]{arg}[/cyan] [dim]({len(messages)} messages, model: {model})[/dim]")
                except FileNotFoundError:
                    console.print(f"[red]File not found:[/red] {arg}")
                except json.JSONDecodeError:
                    console.print(f"[red]Invalid JSON:[/red] {arg}")
                continue
            else:
                console.print(f"[red]Unknown command:[/red] {cmd}")
                continue

        # Add user message
        messages.append({"role": "user", "content": user_input})

        # Build system prompt
        active_system = system
        if json_mode:
            active_system += " Always respond in valid JSON."

        response = client.messages.create(
            model=model,
            max_tokens=1024,
            system=active_system,
            messages=messages,
        )
        text = response.content[0].text
        messages.append({"role": "assistant", "content": text})
        console.print("\n[bold green]Assistant:[/bold green]")
        console.print(Markdown(text))
        console.print()
        context_tokens = response.usage.input_tokens
        total_input_tokens += response.usage.input_tokens
        total_output_tokens += response.usage.output_tokens
        console.print(f"[dim]Tokens — Input: {total_input_tokens}, Output: {total_output_tokens}[/dim]")

if __name__ == "__main__":
    chat()
