#!/usr/bin/env python3
"""
Code Review Assistant — CLI Demo
Uses YOUR trained TextCNN (Phase 3) to classify code, then uses an LLM
(Phase 6 few-shot) to explain WHY.

Usage:
  python code_review_cli.py                     # interactive menu
  python code_review_cli.py --file mycode.c     # review a file directly
  python code_review_cli.py --sample 1          # run a built-in sample

Requirements:
  pip install rich groq torch
  Set GROQ_API_KEY or ANTHROPIC_API_KEY in environment.

Project artefacts expected at (from Phase 2 & 3):
  data/processed/vocab.json
  data/processed/best_textcnn.pt
"""

import os
import re
import sys
import json
import math
import time
import argparse
import textwrap
from pathlib import Path

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.syntax import Syntax
from rich.prompt import Prompt, Confirm
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.text import Text
from rich.rule import Rule
from rich import box

console = Console()

# ─────────────────────────────────────────────────────────────────────────────
# Few-shot prompt  (mirrors phase6_llm_prompting.py FEW_SHOT_TEMPLATE)
# ─────────────────────────────────────────────────────────────────────────────
FEW_SHOT_TEMPLATE = """\
You are an expert code reviewer specialising in C/C++ security vulnerabilities and bugs.
Here are three examples of expert code reviews:

--- Example 1 (BUGGY) ---
Code:
```c
int read_input(char *buf) {{
    gets(buf);
    return strlen(buf);
}}
```
Review: Uses the unsafe gets() function, which performs no bounds checking and is a \
classic stack buffer overflow vulnerability. Replace with fgets(buf, MAX_SIZE, stdin).

--- Example 2 (CLEAN) ---
Code:
```c
int safe_div(int a, int b) {{
    if (b == 0) return -1;
    return a / b;
}}
```
Review: The function correctly guards against division by zero. No issues found.

--- Example 3 (BUGGY) ---
Code:
```c
char *dup_str(const char *s) {{
    char *p = malloc(strlen(s));
    strcpy(p, s);
    return p;
}}
```
Review: Off-by-one error - malloc(strlen(s)) omits the null terminator byte. \
Should be malloc(strlen(s) + 1). Return value of malloc is not checked for NULL.

Now review the following code. The binary classifier already predicted it is {verdict}.
Explain WHY concisely. Respond ONLY as JSON (no markdown, no extra text):
{{
  "issues": [
    {{ "severity": "HIGH" or "MEDIUM" or "LOW", "title": "short title", "detail": "one-sentence explanation" }}
  ],
  "recommendation": "one actionable sentence"
}}
If the code is CLEAN, set issues to [].

Code:
```c
{code}
```
"""

# ─────────────────────────────────────────────────────────────────────────────
# Built-in samples
# ─────────────────────────────────────────────────────────────────────────────
SAMPLES = {
    "1": {
        "name": "vuln_strcpy.c",
        "desc": "Unsafe strcpy - buffer overflow",
        "code": "void copy_user_input(char *dst, char *src) {\n    strcpy(dst, src);\n    printf(\"Copied: %s\\n\", dst);\n}",
    },
    "2": {
        "name": "vuln_malloc.c",
        "desc": "Off-by-one malloc + no NULL check",
        "code": "char *dup_str(const char *s) {\n    char *p = malloc(strlen(s));\n    strcpy(p, s);\n    return p;\n}",
    },
    "3": {
        "name": "vuln_gets.c",
        "desc": "Classic gets() vulnerability",
        "code": "void get_username(char *buf) {\n    printf(\"Enter username: \");\n    gets(buf);\n}",
    },
    "4": {
        "name": "safe_div.c",
        "desc": "Safe division with zero-check",
        "code": "int safe_div(int a, int b) {\n    if (b == 0) return -1;\n    return a / b;\n}",
    },
    "5": {
        "name": "safe_read.c",
        "desc": "Bounds-safe file read",
        "code": "void read_safe(FILE *fp, char *buf, size_t n) {\n    if (!fp || !buf || n == 0) return;\n    fread(buf, 1, n - 1, fp);\n    buf[n-1] = '\\0';\n}",
    },
}

# ─────────────────────────────────────────────────────────────────────────────
# TextCNN inference
# ─────────────────────────────────────────────────────────────────────────────
_cnn_model = None
_cnn_vocab = None

def _find_artefacts():
    candidates = [Path("data/processed"), Path("../data/processed")]
    for base in candidates:
        vocab = base / "vocab.json"
        model = base / "best_textcnn.pt"
        if vocab.exists() and model.exists():
            return vocab, model
    raise FileNotFoundError(
        "Could not find data/processed/vocab.json and data/processed/best_textcnn.pt\n"
        "Run from your project root after completing Phase 2 & 3."
    )

def load_cnn():
    global _cnn_model, _cnn_vocab
    if _cnn_model is not None:
        return
    import torch
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from src.textcnn_model import TextCNN, get_device

    vocab_path, model_path = _find_artefacts()
    with open(vocab_path) as f:
        _cnn_vocab = json.load(f)

    device = get_device("auto")
    _cnn_model = TextCNN(vocab_size=len(_cnn_vocab))
    _cnn_model.load_state_dict(
        torch.load(model_path, weights_only=True, map_location=device)
    )
    _cnn_model.to(device).eval()
    _cnn_model._device = device

def _preprocess(code: str, max_len: int = 512):
    import torch
    code = re.sub(r'/\*.*?\*/', '', code, flags=re.DOTALL)
    code = re.sub(r'//.*$', '', code, flags=re.MULTILINE)
    code = ' '.join(code.split()).strip()
    tokens = re.findall(r'\b\w+\b|[^\s\w]', code)
    unk = _cnn_vocab.get('<UNK>', 1)
    ids = [_cnn_vocab.get(t, unk) for t in tokens]
    ids = ids[:max_len]
    ids += [0] * (max_len - len(ids))
    return torch.tensor(ids, dtype=torch.long).unsqueeze(0)

def cnn_predict(code: str) -> dict:
    import torch
    load_cnn()
    inp = _preprocess(code).to(_cnn_model._device)
    with torch.no_grad():
        logit = _cnn_model(inp).item()
    prob = 1 / (1 + math.exp(-logit))
    return {
        "verdict":    "BUGGY" if prob >= 0.5 else "CLEAN",
        "confidence": prob if prob >= 0.5 else 1 - prob,
        "prob_buggy": prob,
        "logit":      logit,
    }

# ─────────────────────────────────────────────────────────────────────────────
# LLM explanation
# ─────────────────────────────────────────────────────────────────────────────
def llm_explain(code: str, verdict: str) -> dict:
    prompt = FEW_SHOT_TEMPLATE.format(code=code, verdict=verdict)
    groq_key      = os.environ.get("GROQ_API_KEY", "").strip()
    anthropic_key = os.environ.get("ANTHROPIC_API_KEY", "").strip()

    if groq_key:
        try:
            from groq import Groq
            client = Groq(api_key=groq_key)
            resp = client.chat.completions.create(
                model=os.environ.get("GROQ_MODEL", "llama-3.1-8b-instant"),
                messages=[
                    {"role": "system", "content": "You are an expert C/C++ security reviewer. Respond ONLY with the JSON object. No markdown, no extra text."},
                    {"role": "user", "content": prompt},
                ],
                max_tokens=500,
                temperature=0.2,
            )
            raw = resp.choices[0].message.content.strip()
            return json.loads(raw.replace("```json", "").replace("```", "").strip())
        except Exception as e:
            console.print(f"[yellow]  Groq failed ({e}), trying Anthropic...[/yellow]")

    if anthropic_key:
        import urllib.request
        payload = json.dumps({
            "model": "claude-haiku-4-5-20251001",
            "max_tokens": 500,
            "system": "You are an expert C/C++ security reviewer. Respond ONLY with the JSON object. No markdown, no extra text.",
            "messages": [{"role": "user", "content": prompt}],
        }).encode()
        req = urllib.request.Request(
            "https://api.anthropic.com/v1/messages",
            data=payload,
            headers={
                "x-api-key": anthropic_key,
                "anthropic-version": "2023-06-01",
                "content-type": "application/json",
            },
        )
        with urllib.request.urlopen(req) as r:
            data = json.load(r)
        raw = data["content"][0]["text"].strip()
        return json.loads(raw.replace("```json", "").replace("```", "").strip())

    raise RuntimeError("No API key found. Set GROQ_API_KEY or ANTHROPIC_API_KEY.")

# ─────────────────────────────────────────────────────────────────────────────
# Rendering
# ─────────────────────────────────────────────────────────────────────────────
SEV_COLOR = {"HIGH": "red", "MEDIUM": "yellow", "LOW": "green"}
SEV_ICON  = {"HIGH": "●", "MEDIUM": "◆", "LOW": "○"}

def render_header():
    console.print()
    t = Text()
    t.append("  CODE REVIEW ASSISTANT  ", style="bold white on #0d1b2a")
    t.append("  TextCNN + few-shot LLM  ", style="bold cyan on #1b2838")
    console.print(Panel(t, border_style="bright_blue", padding=(0, 2)))
    console.print(
        "  [dim]Your trained TextCNN (Phase 3) classifies the code.\n"
        "  The LLM (Phase 6 few-shot) then explains why.[/dim]\n"
    )

def render_sample_menu():
    table = Table(box=box.SIMPLE, show_header=True, header_style="bold cyan", border_style="dim", padding=(0, 2))
    table.add_column("#",    style="bold white", width=3)
    table.add_column("File", style="green")
    table.add_column("Description", style="dim")
    for k, s in SAMPLES.items():
        table.add_row(k, s["name"], s["desc"])
    table.add_row("p", "[yellow]paste[/yellow]",  "Paste your own code")
    table.add_row("f", "[yellow]file[/yellow]",   "Load from a file path")
    table.add_row("q", "[red]quit[/red]",          "Exit")
    console.print(Panel(table, title="[bold]Select a sample[/bold]", border_style="bright_blue"))

def render_code(code: str, filename: str = "code.c"):
    syntax = Syntax(code, "c", theme="monokai", line_numbers=True, word_wrap=True)
    console.print(Panel(syntax, title=f"[bold cyan]{filename}[/bold cyan]", border_style="dim"))

def render_cnn_result(pred: dict):
    verdict    = pred["verdict"]
    prob_buggy = pred["prob_buggy"]
    prob_clean = 1 - prob_buggy
    color = "red" if verdict == "BUGGY" else "green"
    icon  = "✗" if verdict == "BUGGY" else "✓"

    # Two-sided bar: red (buggy) on left, green (clean) on right
    # The | marker shows where the model sits on the 0-1 scale
    bar_width = 30
    marker_pos = round(prob_buggy * bar_width)
    bar = ""
    for i in range(bar_width):
        if i == marker_pos:
            bar += "|"
        elif i < marker_pos:
            bar += "█"
        else:
            bar += "░"

    console.print()
    console.print(Rule("[bold]Step 1 — TextCNN Prediction[/bold]", style="dim"))
    console.print()
    t = Text()
    t.append(f"  {icon}  {verdict}  ", style=f"bold {color} on {'#3d0000' if verdict == 'BUGGY' else '#003d00'}")
    console.print(t)
    console.print(f"\n  [red]BUGGY[/red] [red]{bar[:marker_pos]}[/red][bold white]{bar[marker_pos]}[/bold white][green]{bar[marker_pos+1:]}[/green] [green]CLEAN[/green]")
    console.print(f"         [red]P(buggy) = {prob_buggy*100:.1f}%[/red]   [green]P(clean) = {prob_clean*100:.1f}%[/green]")
    console.print(f"         logit: {pred['logit']:+.3f}   {'(negative = leans CLEAN)' if pred['logit'] < 0 else '(positive = leans BUGGY)'}\n")

def render_llm_explanation(expl: dict, elapsed: float):
    issues = expl.get("issues", [])
    rec    = expl.get("recommendation", "")

    console.print(Rule("[bold]Step 2 — LLM Explanation (few-shot)[/bold]", style="dim"))
    console.print(f"  [dim]Response time: {elapsed:.1f}s[/dim]\n")

    if issues:
        console.print(f"  [bold white]Issues ({len(issues)})[/bold white]")
        for iss in issues:
            sev  = iss.get("severity", "LOW")
            col  = SEV_COLOR.get(sev, "white")
            icon = SEV_ICON.get(sev, "○")
            console.print(f"  [{col}]{icon} {sev}[/{col}]  [bold]{iss.get('title','')}[/bold]")
            for line in textwrap.wrap(iss.get("detail", ""), width=70):
                console.print(f"       [dim]{line}[/dim]")
            console.print()
    else:
        console.print("  [green]No issues identified.[/green]\n")

    if rec:
        console.print(Rule(style="dim"))
        console.print("  [bold cyan]Recommendation[/bold cyan]")
        for line in textwrap.wrap(rec, width=74):
            console.print(f"  {line}")
    console.print()

# ─────────────────────────────────────────────────────────────────────────────
# Main review flow
# ─────────────────────────────────────────────────────────────────────────────
def review_code(code: str, filename: str = "snippet.c"):
    render_code(code, filename)

    # Step 1: TextCNN
    cnn_error = None
    with Progress(SpinnerColumn(spinner_name="dots", style="cyan"), TextColumn("[cyan]Running TextCNN...[/cyan]"), transient=True) as p:
        p.add_task("cnn", total=None)
        try:
            pred = cnn_predict(code)
        except FileNotFoundError as e:
            cnn_error = str(e)
            pred = None
        except Exception as e:
            cnn_error = f"CNN inference failed: {e}"
            pred = None

    if cnn_error:
        console.print(f"\n  [red]TextCNN unavailable:[/red] {cnn_error}")
        console.print("  [dim]Falling back to LLM-only review.[/dim]\n")
        verdict = "UNKNOWN"
    else:
        render_cnn_result(pred)
        verdict = pred["verdict"]

    # Step 2: LLM explanation
    with Progress(SpinnerColumn(spinner_name="dots2", style="magenta"), TextColumn("[magenta]Generating LLM explanation (few-shot)...[/magenta]"), transient=True) as p:
        p.add_task("llm", total=None)
        t0 = time.time()
        try:
            expl = llm_explain(code, verdict)
        except Exception as e:
            console.print(f"\n  [red]LLM error:[/red] {e}\n")
            console.print("  [dim]Set GROQ_API_KEY (free at console.groq.com) or ANTHROPIC_API_KEY.[/dim]\n")
            return
        elapsed = time.time() - t0

    render_llm_explanation(expl, elapsed)

# ─────────────────────────────────────────────────────────────────────────────
# Interactive loop
# ─────────────────────────────────────────────────────────────────────────────
def interactive():
    render_header()
    while True:
        render_sample_menu()
        choice = Prompt.ask("  [bold cyan]>[/bold cyan]").strip().lower()

        if choice == "q":
            console.print("\n  [dim]Bye![/dim]\n")
            break
        elif choice in SAMPLES:
            s = SAMPLES[choice]
            review_code(s["code"], s["name"])
            if not Confirm.ask("  Review another?", default=True):
                console.print("\n  [dim]Bye![/dim]\n")
                break
        elif choice == "p":
            console.print("\n  [dim]Paste your C/C++ code. Enter a blank line twice when done.[/dim]\n")
            lines = []
            while True:
                try:
                    line = input()
                except EOFError:
                    break
                if line == "" and lines and lines[-1] == "":
                    break
                lines.append(line)
            code = "\n".join(lines).strip()
            if code:
                review_code(code, "custom.c")
            else:
                console.print("  [yellow]No code entered.[/yellow]\n")
        elif choice == "f":
            path = Prompt.ask("  File path").strip()
            p = Path(path)
            if not p.exists():
                console.print(f"  [red]File not found:[/red] {path}\n")
            else:
                review_code(p.read_text(), p.name)
        else:
            console.print("  [red]Invalid choice.[/red]\n")

# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Code Review Assistant - TextCNN + LLM demo")
    parser.add_argument("--file",   "-f", help="Path to a C/C++ file to review")
    parser.add_argument("--sample", "-s", choices=list(SAMPLES.keys()), help="Run a built-in sample (1-5)")
    args = parser.parse_args()

    if args.file:
        render_header()
        p = Path(args.file)
        if not p.exists():
            console.print(f"[red]File not found:[/red] {args.file}")
            sys.exit(1)
        review_code(p.read_text(), p.name)
    elif args.sample:
        render_header()
        s = SAMPLES[args.sample]
        review_code(s["code"], s["name"])
    else:
        interactive()

if __name__ == "__main__":
    main()