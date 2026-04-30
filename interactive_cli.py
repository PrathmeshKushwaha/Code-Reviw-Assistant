#!/usr/bin/env python3
"""
<<<<<<< HEAD
Code Review Assistant — Interactive CLI Demo
=======
Code Review Assistant — CLI Demo
>>>>>>> 786234e1af7e83a1d2dfc59296b5812fe59318d9
Uses YOUR trained TextCNN (Phase 3) to classify code, then uses an LLM
(Phase 6 few-shot) to explain WHY.

Usage:
<<<<<<< HEAD
  python interactive_cli.py                          # interactive menu
  python interactive_cli.py --file mycode.c          # review a file directly
  python interactive_cli.py --sample 1               # run a built-in sample
  python interactive_cli.py --provider groq          # force a specific provider
  python interactive_cli.py --provider ollama        # use local Ollama server

Providers (tried in order if --provider auto):
  groq     → GROQ_API_KEY  (recommended, free at console.groq.com)
  hf       → HF_API_KEY
  ollama   → local Ollama server (ollama serve)
  local    → HuggingFace transformers on GPU/CPU (no key needed, slow)
  anthropic→ ANTHROPIC_API_KEY

Environment variables:
  GROQ_API_KEY, GROQ_MODEL      (default: llama-3.1-8b-instant)
  HF_API_KEY,   HF_MODEL        (default: Salesforce/codet5p-220m)
  OLLAMA_URL,   OLLAMA_MODEL    (default: http://localhost:11434, qwen2.5:0.5b)
  LOCAL_MODEL                   (default: Qwen/Qwen2.5-0.5B-Instruct)
  ANTHROPIC_API_KEY
  API_PROVIDER                  (groq|hf|ollama|local|anthropic|auto)
=======
  python code_review_cli.py                     # interactive menu
  python code_review_cli.py --file mycode.c     # review a file directly
  python code_review_cli.py --sample 1          # run a built-in sample

Requirements:
  pip install rich groq torch
  Set GROQ_API_KEY or ANTHROPIC_API_KEY in environment.
>>>>>>> 786234e1af7e83a1d2dfc59296b5812fe59318d9

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
<<<<<<< HEAD
import hashlib
import argparse
import textwrap
import urllib.request
import urllib.error
from pathlib import Path
from typing import Optional

# ── Rich import guard ────────────────────────────────────────────────────────
try:
    from rich.console import Console
    from rich.panel import Panel
    from rich.table import Table
    from rich.syntax import Syntax
    from rich.prompt import Prompt, Confirm
    from rich.progress import Progress, SpinnerColumn, TextColumn
    from rich.text import Text
    from rich.rule import Rule
    from rich import box
except ImportError:
    print("ERROR: 'rich' is not installed. Run:  pip install rich")
    sys.exit(1)

console = Console()

def _groq_available() -> bool:
    """Check if groq package is installed AND API key is set"""
    if not os.environ.get("GROQ_API_KEY", "").strip():
        return False
    try:
        import groq
        return True
    except ImportError:
        return False


def _hf_available() -> bool:
    """Check if HF API key is set"""
    return bool(os.environ.get("HF_API_KEY", "").strip())

# ─────────────────────────────────────────────────────────────────────────────
# Cache (mirrors phase6_llm_prompting.py)
# ─────────────────────────────────────────────────────────────────────────────
OUTPUT_DIR = Path("outputs")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
CACHE_FILE = OUTPUT_DIR / "llm_cache.json"


def _cache_key(prompt: str, model: str, provider: str) -> str:
    return hashlib.md5(f"{provider}::{model}::{prompt}".encode()).hexdigest()


def _load_cache() -> dict:
    if CACHE_FILE.exists():
        try:
            with open(CACHE_FILE) as f:
                return json.load(f)
        except (json.JSONDecodeError, OSError):
            pass
    return {}


def _save_cache(cache: dict):
    try:
        with open(CACHE_FILE, "w") as f:
            json.dump(cache, f, indent=2)
    except OSError:
        pass  # non-fatal


=======
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

>>>>>>> 786234e1af7e83a1d2dfc59296b5812fe59318d9
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

<<<<<<< HEAD

def _find_artefacts() -> tuple[Path, Path]:
    """
    Search for vocab.json + best_textcnn.pt in order:
      1. configs/defaults.yaml (mirrors phase6 behaviour)
      2. Relative paths: data/processed, ../data/processed
    """
    # Try reading the project config first (same approach as phase6)
    try:
        import yaml
        cfg_path = Path("configs/defaults.yaml")
        if cfg_path.exists():
            with open(cfg_path) as f:
                cfg = yaml.safe_load(f)
            data_dir = Path(cfg["dataset"]["processed_dir"])
            vocab = data_dir / "vocab.json"
            model = data_dir / "best_textcnn.pt"
            if vocab.exists() and model.exists():
                return vocab, model
    except Exception:
        pass

    # Fallback: hardcoded relative paths
    for base in [Path("data/processed"), Path("../data/processed")]:
=======
def _find_artefacts():
    candidates = [Path("data/processed"), Path("../data/processed")]
    for base in candidates:
>>>>>>> 786234e1af7e83a1d2dfc59296b5812fe59318d9
        vocab = base / "vocab.json"
        model = base / "best_textcnn.pt"
        if vocab.exists() and model.exists():
            return vocab, model
<<<<<<< HEAD

    raise FileNotFoundError(
        "Could not find data/processed/vocab.json and data/processed/best_textcnn.pt\n"
        "Run from your project root after completing Phase 2 & 3, or check configs/defaults.yaml."
    )


=======
    raise FileNotFoundError(
        "Could not find data/processed/vocab.json and data/processed/best_textcnn.pt\n"
        "Run from your project root after completing Phase 2 & 3."
    )

>>>>>>> 786234e1af7e83a1d2dfc59296b5812fe59318d9
def load_cnn():
    global _cnn_model, _cnn_vocab
    if _cnn_model is not None:
        return
<<<<<<< HEAD

    import torch

    # Insert the project root (parent of scripts/) so src/ is importable
    # regardless of where the CLI is launched from.
    project_root = Path(__file__).resolve().parent
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

=======
    import torch
    sys.path.insert(0, str(Path(__file__).resolve().parent))
>>>>>>> 786234e1af7e83a1d2dfc59296b5812fe59318d9
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

<<<<<<< HEAD

def _preprocess(code: str, max_len: int = 512):
    import torch

=======
def _preprocess(code: str, max_len: int = 512):
    import torch
>>>>>>> 786234e1af7e83a1d2dfc59296b5812fe59318d9
    code = re.sub(r'/\*.*?\*/', '', code, flags=re.DOTALL)
    code = re.sub(r'//.*$', '', code, flags=re.MULTILINE)
    code = ' '.join(code.split()).strip()
    tokens = re.findall(r'\b\w+\b|[^\s\w]', code)
    unk = _cnn_vocab.get('<UNK>', 1)
    ids = [_cnn_vocab.get(t, unk) for t in tokens]
    ids = ids[:max_len]
    ids += [0] * (max_len - len(ids))
    return torch.tensor(ids, dtype=torch.long).unsqueeze(0)

<<<<<<< HEAD

def cnn_predict(code: str) -> dict:
    import torch

=======
def cnn_predict(code: str) -> dict:
    import torch
>>>>>>> 786234e1af7e83a1d2dfc59296b5812fe59318d9
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

<<<<<<< HEAD

# ─────────────────────────────────────────────────────────────────────────────
# LLM provider helpers  (aligned with phase6_llm_prompting.py)
# ─────────────────────────────────────────────────────────────────────────────

def _ollama_available(model: str) -> bool:
    try:
        import requests
        resp = requests.get("http://localhost:11434/api/tags", timeout=3)
        if resp.status_code == 200:
            models = [m["name"] for m in resp.json().get("models", [])]
            return model in models or any(model in m for m in models)
    except Exception:
        pass
    return False


def _parse_json_response(raw: str) -> dict:
    """Strip markdown fences and parse JSON; raise ValueError on failure."""
    cleaned = re.sub(r'^```(?:json)?\s*', '', raw.strip(), flags=re.IGNORECASE)
    cleaned = re.sub(r'\s*```$', '', cleaned)
    return json.loads(cleaned.strip())


def _call_groq(prompt: str, model: str, max_tokens: int, retries: int, backoff: float) -> Optional[str]:
    groq_key = os.environ.get("GROQ_API_KEY", "").strip()
    if not groq_key:
        return None
    try:
        from groq import Groq
    except ImportError:
        console.print("[yellow]  groq package not installed: pip install groq[/yellow]")
        return None

    client = Groq(api_key=groq_key)
    messages = [
        {"role": "system", "content": "You are an expert C/C++ security reviewer. Respond ONLY with the JSON object. No markdown, no extra text."},
        {"role": "user", "content": prompt},
    ]
    for attempt in range(1, retries + 1):
        try:
            resp = client.chat.completions.create(
                model=model, messages=messages,
                max_tokens=max_tokens, temperature=0.2,
            )
            return resp.choices[0].message.content.strip()
        except Exception as e:
            err = str(e).lower()
            if "rate_limit" in err or "429" in err:
                time.sleep(backoff ** attempt)
            elif "model_not_found" in err or "404" in err:
                return None
            elif "authentication" in err or "401" in err:
                return None
            else:
                time.sleep(backoff)
    return None


def _call_hf(prompt: str, model: str, max_tokens: int, retries: int, backoff: float) -> Optional[str]:
    hf_key = os.environ.get("HF_API_KEY", "").strip()
    if not hf_key:
        return None
    try:
        from huggingface_hub import InferenceClient
    except ImportError:
        console.print("[yellow]  huggingface_hub not installed: pip install huggingface_hub[/yellow]")
        return None

    client = InferenceClient(model=model, token=hf_key)
    for attempt in range(1, retries + 1):
        try:
            result = client.text_generation(prompt, max_new_tokens=max_tokens, temperature=0.2, do_sample=True)
            return result.strip()
        except Exception as e:
            err = str(e).lower()
            if "rate_limit" in err or "429" in err:
                time.sleep(backoff ** attempt)
            elif "model_not_found" in err or "404" in err:
                return None
            elif "authentication" in err or "401" in err:
                return None
            else:
                time.sleep(backoff)
    return None


def _call_ollama(prompt: str, model: str, max_tokens: int, retries: int, backoff: float) -> Optional[str]:
    try:
        import requests
    except ImportError:
        console.print("[yellow]  requests not installed: pip install requests[/yellow]")
        return None

    url = os.environ.get("OLLAMA_URL", "http://localhost:11434/api/generate")
    payload = {"model": model, "prompt": prompt, "stream": False,
               "options": {"num_predict": max_tokens, "temperature": 0.2}}

    for attempt in range(1, retries + 1):
        try:
            resp = requests.post(url, json=payload, timeout=120)
            resp.raise_for_status()
            return resp.json().get("response", "").strip()
        except requests.exceptions.ConnectionError:
            console.print("[yellow]  Ollama not running. Start with: ollama serve[/yellow]")
            return None
        except Exception as e:
            err = str(e).lower()
            if "rate_limit" in err or "429" in err:
                time.sleep(backoff ** attempt)
            elif "model_not_found" in err or "404" in err:
                console.print(f"[yellow]  Ollama model '{model}' not found. Run: ollama pull {model}[/yellow]")
                return None
            else:
                time.sleep(backoff)
    return None


# Module-level cache for the local HF pipeline (loaded once, reused across calls)
_local_pipeline = None
_local_model_name_loaded = None


def _call_local(prompt: str, model: str, max_tokens: int) -> Optional[str]:
    global _local_pipeline, _local_model_name_loaded

    local_model = os.environ.get("LOCAL_MODEL", "Qwen/Qwen2.5-0.5B-Instruct")
    try:
        import torch
        from transformers import pipeline as hf_pipeline
    except ImportError:
        console.print("[yellow]  transformers not installed: pip install transformers[/yellow]")
        return None

    if _local_pipeline is None or _local_model_name_loaded != local_model:
        console.print(f"  [dim]Loading local model: {local_model} (first call — may download)...[/dim]")
        device = 0 if torch.cuda.is_available() else -1
        _local_pipeline = hf_pipeline(
            "text-generation", model=local_model, device=device,
            dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            trust_remote_code=True,
        )
        if hasattr(_local_pipeline, "model") and hasattr(_local_pipeline.model, "generation_config"):
            _local_pipeline.model.generation_config.max_length = None
        _local_model_name_loaded = local_model

    messages = [
        {"role": "system", "content": "You are an expert C/C++ security reviewer. Respond ONLY with the JSON object. No markdown, no extra text."},
        {"role": "user", "content": prompt},
    ]
    try:
        out = _local_pipeline(messages, max_new_tokens=max_tokens, temperature=0.2,
                               do_sample=True, pad_token_id=_local_pipeline.tokenizer.eos_token_id)
        generated = out[0]["generated_text"]
        if isinstance(generated, list):
            turns = [m["content"] for m in generated if m["role"] == "assistant"]
            return turns[-1].strip() if turns else ""
        return str(generated).strip()
    except Exception as e:
        console.print(f"[red]  Local inference failed: {e}[/red]")
        return None


def _call_anthropic(prompt: str, max_tokens: int, retries: int, backoff: float) -> Optional[str]:
    """Direct Anthropic API via stdlib urllib (no SDK dependency)."""
    anthropic_key = os.environ.get("ANTHROPIC_API_KEY", "").strip()
    if not anthropic_key:
        return None

    payload = json.dumps({
        "model": "claude-haiku-4-5-20251001",
        "max_tokens": max_tokens,
        "system": "You are an expert C/C++ security reviewer. Respond ONLY with the JSON object. No markdown, no extra text.",
        "messages": [{"role": "user", "content": prompt}],
    }).encode()

    for attempt in range(1, retries + 1):
        try:
            req = urllib.request.Request(
                "https://api.anthropic.com/v1/messages",
                data=payload,
                headers={
                    "x-api-key": anthropic_key,
                    "anthropic-version": "2023-06-01",
                    "content-type": "application/json",
                },
            )
            with urllib.request.urlopen(req, timeout=60) as r:
                data = json.load(r)
            return data["content"][0]["text"].strip()
        except urllib.error.HTTPError as e:
            if e.code in (401, 403):
                console.print("[yellow]  Anthropic: invalid API key.[/yellow]")
                return None
            elif e.code == 429:
                time.sleep(backoff ** attempt)
            else:
                time.sleep(backoff)
        except Exception as e:
            console.print(f"[yellow]  Anthropic request failed: {e}[/yellow]")
            time.sleep(backoff)
    return None


# ─────────────────────────────────────────────────────────────────────────────
# Unified LLM explanation  (aligned with phase6 provider waterfall)
# ─────────────────────────────────────────────────────────────────────────────

def llm_explain(code: str, verdict: str, provider: str = "auto") -> dict:
    """
    Build the few-shot prompt, check the shared llm_cache.json, call the
    appropriate provider waterfall, parse JSON, and cache the result.

    Raises RuntimeError if no provider succeeds.
    """
    prompt = FEW_SHOT_TEMPLATE.format(code=code, verdict=verdict)

    # Resolve model names from env
    groq_model   = os.environ.get("GROQ_MODEL",   "llama-3.1-8b-instant")
    hf_model     = os.environ.get("HF_MODEL",     "Salesforce/codet5p-220m")
    ollama_model = os.environ.get("OLLAMA_MODEL", "qwen2.5-coder:7b-instruct-q4_K_M")
    local_model  = os.environ.get("LOCAL_MODEL",  "Qwen/Qwen2.5-0.5B-Instruct")

    # Build provider waterfall (mirrors phase6 call_llm_api logic)
    if provider == "auto":
        if _groq_available():
            waterfall = [("groq", groq_model)]
        elif _hf_available():
            waterfall = [("hf", hf_model)]
        elif _ollama_available(ollama_model):
            waterfall = [("ollama", ollama_model)]
        elif os.environ.get("ANTHROPIC_API_KEY"):
            waterfall = [("anthropic", "claude-haiku-4-5-20251001")]
        else:
            waterfall = [("local", local_model)]
    elif provider == "groq":
        waterfall = [("groq", groq_model), ("ollama", ollama_model), ("local", local_model)]
    elif provider == "hf":
        waterfall = [("hf", hf_model), ("ollama", ollama_model), ("local", local_model)]
    elif provider == "ollama":
        waterfall = [("ollama", ollama_model), ("local", local_model)]
    elif provider == "anthropic":
        waterfall = [("anthropic", "claude-haiku-4-5-20251001"), ("local", local_model)]
    else:
        waterfall = [("local", local_model)]

    cache = _load_cache()
    MAX_TOKENS = 500
    RETRIES    = 3
    BACKOFF    = 2.0

    for prov, model in waterfall:
        cache_key = _cache_key(prompt, model, prov)
        if cache_key in cache:
            raw = cache[cache_key]
        else:
            if prov == "groq":
                raw = _call_groq(prompt, model, MAX_TOKENS, RETRIES, BACKOFF)
            elif prov == "hf":
                raw = _call_hf(prompt, model, MAX_TOKENS, RETRIES, BACKOFF)
            elif prov == "ollama":
                raw = _call_ollama(prompt, model, MAX_TOKENS, RETRIES, BACKOFF)
            elif prov == "anthropic":
                raw = _call_anthropic(prompt, MAX_TOKENS, RETRIES, BACKOFF)
            else:  # local
                raw = _call_local(prompt, model, MAX_TOKENS)

            if raw:
                cache[cache_key] = raw
                _save_cache(cache)

        if raw:
            try:
                return _parse_json_response(raw)
            except (json.JSONDecodeError, ValueError) as e:
                console.print(f"  [yellow]{prov.upper()} returned invalid JSON ({e}), trying next provider...[/yellow]")
                continue

    raise RuntimeError(
        "All providers failed or returned unparseable JSON.\n"
        "Set GROQ_API_KEY (free at console.groq.com) or another API key, "
        "or run 'ollama serve' with a local model."
    )

=======
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
>>>>>>> 786234e1af7e83a1d2dfc59296b5812fe59318d9

# ─────────────────────────────────────────────────────────────────────────────
# Rendering
# ─────────────────────────────────────────────────────────────────────────────
SEV_COLOR = {"HIGH": "red", "MEDIUM": "yellow", "LOW": "green"}
SEV_ICON  = {"HIGH": "●", "MEDIUM": "◆", "LOW": "○"}

<<<<<<< HEAD

=======
>>>>>>> 786234e1af7e83a1d2dfc59296b5812fe59318d9
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

<<<<<<< HEAD

def render_sample_menu():
    table = Table(box=box.SIMPLE, show_header=True, header_style="bold cyan",
                  border_style="dim", padding=(0, 2))
=======
def render_sample_menu():
    table = Table(box=box.SIMPLE, show_header=True, header_style="bold cyan", border_style="dim", padding=(0, 2))
>>>>>>> 786234e1af7e83a1d2dfc59296b5812fe59318d9
    table.add_column("#",    style="bold white", width=3)
    table.add_column("File", style="green")
    table.add_column("Description", style="dim")
    for k, s in SAMPLES.items():
        table.add_row(k, s["name"], s["desc"])
    table.add_row("p", "[yellow]paste[/yellow]",  "Paste your own code")
    table.add_row("f", "[yellow]file[/yellow]",   "Load from a file path")
    table.add_row("q", "[red]quit[/red]",          "Exit")
    console.print(Panel(table, title="[bold]Select a sample[/bold]", border_style="bright_blue"))

<<<<<<< HEAD

=======
>>>>>>> 786234e1af7e83a1d2dfc59296b5812fe59318d9
def render_code(code: str, filename: str = "code.c"):
    syntax = Syntax(code, "c", theme="monokai", line_numbers=True, word_wrap=True)
    console.print(Panel(syntax, title=f"[bold cyan]{filename}[/bold cyan]", border_style="dim"))

<<<<<<< HEAD

=======
>>>>>>> 786234e1af7e83a1d2dfc59296b5812fe59318d9
def render_cnn_result(pred: dict):
    verdict    = pred["verdict"]
    prob_buggy = pred["prob_buggy"]
    prob_clean = 1 - prob_buggy
    color = "red" if verdict == "BUGGY" else "green"
    icon  = "✗" if verdict == "BUGGY" else "✓"

<<<<<<< HEAD
    # FIX: clamp marker_pos so it never equals bar_width (IndexError when prob=1.0)
    bar_width  = 30
    marker_pos = min(round(prob_buggy * bar_width), bar_width - 1)

=======
    # Two-sided bar: red (buggy) on left, green (clean) on right
    # The | marker shows where the model sits on the 0-1 scale
    bar_width = 30
    marker_pos = round(prob_buggy * bar_width)
>>>>>>> 786234e1af7e83a1d2dfc59296b5812fe59318d9
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
<<<<<<< HEAD
    t.append(f"  {icon}  {verdict}  ",
             style=f"bold {color} on {'#3d0000' if verdict == 'BUGGY' else '#003d00'}")
    console.print(t)
    console.print(
        f"\n  [red]BUGGY[/red] "
        f"[red]{bar[:marker_pos]}[/red]"
        f"[bold white]{bar[marker_pos]}[/bold white]"
        f"[green]{bar[marker_pos + 1:]}[/green] [green]CLEAN[/green]"
    )
    console.print(
        f"         [red]P(buggy) = {prob_buggy * 100:.1f}%[/red]"
        f"   [green]P(clean) = {prob_clean * 100:.1f}%[/green]"
    )
    console.print(
        f"         logit: {pred['logit']:+.3f}   "
        f"{'(negative = leans CLEAN)' if pred['logit'] < 0 else '(positive = leans BUGGY)'}\n"
    )

=======
    t.append(f"  {icon}  {verdict}  ", style=f"bold {color} on {'#3d0000' if verdict == 'BUGGY' else '#003d00'}")
    console.print(t)
    console.print(f"\n  [red]BUGGY[/red] [red]{bar[:marker_pos]}[/red][bold white]{bar[marker_pos]}[/bold white][green]{bar[marker_pos+1:]}[/green] [green]CLEAN[/green]")
    console.print(f"         [red]P(buggy) = {prob_buggy*100:.1f}%[/red]   [green]P(clean) = {prob_clean*100:.1f}%[/green]")
    console.print(f"         logit: {pred['logit']:+.3f}   {'(negative = leans CLEAN)' if pred['logit'] < 0 else '(positive = leans BUGGY)'}\n")
>>>>>>> 786234e1af7e83a1d2dfc59296b5812fe59318d9

def render_llm_explanation(expl: dict, elapsed: float):
    issues = expl.get("issues", [])
    rec    = expl.get("recommendation", "")

    console.print(Rule("[bold]Step 2 — LLM Explanation (few-shot)[/bold]", style="dim"))
    console.print(f"  [dim]Response time: {elapsed:.1f}s[/dim]\n")

    if issues:
        console.print(f"  [bold white]Issues ({len(issues)})[/bold white]")
        for iss in issues:
<<<<<<< HEAD
            sev  = iss.get("severity", "LOW").upper()
            col  = SEV_COLOR.get(sev, "white")
            icon = SEV_ICON.get(sev, "○")
            console.print(f"  [{col}]{icon} {sev}[/{col}]  [bold]{iss.get('title', '')}[/bold]")
=======
            sev  = iss.get("severity", "LOW")
            col  = SEV_COLOR.get(sev, "white")
            icon = SEV_ICON.get(sev, "○")
            console.print(f"  [{col}]{icon} {sev}[/{col}]  [bold]{iss.get('title','')}[/bold]")
>>>>>>> 786234e1af7e83a1d2dfc59296b5812fe59318d9
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

<<<<<<< HEAD

# ─────────────────────────────────────────────────────────────────────────────
# Main review flow
# ─────────────────────────────────────────────────────────────────────────────
def review_code(code: str, filename: str = "snippet.c", provider: str = "auto"):
=======
# ─────────────────────────────────────────────────────────────────────────────
# Main review flow
# ─────────────────────────────────────────────────────────────────────────────
def review_code(code: str, filename: str = "snippet.c"):
>>>>>>> 786234e1af7e83a1d2dfc59296b5812fe59318d9
    render_code(code, filename)

    # Step 1: TextCNN
    cnn_error = None
<<<<<<< HEAD
    with Progress(
        SpinnerColumn(spinner_name="dots", style="cyan"),
        TextColumn("[cyan]Running TextCNN...[/cyan]"),
        transient=True,
    ) as prog:
        prog.add_task("cnn", total=None)
=======
    with Progress(SpinnerColumn(spinner_name="dots", style="cyan"), TextColumn("[cyan]Running TextCNN...[/cyan]"), transient=True) as p:
        p.add_task("cnn", total=None)
>>>>>>> 786234e1af7e83a1d2dfc59296b5812fe59318d9
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
<<<<<<< HEAD
    with Progress(
        SpinnerColumn(spinner_name="dots2", style="magenta"),
        TextColumn("[magenta]Generating LLM explanation (few-shot)...[/magenta]"),
        transient=True,
    ) as prog:
        prog.add_task("llm", total=None)
        t0 = time.time()
        try:
            expl = llm_explain(code, verdict, provider=provider)
        except RuntimeError as e:
            console.print(f"\n  [red]LLM error:[/red] {e}\n")
=======
    with Progress(SpinnerColumn(spinner_name="dots2", style="magenta"), TextColumn("[magenta]Generating LLM explanation (few-shot)...[/magenta]"), transient=True) as p:
        p.add_task("llm", total=None)
        t0 = time.time()
        try:
            expl = llm_explain(code, verdict)
        except Exception as e:
            console.print(f"\n  [red]LLM error:[/red] {e}\n")
            console.print("  [dim]Set GROQ_API_KEY (free at console.groq.com) or ANTHROPIC_API_KEY.[/dim]\n")
>>>>>>> 786234e1af7e83a1d2dfc59296b5812fe59318d9
            return
        elapsed = time.time() - t0

    render_llm_explanation(expl, elapsed)

<<<<<<< HEAD

# ─────────────────────────────────────────────────────────────────────────────
# Interactive loop
# ─────────────────────────────────────────────────────────────────────────────
def interactive(provider: str = "auto"):
=======
# ─────────────────────────────────────────────────────────────────────────────
# Interactive loop
# ─────────────────────────────────────────────────────────────────────────────
def interactive():
>>>>>>> 786234e1af7e83a1d2dfc59296b5812fe59318d9
    render_header()
    while True:
        render_sample_menu()
        choice = Prompt.ask("  [bold cyan]>[/bold cyan]").strip().lower()

        if choice == "q":
            console.print("\n  [dim]Bye![/dim]\n")
            break
<<<<<<< HEAD

        elif choice in SAMPLES:
            s = SAMPLES[choice]
            review_code(s["code"], s["name"], provider=provider)
            if not Confirm.ask("  Review another?", default=True):
                console.print("\n  [dim]Bye![/dim]\n")
                break

        elif choice == "p":
            console.print(
                "\n  [dim]Paste your C/C++ code. "
                "Type END on a blank line to finish.[/dim]\n"
            )
            lines: list[str] = []
            try:
                while True:
                    line = input()
                    if line.strip().upper() == "END":
                        break
                    lines.append(line)
            except EOFError:
                pass  # piped input
            code = "\n".join(lines).strip()
            if code:
                review_code(code, "custom.c", provider=provider)
            else:
                console.print("  [yellow]No code entered.[/yellow]\n")

        elif choice == "f":
            file_path_str = Prompt.ask("  File path").strip()
            file_path = Path(file_path_str)
            if not file_path.exists():
                console.print(f"  [red]File not found:[/red] {file_path_str}\n")
            else:
                review_code(file_path.read_text(), file_path.name, provider=provider)

        else:
            console.print("  [red]Invalid choice.[/red]\n")


=======
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

>>>>>>> 786234e1af7e83a1d2dfc59296b5812fe59318d9
# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Code Review Assistant - TextCNN + LLM demo")
<<<<<<< HEAD
    parser.add_argument("--file",     "-f", help="Path to a C/C++ file to review")
    parser.add_argument("--sample",   "-s", choices=list(SAMPLES.keys()),
                        help="Run a built-in sample (1-5)")
    parser.add_argument("--provider", "-p",
                        choices=["auto", "groq", "hf", "ollama", "local", "anthropic"],
                        default="auto",
                        help="LLM provider to use (default: auto)")
=======
    parser.add_argument("--file",   "-f", help="Path to a C/C++ file to review")
    parser.add_argument("--sample", "-s", choices=list(SAMPLES.keys()), help="Run a built-in sample (1-5)")
>>>>>>> 786234e1af7e83a1d2dfc59296b5812fe59318d9
    args = parser.parse_args()

    if args.file:
        render_header()
<<<<<<< HEAD
        file_path = Path(args.file)
        if not file_path.exists():
            console.print(f"[red]File not found:[/red] {args.file}")
            sys.exit(1)
        review_code(file_path.read_text(), file_path.name, provider=args.provider)
    elif args.sample:
        render_header()
        s = SAMPLES[args.sample]
        review_code(s["code"], s["name"], provider=args.provider)
    else:
        interactive(provider=args.provider)

=======
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
>>>>>>> 786234e1af7e83a1d2dfc59296b5812fe59318d9

if __name__ == "__main__":
    main()