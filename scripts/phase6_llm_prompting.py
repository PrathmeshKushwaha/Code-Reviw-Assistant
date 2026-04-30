"""
Phase 6: LLM Prompt Engineering for Code Review Comments
Generates bug-review comments for 50 test samples using three prompt templates:
  - Zero-shot
  - One-shot
  - Few-shot (3 examples)
Uses Groq API for fast inference (requires GROQ_API_KEY).
Falls back to local transformers pipeline if key is missing.
Results saved to outputs/llm_comments.json and outputs/llm_comments.csv
"""
 
import os
import sys
import re
import json
import time
import logging
import hashlib
import yaml
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional

REQUEST_DELAY = 2.0

# ADD THESE LINES:
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass  # Optional: pip install python-dotenv
 
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s"
)

# DEBUG: Print what the script sees (MUST be after basicConfig)
logging.info(f"🔍 DEBUG: API_PROVIDER={os.environ.get('API_PROVIDER')}, OLLAMA_MODEL={os.environ.get('OLLAMA_MODEL')}")

def _ollama_available(model: str) -> bool:
    """Check if Ollama server is running and has the specified model"""
    try:
        import requests
        resp = requests.get("http://localhost:11434/api/tags", timeout=5)
        if resp.status_code == 200:
            models = [m["name"] for m in resp.json().get("models", [])]
            # Check exact match or substring match (for tags like qwen2.5-coder:7b-instruct-q4_K_M)
            return model in models or any(model.split(":")[0] in m for m in models)
    except Exception as e:
        logging.debug(f"Ollama availability check failed: {e}")
    return False
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
OUTPUT_DIR = Path("outputs")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
CACHE_FILE = OUTPUT_DIR / "llm_cache.json"
 
# ---------------------------------------------------------------------------
# Prompt templates
# ---------------------------------------------------------------------------
ZERO_SHOT_TEMPLATE = """\
You are an expert code reviewer specialising in C/C++ security vulnerabilities and bugs.
Review the following code and write a concise, technical comment explaining any bugs or vulnerabilities found. If the code looks correct, say so.
 
Code:
```c
{code}
```
 
Review comment:"""
 
ONE_SHOT_TEMPLATE = """\
You are an expert code reviewer specialising in C/C++ security vulnerabilities and bugs.
Here is an example of a good code review:
 
Example code:
```c
void copy_data(char *dst, char *src, int n) {{
    memcpy(dst, src, n);
    // no null-check on dst or src
}}
```
Review: The function does not validate that `dst` or `src` are non-NULL before calling `memcpy`, which can lead to a NULL-pointer dereference. Additionally, there is no bounds check – if `n` exceeds the allocated size of `dst`, a heap/stack buffer overflow will occur.
 
Now review the following code:
 
Code:
```c
{code}
```
 
Review comment:"""
 
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
Review: Uses the unsafe `gets()` function, which performs no bounds checking and is a classic stack buffer overflow vulnerability. Replace with `fgets(buf, MAX_SIZE, stdin)`.
 
--- Example 2 (CLEAN) ---
Code:
```c
int safe_div(int a, int b) {{
    if (b == 0) return -1;
    return a / b;
}}
```
Review: The function correctly guards against division by zero before performing the division. No issues found.
 
--- Example 3 (BUGGY) ---
Code:
```c
char *dup_str(const char *s) {{
    char *p = malloc(strlen(s));
    strcpy(p, s);
    return p;
}}
```
Review: Off-by-one error: `malloc(strlen(s))` allocates one byte too few – it omits space for the null terminator. Should be `malloc(strlen(s) + 1)`. Additionally, the return value of `malloc` is not checked for NULL.
 
Now review the following code:
 
Code:
```c
{code}
```
 
Review comment:"""
 
TEMPLATES = {
    "zero_shot": ZERO_SHOT_TEMPLATE,
    "one_shot":  ONE_SHOT_TEMPLATE,
    "few_shot":  FEW_SHOT_TEMPLATE,
}
 
# ---------------------------------------------------------------------------
# Cache helpers
# ---------------------------------------------------------------------------
def _cache_key(prompt: str, model: str, provider: str) -> str:
    return hashlib.md5(f"{provider}::{model}::{prompt}".encode()).hexdigest()
 
 
def load_cache() -> dict:
    if CACHE_FILE.exists():
        with open(CACHE_FILE) as f:
            return json.load(f)
    return {}
 
 
def save_cache(cache: dict):
    with open(CACHE_FILE, "w") as f:
        json.dump(cache, f, indent=2)
 
 
# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# Inference helpers
# ---------------------------------------------------------------------------
def call_llm_api(
    prompt: str,
    provider: str,
    model: str,
    max_new_tokens: int = 128,
    retries: int = 3,
    backoff: float = 2.0,
) -> Optional[str]:
    """Unified LLM caller: uses the provider/model from run(), with smart fallback."""
    
    # Get the correct model name for EACH provider
    ollama_model = os.environ.get("OLLAMA_MODEL", "qwen2.5-coder:7b-instruct-q4_K_M")
    groq_model = os.environ.get("GROQ_MODEL", "llama-3.1-8b-instant")
    hf_model = os.environ.get("HF_MODEL", "Salesforce/codet5p-220m")
    local_model = os.environ.get("LOCAL_MODEL", "Qwen/Qwen2.5-0.5B-Instruct")
    
    # Map provider -> correct model name
    provider_model_map = {
        "groq": groq_model,
        "hf": hf_model,
        "ollama": ollama_model,
        "local": local_model,
    }
    
    # Determine fallback order based on the PRIMARY provider selected by run()
    if provider == "groq":
        providers_to_try = ["groq", "ollama", "local"]
    elif provider == "hf":
        providers_to_try = ["hf", "ollama", "local"]
    elif provider == "ollama":
        providers_to_try = ["ollama", "local"]
    else:  # "local"
        providers_to_try = ["local"]
    
    for prov in providers_to_try:
        # USE THE CORRECT MODEL FOR THIS PROVIDER
        prov_model = provider_model_map[prov]
        
        try:
            if prov == "groq":
                result = _call_groq(prompt, prov_model, max_new_tokens, retries, backoff)
                if result: return result
            elif prov == "hf":
                result = _call_hf(prompt, prov_model, max_new_tokens, retries, backoff)
                if result: return result
            elif prov == "ollama":
                result = _call_ollama(prompt, prov_model, max_new_tokens, retries, backoff)
                if result: return result
            elif prov == "local":
                result = _call_local(prompt, prov_model, max_new_tokens)
                if result: return result
        except Exception as e:
            logging.warning(f"⚠️ {prov.upper()} failed: {e}")
            continue
    return None


def _call_groq(prompt: str, model: str, max_new_tokens: int, retries: int, backoff: float) -> Optional[str]:
    groq_key = os.environ.get("GROQ_API_KEY", "").strip()
    if not groq_key: return None
    try:
        from groq import Groq
    except ImportError:
        logging.error("groq not installed: pip install groq"); return None
    
    client = Groq(api_key=groq_key)
    messages = [
        {"role": "system", "content": "You are an expert code reviewer specialising in C/C++ security vulnerabilities and bugs. Be concise and technical."},
        {"role": "user", "content": prompt},
    ]
    for attempt in range(1, retries + 1):
        try:
            response = client.chat.completions.create(model=model, messages=messages, max_tokens=max_new_tokens, temperature=0.3)
            return response.choices[0].message.content.strip()
        except Exception as e:
            err = str(e).lower()
            if "rate_limit" in err or "429" in err: time.sleep(backoff ** attempt)
            elif "model_not_found" in err or "404" in err: return None
            elif "authentication" in err or "401" in err: return None
            else: time.sleep(backoff)
    return None


def _call_hf(prompt: str, model: str, max_new_tokens: int, retries: int, backoff: float) -> Optional[str]:
    hf_key = os.environ.get("HF_API_KEY", "").strip()
    if not hf_key: return None
    try:
        from huggingface_hub import InferenceClient
    except ImportError:
        logging.error("huggingface_hub not installed: pip install huggingface_hub"); return None
    
    client = InferenceClient(model=model, token=hf_key)
    for attempt in range(1, retries + 1):
        try:
            result = client.text_generation(prompt, max_new_tokens=max_new_tokens, temperature=0.3, do_sample=True)
            return result.strip()
        except Exception as e:
            err = str(e).lower()
            if "rate_limit" in err or "429" in err: time.sleep(backoff ** attempt)
            elif "model_not_found" in err or "404" in err: return None
            elif "authentication" in err or "401" in err: return None
            else: time.sleep(backoff)
    return None

# Module-level cache for the local pipeline (loaded once, reused across calls)
_local_pipeline = None
_local_model_name = None

def _call_local(prompt: str, model: str, max_new_tokens: int = 128) -> Optional[str]:
    """Run inference locally using transformers. Downloads model on first call."""
    global _local_pipeline, _local_model_name

    local_model = os.environ.get("LOCAL_MODEL", "Qwen/Qwen2.5-0.5B-Instruct")

    try:
        import torch
        from transformers import pipeline as hf_pipeline
    except ImportError:
        logging.error("transformers not installed: pip install transformers")
        return None

    if _local_pipeline is None or _local_model_name != local_model:
        logging.info(f"   Loading local model: {local_model} (first call — may download)...")
        device = 0 if torch.cuda.is_available() else -1
        _local_pipeline = hf_pipeline(
            "text-generation",
            model=local_model,
            device=device,
            dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            trust_remote_code=True,
        )
        # Clear max_length to avoid conflict with max_new_tokens at call time
        if hasattr(_local_pipeline, "model") and hasattr(_local_pipeline.model, "generation_config"):
            _local_pipeline.model.generation_config.max_length = None
        _local_model_name = local_model
        logging.info(f"   Local model ready on {'GPU' if device == 0 else 'CPU'}.")

    messages = [
        {
            "role": "system",
            "content": (
                "You are an expert code reviewer specialising in C/C++ security "
                "vulnerabilities and bugs. Be concise and technical."
            ),
        },
        {"role": "user", "content": prompt},
    ]

    try:
        out = _local_pipeline(
            messages,
            max_new_tokens=max_new_tokens,
            temperature=0.3,
            do_sample=True,
            pad_token_id=_local_pipeline.tokenizer.eos_token_id,
        )
        generated = out[0]["generated_text"]
        if isinstance(generated, list):
            assistant_turns = [m["content"] for m in generated if m["role"] == "assistant"]
            return assistant_turns[-1].strip() if assistant_turns else ""
        return str(generated).strip()
    except Exception as e:
        logging.error(f"   Local inference failed: {repr(e)}")
        return None

def _call_ollama(
    prompt: str,
    model: str,
    max_new_tokens: int,
    retries: int,
    backoff: float,
) -> Optional[str]:
    """Call local Ollama server (http://localhost:11434)"""
    try:
        import requests
    except ImportError:
        logging.error("requests not installed: pip install requests"); return None
    
    ollama_url = os.environ.get("OLLAMA_URL", "http://localhost:11434/api/generate")
    
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": {
            "num_predict": max_new_tokens,
            "temperature": 0.3,
        }
    }
    
    for attempt in range(1, retries + 1):
        try:
            response = requests.post(ollama_url, json=payload, timeout=120)
            response.raise_for_status()
            result = response.json()
            return result.get("response", "").strip()
        except requests.exceptions.ConnectionError:
            logging.error("❌ Could not connect to Ollama. Is it running? (ollama serve)")
            return None
        except Exception as e:
            err = str(e).lower()
            if "rate_limit" in err or "429" in err:
                time.sleep(backoff ** attempt)
            elif "model_not_found" in err or "404" in err:
                logging.warning(f"   Ollama model '{model}' not found. Run: ollama pull {model}")
                return None
            else:
                time.sleep(backoff)
    return None

def generate_comment(
    code: str,
    prompt_type: str,
    provider: str,
    model: str,
    cache: dict,
    max_new_tokens: int = 128,
) -> str:
    """Build prompt, check cache, call API."""
    template = TEMPLATES[prompt_type]
    prompt = template.format(code=code)
    key = _cache_key(prompt, model, provider)
    if key in cache:
        return cache[key]
    comment = call_llm_api(prompt=prompt, provider=provider, model=model, max_new_tokens=max_new_tokens)
    if comment is None:
        comment = "[API call failed – see logs]"
 
    cache[key] = comment
    save_cache(cache)
    return comment
 
 
# ---------------------------------------------------------------------------
# Load code samples
# ---------------------------------------------------------------------------
def load_test_samples(n: int = 50) -> list[dict]:
    """
    Load N test samples from Phase 2 artefacts if available,
    otherwise use inline fallback samples.
    """
    import torch
 
    try:
        import yaml as _yaml
        with open("configs/defaults.yaml") as f:
            cfg = _yaml.safe_load(f)
        data_dir = Path(cfg["dataset"]["processed_dir"])
        vocab_path = data_dir / "vocab.json"
        test_inputs_path = data_dir / "test_inputs.pt"
        test_labels_path = data_dir / "test_labels.pt"
 
        if not all(p.exists() for p in [vocab_path, test_inputs_path, test_labels_path]):
            raise FileNotFoundError("Phase 2 artefacts missing.")
 
        with open(vocab_path) as f:
            vocab = json.load(f)
        id2tok = {v: k for k, v in vocab.items()}
 
        test_inputs = torch.load(test_inputs_path, weights_only=True)
        test_labels = torch.load(test_labels_path, weights_only=True)
 
        rng = np.random.default_rng(42)
        indices = rng.choice(len(test_inputs), size=min(n, len(test_inputs)), replace=False)
 
        samples = []
        for idx in indices:
            ids = test_inputs[idx].tolist()
            tokens = [id2tok.get(i, "<UNK>") for i in ids if i != 0]  # strip <PAD>
            code_approx = " ".join(tokens[:200])  # first 200 tokens as proxy
            samples.append({
                "sample_id": int(idx),
                "code": code_approx,
                "label": int(test_labels[idx].item()),
            })
        logging.info(f"✅ Loaded {len(samples)} test samples from Phase 2 artefacts.")
        return samples
 
    except Exception as e:
        logging.warning(f"⚠️  Could not load Phase 2 artefacts ({e}). Using fallback samples.")
 
    # Fallback
    fallback = [
        {"sample_id": 0, "label": 1, "code": 'void vuln(char *input) {\n    char buf[64];\n    strcpy(buf, input);\n    printf("%s\\n", buf);\n}'},
        {"sample_id": 1, "label": 0, "code": 'int safe_add(int a, int b) {\n    if ((b > 0) && (a > INT_MAX - b)) return -1;\n    return a + b;\n}'},
        {"sample_id": 2, "label": 1, "code": 'char *get_token(char *s) {\n    char *p = malloc(strlen(s));\n    strcpy(p, s);\n    return p;\n}'},
        {"sample_id": 3, "label": 0, "code": 'void read_safe(FILE *fp, char *buf, size_t n) {\n    if (!fp || !buf || n == 0) return;\n    fread(buf, 1, n - 1, fp);\n    buf[n-1] = \'\\0\';\n}'},
        {"sample_id": 4, "label": 1, "code": 'int *make_array(int n) {\n    int arr[n];\n    for (int i=0;i<n;i++) arr[i]=i;\n    return arr;\n}'},
    ]
    # Repeat to reach n
    samples = (fallback * ((n // len(fallback)) + 1))[:n]
    for i, s in enumerate(samples):
        s["sample_id"] = i  # re-index so IDs are unique
    logging.info(f"✅ Using {len(samples)} fallback samples.")
    return samples
 
 
# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def run():
    logging.info("=" * 60)
    logging.info("Phase 6: LLM Prompt Engineering")
    logging.info("=" * 60)
 
    # Configuration from environment
    api_provider = os.environ.get("API_PROVIDER", "auto").lower()
    ollama_model = os.environ.get("OLLAMA_MODEL", "qwen2.5-coder:7b-instruct-q4_K_M")  # ← Default fallback
    groq_model = os.environ.get("GROQ_MODEL", "llama-3.1-8b-instant")
    hf_model = os.environ.get("HF_MODEL", "Salesforce/codet5p-220m")
    n_samples = int(os.environ.get("N_SAMPLES", "50"))
    
    # Update the provider selection logic:
    if api_provider == "groq":
        provider, model_name = "groq", groq_model
    elif api_provider == "hf":
        provider, model_name = "hf", hf_model
    elif api_provider == "ollama":  # ← New
        provider, model_name = "ollama", ollama_model
    # FIXED LOGIC: Check if packages are actually installed
    else:  # auto
        if _groq_available():
            provider, model_name = "groq", groq_model
        elif _hf_available():
            provider, model_name = "hf", hf_model
        elif _ollama_available(ollama_model):
            provider, model_name = "ollama", ollama_model
        else:
            provider, model_name = "local", os.environ.get("LOCAL_MODEL", "Qwen/Qwen2.5-0.5B-Instruct")
    logging.info(f"   Provider:  {provider}")
    logging.info(f"   Model:     {model_name}")
    # Warn if user picked a base (non-instruct) model unlikely to follow prompts
    # Recommended Groq models for code review:
    # llama-3.1-8b-instant  — fast, free, good quality
    # llama-3.3-70b-versatile — best quality, slower
    # mixtral-8x7b-32768     — large context window
    _GROQ_MODELS = {"llama-3.1-8b-instant", "llama-3.3-70b-versatile", "mixtral-8x7b-32768", "gemma2-9b-it"}
    if model_name not in _GROQ_MODELS:
        logging.warning(
            f"   Unknown Groq model '{model_name}'. "
            "Recommended: llama-3.1-8b-instant or llama-3.3-70b-versatile"
        )
    logging.info(f"   Samples: {n_samples}")
 
    cache = load_cache()
    samples = load_test_samples(n=n_samples)
 
    results = []
    total = len(samples) * len(TEMPLATES)
    done  = 0
 
    for sample in samples:
        row = {
            "sample_id": sample["sample_id"],
            "label":     sample["label"],  # ground-truth: 1=buggy, 0=clean
            "code":      sample["code"],
        }
        for prompt_type in TEMPLATES:
            logging.info(
                f"[{done+1}/{total}] sample={sample['sample_id']} | prompt={prompt_type}"
            )
            comment = generate_comment(
                code=sample["code"],
                prompt_type=prompt_type,
                provider=provider,
                model=model_name,
                cache=cache,
                max_new_tokens=128,
            )
            row[f"comment_{prompt_type}"] = comment
            done += 1
            time.sleep(0.5)  # polite pacing
 
        results.append(row)
 
    # ------------------------------------------------------------------
    # Save outputs
    # ------------------------------------------------------------------
    json_path = OUTPUT_DIR / "llm_comments.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump({"provider": provider, "model": model_name, "results": results}, f, indent=2, ensure_ascii=False)
    logging.info(f"\n💾 Saved JSON: {json_path}")
 
    csv_path = OUTPUT_DIR / "llm_comments.csv"
    df = pd.DataFrame(results)
    df.to_csv(csv_path, index=False)
    logging.info(f"💾 Saved CSV:  {csv_path}")
 
    # ------------------------------------------------------------------
    # Quick preview
    # ------------------------------------------------------------------
    logging.info("\n" + "=" * 60)
    logging.info("SAMPLE OUTPUT PREVIEW (first 2 samples)")
    logging.info("=" * 60)
    for row in results[:2]:
        logging.info(f"\nSample {row['sample_id']} | Label: {'BUGGY' if row['label'] else 'CLEAN'}")
        for pt in TEMPLATES:
            comment = row.get(f"comment_{pt}", "")
            logging.info(f"  [{pt}]: {comment[:200]}{'...' if len(comment) > 200 else ''}")
 
    logging.info("\n✅ Phase 6 complete.")
    return results
 
 
if __name__ == "__main__":
    run()