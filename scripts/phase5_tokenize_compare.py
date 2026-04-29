"""
Phase 5: Tokenization Comparison
Compares GPT-2 BPE tokenizer vs. project word-level tokenizer on C and Python code.
Metrics: Fertility, OOV rate, Vocab coverage.
Saves results to outputs/tokenization_comparison.json and outputs/tokenization_comparison.csv
"""

import os
import sys
import re
import json
import logging
import yaml
import pandas as pd
import numpy as np
from pathlib import Path
from collections import Counter

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.data_processor import CodePreprocessor

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s"
)

# ---------------------------------------------------------------------------
# Constants & Config
# ---------------------------------------------------------------------------
OUTPUT_DIR = Path("outputs")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Sample C code snippets (representative subset – used when Devign parquet exists)
FALLBACK_C_SAMPLES = [
    """void process(char *buf, int len) {
    char *tmp = malloc(len);
    if (!tmp) return;
    memcpy(tmp, buf, len);
    free(tmp);
}""",
    """int read_file(const char *path) {
    FILE *fp = fopen(path, "r");
    if (fp == NULL) { return -1; }
    char line[256];
    while (fgets(line, sizeof(line), fp)) {
        printf("%s", line);
    }
    fclose(fp);
    return 0;
}""",
    """static int check_bounds(int *arr, int idx, int size) {
    if (idx < 0 || idx >= size) {
        fprintf(stderr, "Index %d out of bounds\\n", idx);
        return -1;
    }
    return arr[idx];
}""",
    """void copy_string(char *dst, const char *src) {
    while (*src) {
        *dst++ = *src++;
    }
    *dst = '\\0';
}""",
    """int parse_int(const char *s, int *out) {
    char *end;
    errno = 0;
    long val = strtol(s, &end, 10);
    if (errno || end == s || *end != '\\0') return -1;
    *out = (int)val;
    return 0;
}""",
]

FALLBACK_PYTHON_SAMPLES = [
    """def process_data(items):
    results = []
    for item in items:
        if item is None:
            continue
        results.append(item.strip().lower())
    return results""",
    """class DataLoader:
    def __init__(self, path, batch_size=32):
        self.path = path
        self.batch_size = batch_size
        self.data = []

    def load(self):
        with open(self.path, 'r') as f:
            self.data = [line.strip() for line in f]
        return self""",
    """def compute_metrics(y_true, y_pred):
    tp = sum(1 for a, b in zip(y_true, y_pred) if a == 1 and b == 1)
    fp = sum(1 for a, b in zip(y_true, y_pred) if a == 0 and b == 1)
    fn = sum(1 for a, b in zip(y_true, y_pred) if a == 1 and b == 0)
    precision = tp / (tp + fp + 1e-9)
    recall = tp / (tp + fn + 1e-9)
    f1 = 2 * precision * recall / (precision + recall + 1e-9)
    return {"precision": precision, "recall": recall, "f1": f1}""",
    """import os
import json

def load_config(config_path="config.json"):
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config not found: {config_path}")
    with open(config_path) as f:
        return json.load(f)""",
    """def tokenize(text, lower=True, remove_punct=False):
    if lower:
        text = text.lower()
    tokens = text.split()
    if remove_punct:
        tokens = [t.strip('.,!?;:\\"') for t in tokens]
    return tokens""",
]


# ---------------------------------------------------------------------------
# Word-level tokenizer (mirrors CodePreprocessor)
# ---------------------------------------------------------------------------
WORD_TOKENIZER_RE = re.compile(r'\b\w+\b|[^\s\w]')


def word_tokenize(code: str) -> list[str]:
    """Replicate the project's own tokenizer exactly."""
    # Strip C comments
    code = re.sub(r'/\*.*?\*/', '', code, flags=re.DOTALL)
    code = re.sub(r'//.*$', '', code, flags=re.MULTILINE)
    return WORD_TOKENIZER_RE.findall(code)


# ---------------------------------------------------------------------------
# Metrics computation
# ---------------------------------------------------------------------------
def compute_tokenizer_metrics(
    samples: list[str],
    word_tokens_all: list[list[str]],
    bpe_tokens_all: list[list[str]],
    word_vocab: set[str],
    bpe_vocab: set[str],
) -> dict:
    """
    Compute fertility, OOV rate, and vocab coverage for both tokenizers.

    Fertility   = avg(|bpe_tokens| / |word_tokens|) per sample
    OOV rate    = tokens not in the tokenizer's own vocabulary / total tokens
    Coverage    = unique tokens seen / total tokens in all samples
    """
    word_lengths = [len(t) for t in word_tokens_all]
    bpe_lengths  = [len(t) for t in bpe_tokens_all]

    # Fertility: bpe_tokens / word_tokens per sample (skip empties)
    fertilities = [
        b / w for b, w in zip(bpe_lengths, word_lengths) if w > 0
    ]
    fertility = float(np.mean(fertilities)) if fertilities else 0.0

    # Flatten
    all_word_tokens = [t for seq in word_tokens_all for t in seq]
    all_bpe_tokens  = [t for seq in bpe_tokens_all  for t in seq]

    # OOV rate (tokens not in the respective vocab)
    word_oov = sum(1 for t in all_word_tokens if t not in word_vocab) / max(len(all_word_tokens), 1)
    bpe_oov  = sum(1 for t in all_bpe_tokens  if t not in bpe_vocab)  / max(len(all_bpe_tokens),  1)

    # Coverage = unique token types observed / total tokens (type-token ratio variant)
    word_coverage = len(set(all_word_tokens)) / max(len(all_word_tokens), 1)
    bpe_coverage  = len(set(all_bpe_tokens))  / max(len(all_bpe_tokens),  1)

    # Avg sequence length
    avg_word_len = float(np.mean(word_lengths)) if word_lengths else 0.0
    avg_bpe_len  = float(np.mean(bpe_lengths))  if bpe_lengths  else 0.0

    return {
        "fertility":          round(fertility, 4),
        "word_oov_rate":      round(word_oov, 4),
        "bpe_oov_rate":       round(bpe_oov, 4),
        "word_vocab_coverage": round(word_coverage, 4),
        "bpe_vocab_coverage":  round(bpe_coverage, 4),
        "avg_word_seq_len":   round(avg_word_len, 2),
        "avg_bpe_seq_len":    round(avg_bpe_len, 2),
        "n_samples":          len(samples),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def run():
    logging.info("=" * 60)
    logging.info("Phase 5: Tokenization Comparison")
    logging.info("=" * 60)

    # ------------------------------------------------------------------
    # 1. Load GPT-2 tokenizer
    # ------------------------------------------------------------------
    logging.info("📦 Loading GPT-2 BPE tokenizer...")
    try:
        from transformers import AutoTokenizer
        gpt2_tok = AutoTokenizer.from_pretrained("gpt2")
        bpe_vocab = set(gpt2_tok.get_vocab().keys())
        logging.info(f"   GPT-2 vocab size: {len(bpe_vocab):,}")
    except Exception as e:
        logging.error(f"❌ Could not load GPT-2 tokenizer: {e}")
        logging.error("   Run: pip install transformers")
        sys.exit(1)

    # ------------------------------------------------------------------
    # 2. Load word-level vocab from Phase 2 artefacts (if available)
    # ------------------------------------------------------------------
    vocab_path = Path("data/processed/vocab.json")
    if vocab_path.exists():
        with open(vocab_path) as f:
            word_vocab_dict = json.load(f)
        word_vocab = set(word_vocab_dict.keys())
        logging.info(f"   Loaded project word vocab: {len(word_vocab):,} tokens")
    else:
        logging.warning("⚠️  data/processed/vocab.json not found – building vocab from samples.")
        word_vocab = set()  # will be populated below

    # ------------------------------------------------------------------
    # 3. Gather code samples
    # ------------------------------------------------------------------
    # C samples: try to use real Devign data
    c_samples: list[str] = []
    parquet_path = Path("data/raw/devign_raw.parquet")
    if parquet_path.exists():
        logging.info("   Using Devign parquet for C samples...")
        df = pd.read_parquet(parquet_path)
        preprocessor = CodePreprocessor({"model": {"max_len": 512}})
        c_samples = (
            df["func"]
            .dropna()
            .sample(min(50, len(df)), random_state=42)
            .apply(preprocessor.clean_code)
            .tolist()
        )
    else:
        logging.warning("⚠️  Devign parquet not found – using built-in C fallback samples.")
        c_samples = FALLBACK_C_SAMPLES

    # Python samples: use CodeSearchNet-style built-in snippets
    python_samples: list[str] = FALLBACK_PYTHON_SAMPLES
    try:
        from datasets import load_dataset
        logging.info("   Trying to load CodeSearchNet Python test split...")
        ds = load_dataset(
            "code_search_net",
            "python",
            split="test",
            trust_remote_code=True,
        )
        python_samples = (
            pd.DataFrame(ds)["whole_func_string"]
            .dropna()
            .sample(min(50, len(ds)), random_state=42)
            .tolist()
        )
        logging.info(f"   Loaded {len(python_samples)} Python samples from CodeSearchNet.")
    except Exception:
        logging.warning("⚠️  CodeSearchNet not available – using built-in Python fallback samples.")

    logging.info(f"   C samples: {len(c_samples)} | Python samples: {len(python_samples)}")

    # ------------------------------------------------------------------
    # 4. Tokenize with both tokenizers
    # ------------------------------------------------------------------
    results = {}
    per_sample_rows = []

    for lang, samples in [("C (Devign)", c_samples), ("Python (CodeSearchNet)", python_samples)]:
        logging.info(f"\n🔍 Tokenizing [{lang}]...")

        # Word-level
        word_seqs = [word_tokenize(s) for s in samples]

        # If no vocab loaded from disk, build from these samples
        if not word_vocab:
            word_vocab = set(t for seq in word_seqs for t in seq)

        # GPT-2 BPE (batch)
        encoded = gpt2_tok(
            samples,
            add_special_tokens=False,
            return_attention_mask=False,
        )
        bpe_seqs = [
            gpt2_tok.convert_ids_to_tokens(ids)
            for ids in encoded["input_ids"]
        ]

        metrics = compute_tokenizer_metrics(
            samples, word_seqs, bpe_seqs, word_vocab, bpe_vocab
        )
        results[lang] = metrics
        logging.info(f"   Fertility (BPE/Word): {metrics['fertility']:.3f}")
        logging.info(f"   Word OOV rate:        {metrics['word_oov_rate']:.3%}")
        logging.info(f"   BPE  OOV rate:        {metrics['bpe_oov_rate']:.3%}")
        logging.info(f"   Word vocab coverage:  {metrics['word_vocab_coverage']:.3%}")
        logging.info(f"   BPE  vocab coverage:  {metrics['bpe_vocab_coverage']:.3%}")
        logging.info(f"   Avg word seq len:     {metrics['avg_word_seq_len']:.1f}")
        logging.info(f"   Avg BPE  seq len:     {metrics['avg_bpe_seq_len']:.1f}")

        # Collect per-sample rows
        for i, (s, wseq, bseq) in enumerate(zip(samples, word_seqs, bpe_seqs)):
            per_sample_rows.append({
                "language": lang,
                "sample_id": i,
                "word_token_count": len(wseq),
                "bpe_token_count":  len(bseq),
                "fertility": round(len(bseq) / max(len(wseq), 1), 4),
                "word_oov": sum(1 for t in wseq if t not in word_vocab),
                "bpe_oov":  sum(1 for t in bseq if t not in bpe_vocab),
            })

    # ------------------------------------------------------------------
    # 5. Save outputs
    # ------------------------------------------------------------------
    json_path = OUTPUT_DIR / "tokenization_comparison.json"
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)
    logging.info(f"\n💾 Saved summary JSON: {json_path}")

    csv_path = OUTPUT_DIR / "tokenization_comparison.csv"
    df_out = pd.DataFrame(per_sample_rows)
    df_out.to_csv(csv_path, index=False)
    logging.info(f"💾 Saved per-sample CSV: {csv_path}")

    # ------------------------------------------------------------------
    # 6. Print comparison table
    # ------------------------------------------------------------------
    logging.info("\n" + "=" * 60)
    logging.info("TOKENIZATION COMPARISON SUMMARY")
    logging.info("=" * 60)
    header = f"{'Metric':<30} {'C (Devign)':>18} {'Python (CSN)':>18}"
    logging.info(header)
    logging.info("-" * 70)
    metric_labels = {
        "fertility":            "Fertility (BPE/Word ratio)",
        "word_oov_rate":        "Word OOV Rate",
        "bpe_oov_rate":         "BPE  OOV Rate",
        "word_vocab_coverage":  "Word Vocab Coverage",
        "bpe_vocab_coverage":   "BPE  Vocab Coverage",
        "avg_word_seq_len":     "Avg Word Seq Len",
        "avg_bpe_seq_len":      "Avg BPE  Seq Len",
        "n_samples":            "# Samples",
    }
    c_res = results.get("C (Devign)", {})
    py_res = results.get("Python (CodeSearchNet)", {})
    for key, label in metric_labels.items():
        cv = c_res.get(key, "N/A")
        pv = py_res.get(key, "N/A")
        logging.info(f"  {label:<28} {str(cv):>18} {str(pv):>18}")
    logging.info("=" * 60)
    logging.info("✅ Phase 5 complete.")

    return results


if __name__ == "__main__":
    run()
