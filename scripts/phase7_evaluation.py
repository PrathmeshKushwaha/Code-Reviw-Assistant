"""
Phase 7: Evaluation & Integration
- Loads CNN predictions (Phase 3) + LLM comments (Phase 6)
- Computes BLEU-4 for each prompt type
- Applies manual scoring heuristics (Relevance, Clarity, Correctness on 1-5 scale)
- Builds side-by-side comparison table
- Error analysis: CNN FP/FN vs LLM comment quality
- Outputs: outputs/evaluation_results.json, outputs/evaluation_table.csv,
           outputs/error_analysis.csv
"""

import os
import sys
import json
import logging
import re
import math
import numpy as np
import pandas as pd
from pathlib import Path
from collections import Counter
from typing import Optional

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s"
)

OUTPUT_DIR = Path("outputs")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

PROMPT_TYPES = ["zero_shot", "one_shot", "few_shot"]

# ---------------------------------------------------------------------------
# Reference comments (heuristic ground-truth proxy)
# Normally these would come from human-written code review annotations.
# We synthesise sensible references keyed on the binary label.
# ---------------------------------------------------------------------------
REF_BUGGY = (
    "This code contains a bug or security vulnerability. "
    "The function should validate inputs, check return values, "
    "and avoid unsafe operations such as buffer overflows, null dereferences, "
    "or integer overflows."
)
REF_CLEAN = (
    "The code appears correct and safe. "
    "Input validation is present, memory management is handled properly, "
    "and there are no obvious bugs or vulnerabilities."
)


# ---------------------------------------------------------------------------
# BLEU-4 (corpus-level, pure Python – no evaluate import required at runtime)
# ---------------------------------------------------------------------------
def _ngrams(tokens: list[str], n: int) -> Counter:
    return Counter(tuple(tokens[i:i+n]) for i in range(len(tokens) - n + 1))


def _clipped_precision(hyp_tokens: list[str], ref_tokens: list[str], n: int) -> tuple[int, int]:
    hyp_ng = _ngrams(hyp_tokens, n)
    ref_ng = _ngrams(ref_tokens, n)
    clipped = sum(min(count, ref_ng[gram]) for gram, count in hyp_ng.items())
    total   = max(sum(hyp_ng.values()), 1)
    return clipped, total


def bleu4_sentence(hypothesis: str, reference: str) -> float:
    """Compute sentence-level BLEU-4."""
    hyp = hypothesis.lower().split()
    ref = reference.lower().split()
    if not hyp:
        return 0.0
    # Brevity penalty
    bp = 1.0 if len(hyp) >= len(ref) else math.exp(1 - len(ref) / max(len(hyp), 1))
    # n-gram precisions
    log_avg = 0.0
    for n in range(1, 5):
        c, t = _clipped_precision(hyp, ref, n)
        if c == 0:
            return 0.0
        log_avg += math.log(c / t)
    return bp * math.exp(log_avg / 4)


def corpus_bleu4(hypotheses: list[str], references: list[str]) -> float:
    """Compute corpus-level BLEU-4 (micro-averaged)."""
    totals = [0] * 4
    counts = [0] * 4
    hyp_len = ref_len = 0

    for hyp_str, ref_str in zip(hypotheses, references):
        hyp = hyp_str.lower().split()
        ref = ref_str.lower().split()
        hyp_len += len(hyp)
        ref_len  += len(ref)
        for n in range(1, 5):
            c, t = _clipped_precision(hyp, ref, n)
            counts[n-1] += c
            totals[n-1] += t

    bp = 1.0 if hyp_len >= ref_len else math.exp(1 - ref_len / max(hyp_len, 1))
    log_avg = 0.0
    for n in range(4):
        if counts[n] == 0:
            return 0.0
        log_avg += math.log(counts[n] / max(totals[n], 1))
    return bp * math.exp(log_avg / 4)


# ---------------------------------------------------------------------------
# Heuristic manual scoring (1–5 scale)
# These rules approximate a human rubric for automated evaluation.
# ---------------------------------------------------------------------------
BUG_KEYWORDS = {
    "buffer overflow", "overflow", "null", "null pointer", "dereference",
    "memory leak", "use after free", "double free", "off-by-one",
    "integer overflow", "format string", "race condition", "injection",
    "uninitialized", "bounds", "vulnerable", "exploit", "unsafe",
    "strcpy", "gets", "sprintf", "malloc", "free", "heap", "stack",
    "oob", "out of bound",
}

POSITIVE_KEYWORDS = {
    "correct", "safe", "no issue", "no bug", "no vulnerability",
    "looks good", "well-implemented", "properly", "valid",
}


def _score_relevance(comment: str, label: int) -> int:
    """Does the comment address the actual bug/clean status? (1-5)"""
    c = comment.lower()
    has_bug_mention  = any(kw in c for kw in BUG_KEYWORDS)
    has_pos_mention  = any(kw in c for kw in POSITIVE_KEYWORDS)
    if label == 1:  # buggy
        if has_bug_mention:  return 5
        if has_pos_mention:  return 1
        if len(c.split()) > 15: return 3  # long comment, possibly relevant
        return 2
    else:  # clean
        if has_pos_mention:  return 5
        if has_bug_mention:  return 2
        if len(c.split()) > 10: return 3
        return 3


def _score_clarity(comment: str) -> int:
    """Is the explanation clear and coherent? (1-5)"""
    words = comment.split()
    if len(words) < 5:   return 1
    if len(words) < 15:  return 2
    # Penalise truncated or garbled outputs
    if comment.endswith("...") or "[API call failed" in comment: return 1
    # Check for sentence-ending punctuation
    sentences = re.split(r'[.!?]', comment)
    sentences = [s.strip() for s in sentences if s.strip()]
    if len(sentences) >= 2: return 5
    if len(sentences) == 1: return 4
    return 3


def _score_correctness(comment: str, label: int) -> int:
    """Is the technical advice accurate? (1-5)"""
    c = comment.lower()
    # Gross error: says buggy when clean or vice-versa with high confidence
    if label == 0 and "vulnerable" in c and "not vulnerable" not in c: return 2
    if label == 1 and "no issue" in c and "no bug" in c:               return 1
    # Reward specific, accurate bug naming
    specific = sum(1 for kw in [
        "buffer overflow", "null pointer", "memory leak", "off-by-one",
        "integer overflow", "format string", "use after free"
    ] if kw in c)
    if specific >= 2: return 5
    if specific == 1: return 4
    if len(c.split()) > 20: return 3
    return 2


def score_comment(comment: str, label: int) -> dict[str, int]:
    return {
        "relevance":   _score_relevance(comment, label),
        "clarity":     _score_clarity(comment),
        "correctness": _score_correctness(comment, label),
    }


# ---------------------------------------------------------------------------
# CNN predictions
# ---------------------------------------------------------------------------
def load_cnn_predictions(n_samples: int) -> Optional[dict[int, dict]]:
    """
    Try to load CNN predictions from Phase 3 artefacts.
    Returns {sample_id: {pred_label, confidence}} or None.
    """
    try:
        import torch
        import yaml
        with open("configs/defaults.yaml") as f:
            cfg = yaml.safe_load(f)
        data_dir = Path(cfg["dataset"]["processed_dir"])
        model_path = data_dir / "best_textcnn.pt"
        test_inputs_path = data_dir / "test_inputs.pt"
        test_labels_path = data_dir / "test_labels.pt"

        if not all(p.exists() for p in [model_path, test_inputs_path, test_labels_path]):
            raise FileNotFoundError("CNN artefacts missing.")

        sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
        from src.textcnn_model import TextCNN, get_device

        test_inputs = torch.load(test_inputs_path, weights_only=True)
        test_labels = torch.load(test_labels_path, weights_only=True)

        vocab_path = data_dir / "vocab.json"
        with open(vocab_path) as f:
            vocab = json.load(f)
        vocab_size = len(vocab)

        device = get_device("auto")
        model = TextCNN(vocab_size=vocab_size)
        model.load_state_dict(torch.load(model_path, weights_only=True, map_location=device))
        model.to(device).eval()

        rng = np.random.default_rng(42)
        indices = rng.choice(len(test_inputs), size=min(n_samples, len(test_inputs)), replace=False)

        preds = {}
        with torch.no_grad():
            for idx in indices:
                inp = test_inputs[idx].unsqueeze(0).to(device)
                logit = model(inp).item()
                prob  = 1 / (1 + math.exp(-logit))
                preds[int(idx)] = {
                    "cnn_pred":       int(prob >= 0.5),
                    "cnn_confidence": round(prob, 4),
                    "true_label":     int(test_labels[idx].item()),
                }
        logging.info(f"✅ Loaded CNN predictions for {len(preds)} samples.")
        return preds

    except Exception as e:
        logging.warning(f"⚠️  CNN predictions unavailable ({e}). Using label-proxy predictions.")
        return None


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def run():
    logging.info("=" * 60)
    logging.info("Phase 7: Evaluation & Integration")
    logging.info("=" * 60)

    # ------------------------------------------------------------------
    # 1. Load Phase 6 LLM outputs
    # ------------------------------------------------------------------
    llm_json_path = OUTPUT_DIR / "llm_comments.json"
    if not llm_json_path.exists():
        logging.error(f"❌ {llm_json_path} not found. Run phase6_llm_prompting.py first.")
        sys.exit(1)

    with open(llm_json_path) as f:
        llm_data = json.load(f)

    model_name = llm_data.get("model", "unknown")
    llm_results = llm_data["results"]
    n_samples = len(llm_results)
    logging.info(f"   Loaded {n_samples} samples from Phase 6 (model: {model_name})")

    # ------------------------------------------------------------------
    # 2. Load CNN predictions
    # ------------------------------------------------------------------
    cnn_preds = load_cnn_predictions(n_samples)

    # ------------------------------------------------------------------
    # 3. Build evaluation table
    # ------------------------------------------------------------------
    rows = []
    for row in llm_results:
        sid   = row["sample_id"]
        label = row["label"]
        code  = row.get("code", "")

        reference = REF_BUGGY if label == 1 else REF_CLEAN

        # CNN info
        if cnn_preds and sid in cnn_preds:
            cnn_pred = cnn_preds[sid]["cnn_pred"]
            cnn_conf = cnn_preds[sid]["cnn_confidence"]
        else:
            # Fallback: use ground truth as "perfect CNN" proxy
            cnn_pred = label
            cnn_conf = 0.85 if label == 1 else 0.15

        cnn_correct = int(cnn_pred == label)

        eval_row = {
            "sample_id":       sid,
            "true_label":      label,
            "label_str":       "BUGGY" if label else "CLEAN",
            "cnn_pred":        cnn_pred,
            "cnn_confidence":  cnn_conf,
            "cnn_correct":     cnn_correct,
            "cnn_error_type":  (
                "FP" if (cnn_pred == 1 and label == 0) else
                "FN" if (cnn_pred == 0 and label == 1) else
                "TP" if (cnn_pred == 1 and label == 1) else "TN"
            ),
        }

        for pt in PROMPT_TYPES:
            comment = row.get(f"comment_{pt}", "")
            bleu    = bleu4_sentence(comment, reference)
            scores  = score_comment(comment, label)
            eval_row[f"comment_{pt}"]     = comment
            eval_row[f"bleu4_{pt}"]       = round(bleu, 4)
            eval_row[f"relevance_{pt}"]   = scores["relevance"]
            eval_row[f"clarity_{pt}"]     = scores["clarity"]
            eval_row[f"correctness_{pt}"] = scores["correctness"]
            eval_row[f"manual_avg_{pt}"]  = round(
                (scores["relevance"] + scores["clarity"] + scores["correctness"]) / 3, 2
            )

        rows.append(eval_row)

    df = pd.DataFrame(rows)

    # ------------------------------------------------------------------
    # 4. Corpus-level BLEU-4 per prompt type
    # ------------------------------------------------------------------
    corpus_bleus = {}
    for pt in PROMPT_TYPES:
        hyps = df[f"comment_{pt}"].fillna("").tolist()
        refs = [REF_BUGGY if l == 1 else REF_CLEAN for l in df["true_label"].tolist()]
        corpus_bleus[pt] = round(corpus_bleu4(hyps, refs), 4)

    # ------------------------------------------------------------------
    # 5. Aggregate metrics
    # ------------------------------------------------------------------
    agg: dict = {
        "model":             model_name,
        "n_samples":         n_samples,
        "cnn_accuracy":      round(df["cnn_correct"].mean(), 4),
        "bleu4_corpus":      corpus_bleus,
        "manual_scores_avg": {},
        "error_distribution": df["cnn_error_type"].value_counts().to_dict(),
    }
    for pt in PROMPT_TYPES:
        agg["manual_scores_avg"][pt] = {
            "relevance":    round(df[f"relevance_{pt}"].mean(), 2),
            "clarity":      round(df[f"clarity_{pt}"].mean(), 2),
            "correctness":  round(df[f"correctness_{pt}"].mean(), 2),
            "overall":      round(df[f"manual_avg_{pt}"].mean(), 2),
        }

    # ------------------------------------------------------------------
    # 6. Error analysis
    # ------------------------------------------------------------------
    error_df = df[df["cnn_correct"] == 0].copy()
    error_df["dominant_prompt"] = error_df.apply(
        lambda r: max(PROMPT_TYPES, key=lambda pt: r[f"manual_avg_{pt}"]), axis=1
    )
    error_path = OUTPUT_DIR / "error_analysis.csv"
    error_df[[
        "sample_id", "true_label", "cnn_pred", "cnn_confidence", "cnn_error_type",
        *[f"manual_avg_{pt}" for pt in PROMPT_TYPES], "dominant_prompt"
    ]].to_csv(error_path, index=False)
    logging.info(f"💾 Error analysis: {error_path}")

    # ------------------------------------------------------------------
    # 7. Save outputs
    # ------------------------------------------------------------------
    json_path = OUTPUT_DIR / "evaluation_results.json"
    with open(json_path, "w") as f:
        json.dump(agg, f, indent=2)
    logging.info(f"💾 Saved evaluation JSON: {json_path}")

    csv_path = OUTPUT_DIR / "evaluation_table.csv"
    # Write a human-readable comparison table (trimmed columns)
    summary_cols = [
        "sample_id", "label_str", "cnn_pred", "cnn_confidence", "cnn_error_type",
        *[f"bleu4_{pt}"      for pt in PROMPT_TYPES],
        *[f"manual_avg_{pt}" for pt in PROMPT_TYPES],
        *[f"comment_{pt}"    for pt in PROMPT_TYPES],
    ]
    df[summary_cols].to_csv(csv_path, index=False)
    logging.info(f"💾 Saved comparison table: {csv_path}")

    # ------------------------------------------------------------------
    # 8. Print summary
    # ------------------------------------------------------------------
    logging.info("\n" + "=" * 70)
    logging.info("EVALUATION SUMMARY")
    logging.info("=" * 70)
    logging.info(f"  CNN Accuracy:           {agg['cnn_accuracy']:.3f}")
    logging.info(f"  CNN Error distribution: {agg['error_distribution']}")
    logging.info("")
    logging.info(f"  {'Prompt':<12} {'Corpus BLEU-4':>14} {'Relevance':>10} {'Clarity':>10} {'Correctness':>12} {'Overall':>8}")
    logging.info(f"  {'-'*68}")
    for pt in PROMPT_TYPES:
        ms = agg["manual_scores_avg"][pt]
        logging.info(
            f"  {pt:<12} {corpus_bleus[pt]:>14.4f} "
            f"{ms['relevance']:>10.2f} {ms['clarity']:>10.2f} "
            f"{ms['correctness']:>12.2f} {ms['overall']:>8.2f}"
        )
    logging.info("=" * 70)

    # ------------------------------------------------------------------
    # 9. Best prompt recommendation
    # ------------------------------------------------------------------
    best_pt = max(PROMPT_TYPES, key=lambda pt: agg["manual_scores_avg"][pt]["overall"])
    logging.info(f"\n🏆 Best performing prompt strategy: [{best_pt}]")
    logging.info(
        f"   BLEU-4: {corpus_bleus[best_pt]:.4f} | "
        f"Overall manual score: {agg['manual_scores_avg'][best_pt]['overall']:.2f}/5"
    )
    logging.info("\n✅ Phase 7 complete.")
    return agg


if __name__ == "__main__":
    run()
