# 📋 P04 CODE REVIEW ASSISTANT - PROJECT CONTEXT
*Last Updated: 2026-04-26 | Status: DL Track ✅ | NLP Track ✅ | For AI Agent Handoff*

---

## 🎯 PROJECT OVERVIEW
**Goal**: Hybrid code review assistant with CNN bug classifier + LLM prompt engineering  
**Dataset**: Devign (27,316 C functions, 50/50 buggy/clean)  
**Hardware**: Local Python + RTX 3050 GPU  
**Timeline**: 8 days (~30 hrs)  
**Current Progress**: 100% complete (DL Track ✅, NLP Track ✅)

---

## 🗂️ COMPLETE REPO STRUCTURE WITH FILE PURPOSES

```
P04_CodeReviewAssistant/
├── configs/
│   └── defaults.yaml
├── data/
│   ├── raw/
│   │   ├── dataset.json
│   │   └── devign_raw.parquet
│   └── processed/
│       ├── train_inputs.pt    # Shape: [19121, 512], dtype: torch.long
│       ├── train_labels.pt    # Shape: [19121], dtype: torch.float32
│       ├── val_inputs.pt      # Shape: [2732, 512]
│       ├── val_labels.pt      # Shape: [2732]
│       ├── test_inputs.pt     # Shape: [5463, 512]
│       ├── test_labels.pt     # Shape: [5463]
│       ├── vocab.json         # 124,645 tokens (word-level)
│       ├── best_textcnn.pt    # Trained model weights
│       └── metrics.json
├── scripts/
│   ├── phase1_inspect.py
│   ├── phase2_preprocess.py
│   ├── phase3_train_cnn.py
│   ├── phase4_visualize_activations.py
│   ├── phase5_tokenize_compare.py     # NEW - Tokenization comparison
│   ├── phase6_llm_prompting.py        # NEW - LLM prompt engineering
│   └── phase7_evaluation.py           # NEW - Evaluation & integration
├── src/
│   ├── data_processor.py
│   ├── embedding_utils.py
│   ├── textcnn_model.py
│   └── __init__.py
├── outputs/
│   ├── visualizations/
│   │   ├── activation_sample_10.png
│   │   ├── activation_sample_450.png
│   │   └── activation_sample_980.png
│   ├── tokenization_comparison.json   # Phase 5 summary metrics
│   ├── tokenization_comparison.csv    # Phase 5 per-sample data
│   ├── llm_comments.json              # Phase 6 generated comments
│   ├── llm_comments.csv               # Phase 6 tabular view
│   ├── llm_cache.json                 # Phase 6 API response cache
│   ├── evaluation_results.json        # Phase 7 aggregate metrics
│   ├── evaluation_table.csv           # Phase 7 side-by-side table
│   └── error_analysis.csv             # Phase 7 CNN FP/FN analysis
├── requirements.txt
├── CONTEXT.md                         # THIS FILE
└── DECISION_LOG.md
```

---

## 1️⃣ REQUIREMENTS.TXT (EXACT CONTENTS)

```txt
# Core ML
torch>=2.0.0
torchvision>=0.15.0
torchaudio>=2.0.0

# HuggingFace ecosystem
transformers>=4.30.0
datasets>=2.14.0
huggingface_hub>=0.16.0
evaluate>=0.4.0

# Data & Utils
pandas>=2.1.0
numpy>=1.24.0
scikit-learn>=1.3.0
pyyaml>=6.0
tqdm>=4.66.0
matplotlib>=3.7.0
seaborn>=0.12.0
gensim>=4.3.0

# Code tokenization
tokenizers>=0.13.0
```

<<<<<<< HEAD
**Installation**: `pip install -r requirements.txt`

=======
>>>>>>> 786234e1af7e83a1d2dfc59296b5812fe59318d9
---

## 2️⃣ CONFIGS/DEFAULTS.YAML (EXACT CONTENTS)

```yaml
dataset:
  raw_path: "data/raw/dataset.json"
  processed_dir: "data/processed"
  val_split: 0.1
  test_split: 0.2
  random_state: 42

model:
  max_len: 512
  embed_dim: 128
  hidden_dim: 128
  dropout: 0.3
  lr: 1e-3
  batch_size: 32
  epochs: 10
  device: "auto"
```

**⚠️ CRITICAL FIX**: YAML parser loads `lr: 1e-3` as string. Must cast to `float()` in training script.

---

## 3️⃣ KEY IMPLEMENTATIONS

### 3.1 Phases 1–4 (DL Track) — COMPLETE
See existing scripts. Final CNN metrics:
```json
{"test_results": {"accuracy": 0.607, "f1": 0.501, "auc_roc": 0.661}}
```

---

### 3.2 Phase 5: `scripts/phase5_tokenize_compare.py`

**Purpose**: Compare GPT-2 BPE tokenizer vs. project word-level tokenizer on C and Python code.

**Metrics computed**:
- **Fertility** = avg(|bpe_tokens| / |word_tokens|) per sample
- **OOV rate** = tokens not in the tokenizer's own vocabulary / total tokens
- **Vocab coverage** = unique token types / total tokens
- **Average sequence length** per tokenizer

**Data sources**:
- C samples: Devign parquet if available, falls back to 5 built-in snippets
- Python samples: HuggingFace CodeSearchNet test split if available, falls back to 5 built-in snippets

**Outputs**:
- `outputs/tokenization_comparison.json`
- `outputs/tokenization_comparison.csv`

**Run**: `python scripts/phase5_tokenize_compare.py`

---

### 3.3 Phase 6: `scripts/phase6_llm_prompting.py`

**Purpose**: Generate bug-review comments for 50 test samples using 3 prompt strategies.

**Prompt templates**:
| Template | Description |
|----------|-------------|
| `zero_shot` | Direct code + "Review comment:" instruction |
| `one_shot` | + 1 annotated strcpy/null-check example |
| `few_shot` | + 3 examples (gets overflow, safe_div clean, off-by-one malloc) |

**Model**: `Salesforce/codet5p-220m` via HuggingFace Inference API  
Override: `HF_MODEL=your/model N_SAMPLES=10 python scripts/phase6_llm_prompting.py`

**Caching**: All API responses cached in `outputs/llm_cache.json` (MD5 keyed). Re-runs are free.

**Rate limiting**: Exponential backoff (2^attempt seconds) + 0.5s polite pacing.

**Outputs**:
- `outputs/llm_comments.json`
- `outputs/llm_comments.csv`

**Run**: `python scripts/phase6_llm_prompting.py`

---

### 3.4 Phase 7: `scripts/phase7_evaluation.py`

**Purpose**: BLEU-4 + heuristic manual scoring + CNN error analysis + final comparison table.

**BLEU-4**: Pure-Python implementation (no external `evaluate` dependency at runtime).
- Sentence-level per sample, corpus-level per prompt type.
- References: synthetic label-keyed strings (REF_BUGGY / REF_CLEAN).

**Manual scoring rubric** (heuristic, 1–5 scale):
| Dimension | Logic |
|-----------|-------|
| Relevance | Keyword presence + label alignment |
| Clarity | Sentence count, length, truncation check |
| Correctness | Specific bug-type naming, no false claims |

**CNN error classification**: TP / TN / FP / FN per sample.

**Outputs**:
- `outputs/evaluation_results.json` — aggregate BLEU-4, manual scores, CNN accuracy
- `outputs/evaluation_table.csv` — side-by-side table
- `outputs/error_analysis.csv` — FP/FN samples with best prompt type

**Expected `evaluation_results.json` schema**:
```json
{
  "model": "Salesforce/codet5p-220m",
  "n_samples": 50,
  "cnn_accuracy": 0.607,
  "bleu4_corpus": {"zero_shot": 0.042, "one_shot": 0.057, "few_shot": 0.071},
  "manual_scores_avg": {
    "zero_shot": {"relevance": 3.2, "clarity": 3.5, "correctness": 2.9, "overall": 3.2},
    "one_shot":  {"relevance": 3.6, "clarity": 3.7, "correctness": 3.1, "overall": 3.5},
    "few_shot":  {"relevance": 3.9, "clarity": 3.8, "correctness": 3.4, "overall": 3.7}
  },
  "error_distribution": {"TN": 22, "TP": 18, "FP": 5, "FN": 5}
}
```

**Run**: `python scripts/phase7_evaluation.py`

---

## 4️⃣ HOW TO VERIFY EACH COMPONENT

```bash
python scripts/phase1_inspect.py       # → data/raw/devign_raw.parquet (27,316 rows)
python scripts/phase2_preprocess.py    # → 6 .pt files + vocab.json
python scripts/phase3_train_cnn.py     # → best_textcnn.pt + metrics.json
python scripts/phase4_visualize_activations.py  # → 3 PNG heatmaps
python scripts/phase5_tokenize_compare.py       # → tokenization_comparison.{json,csv}
python scripts/phase6_llm_prompting.py          # → llm_comments.{json,csv}
python scripts/phase7_evaluation.py             # → evaluation_results.json + table + error_analysis
```

---

## 5️⃣ KNOWN ISSUES & FIXES APPLIED

| Issue | Symptom | Fix Applied |
|-------|---------|-------------|
| YAML `lr` as string | TypeError on float comparison | Cast: `lr=float(model_cfg["lr"])` |
| Labels as `torch.long` | BCEWithLogitsLoss dtype error | Changed to `torch.float32` in data_processor.py |
| Dropout mismatch | Config 0.3 vs rubric 0.5 | Hardcoded `dropout=0.5` in TextCNN |
| Path import error | NameError in textcnn_model.py | Moved `from pathlib import Path` before usage |
| HF API rate limits | HTTP 429 | Exponential backoff + llm_cache.json |
| CodeT5+ cold-start | HTTP 503 | 20s wait + retry loop |
| No human BLEU refs | Evaluation impossible | Synthetic label-keyed reference strings |

---

## 6️⃣ GRADING ALIGNMENT CHECKLIST

| Unit | Requirement | Status | Evidence |
|------|-------------|--------|----------|
| CSR311-I | CNN architecture | ✅ | `src/textcnn_model.py`: Conv1D(3,4,5) + max-pool |
| CSR311-III | Filter visualization | ✅ | `scripts/phase4_visualize_activations.py` + 3 heatmaps |
| CSR311 | Metrics (Acc, F1, AUC) | ✅ | `data/processed/metrics.json` |
| CSR322-I | Tokenization comparison | ✅ | `scripts/phase5_tokenize_compare.py` + 2 output files |
| CSR322-II | Prompt engineering | ✅ | `scripts/phase6_llm_prompting.py` (3 templates, cached) |
| CSR322-IV | BLEU-4 + manual eval | ✅ | `scripts/phase7_evaluation.py` + 3 output files |
| Integration | Side-by-side table | ✅ | `outputs/evaluation_table.csv` |

---

## 7️⃣ HOW TO RESUME WORK

### Full pipeline:
```bash
pip install -r requirements.txt
# Download dataset.json → data/raw/dataset.json
python scripts/phase1_inspect.py
python scripts/phase2_preprocess.py
python scripts/phase3_train_cnn.py
python scripts/phase4_visualize_activations.py
python scripts/phase5_tokenize_compare.py
python scripts/phase6_llm_prompting.py
python scripts/phase7_evaluation.py
```

### NLP track only (DL artefacts already exist):
```bash
python scripts/phase5_tokenize_compare.py
python scripts/phase6_llm_prompting.py
python scripts/phase7_evaluation.py
```

### Re-score without API calls:
```bash
# Phase 7 reads llm_comments.json only – no API calls
python scripts/phase7_evaluation.py
```

---

## 8️⃣ ENVIRONMENT VARIABLES

| Variable | Default | Used In | Purpose |
|----------|---------|---------|---------|
| `HF_MODEL` | `Salesforce/codet5p-220m` | phase6 | Override LLM model |
| `N_SAMPLES` | `50` | phase6 | Number of test samples |
