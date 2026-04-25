# 📋 CONTEXT.md — P04 Code Review Assistant
*Last Updated: 2026-04-25 | Status: DL Track ✅ Complete | NLP Track 🚧 Pending*

---

## 🎯 Project Goal (Rubric-Aligned)
Hybrid code review assistant with two equal-weight tracks:
1. **DL Track (CSR311)**: TextCNN classifier for buggy/clean C functions (Devign)
2. **NLP Track (CSR322)**: LLM prompt engineering + tokenization analysis for code review comments

**Timeline**: 8 days (~30 hrs) | **Hardware**: Local Python + RTX 3050

---

## 🗂️ Current Repo Structure

P04_CodeReviewAssistant/
├── configs/
│   └── defaults.yaml          # ✅ Verified (max_len=512, lr=1e-3, device=auto)
├── data/
│   ├── raw/                   # ✅ dataset.json + devign_raw.parquet
│   └── processed/             # ✅ train/val/test tensors, vocab.json, best_textcnn.pt, metrics.json
├── scripts/
│   ├── phase1_inspect.py      # ✅ Loads JSON → EDA → Parquet
│   ├── phase2_preprocess.py   # ✅ Tokenizes → splits → saves .pt tensors
│   ├── phase3_train_cnn.py    # ✅ Training loop + metrics + early stopping
│   └── phase4_visualize_activations.py  # ✅ Filter activation heatmaps
├── src/
│   ├── data_processor.py      # ✅ CodePreprocessor (word-level tokenizer)
│   ├── embedding_utils.py     # ✅ Word2Vec helpers (unused but present)
│   ├── textcnn_model.py       # ✅ Rubric-compliant TextCNN architecture
│   └── init.py
├── outputs/
│   └── visualizations/        # ✅ activation_sample_10.png, etc.
├── requirements.txt           # ✅ Updated (torch, transformers, evaluate, etc.)
├── CONTEXT.md                 # ← This file
└── DECISION_LOG.md            # 🆕 Detailed technical rationale & real-world context


---

## ✅ COMPLETED: DL Track (CSR311)
| Requirement | Status | Details |
|-------------|--------|---------|
| Devign dataset loaded & cleaned | ✅ | 27,316 samples, 50/50 split, saved as Parquet |
| Sub-word tokenization | ⚠️ Partial | Used word-level regex for CNN pipeline stability; BPE analysis deferred to NLP track |
| TextCNN architecture | ✅ | Conv1D(3,4,5) → max-over-time → BN → Dropout(0.5) → FC |
| Training loop | ✅ | BCEWithLogitsLoss + Adam + early stopping (patience=2) |
| Metrics | ✅ | Acc: 0.607, F1: 0.501, AUC-ROC: 0.661 (saved to `data/processed/metrics.json`) |
| Filter visualization | ✅ | Heatmaps generated for 3 test samples; high activation on pointer/arith patterns |

---

## 🚧 PENDING: NLP Track (CSR322)
- [ ] BPE (GPT2) vs word tokenization comparison (Python vs C)
- [ ] Prompt templates: zero-shot, one-shot, few-shot (3 examples)
- [ ] HuggingFace Inference API integration (CodeT5 or similar)
- [ ] BLEU-4 scoring + manual rubric (relevance/clarity/correctness)
- [ ] Side-by-side comparison table: CNN prediction | LLM comment per sample

---

## 📊 Key Artifacts & Locations
| Artifact | Path | Notes |
|----------|------|-------|
| Trained model | `data/processed/best_textcnn.pt` | Loaded via `torch.load(..., weights_only=True)` |
| Vocabulary | `data/processed/vocab.json` | 124,645 tokens (word-level) |
| Tensors | `data/processed/{train,val,test}_inputs.pt` | Shape: `[N, 512]`, dtype `torch.long` |
| Labels | `data/processed/{train,val,test}_labels.pt` | Shape: `[N]`, dtype `torch.float32` |
| Metrics | `data/processed/metrics.json` | Epoch history + test results |
| Visualizations | `outputs/visualizations/` | 3 heatmap PNGs |

---

## 🎯 Immediate Next Steps (Priority)
1. **Tokenization Analysis Script** (`scripts/phase5_tokenize_compare.py`)
2. **LLM Prompting Pipeline** (`scripts/phase6_llm_review.py`)
3. **Evaluation & Comparison Table** (`scripts/phase7_evaluate_nlp.py`)
4. **Final Report Compilation** (align with Units I, II, III, IV)

---

## 🤝 Team Handoff Notes
- **Environment**: `venv` activated, CUDA 12.1, PyTorch 2.x
- **Known Quirks**: 
  - `defaults.yaml` loads `lr` as string → cast to `float()` in training script
  - Dropout hardcoded to `0.5` (rubric) instead of config `0.3`
  - Word-level tokenizer used for CNN; BPE will be added separately for NLP track
- **How to Resume**: 
  1. Activate venv
  2. Run `python scripts/phase3_train_cnn.py` (if retraining needed)
  3. Start NLP track from scratch (no dependencies on DL tensors)