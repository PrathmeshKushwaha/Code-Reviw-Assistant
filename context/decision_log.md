# 📘 DECISION LOG & TECHNICAL RATIONALE
*P04 Code Review Assistant | DL + NLP Tracks | 2026-04-25*

This document records every major technical decision, the rationale behind it, and real-world alternatives you would face in production. Use this for report writing, viva defense, and future extensions.

---

## 🧩 PHASE-BY-PHASE SUMMARY

### Phase 1: Data Inspection
- **What**: Loaded raw `dataset.json`, handled nested structures, saved as Parquet.
- **Why**: Parquet enables fast columnar I/O for repeated experiments. JSON parsing wrapped in fallback logic to handle research dataset inconsistencies.
- **Real-World Alternative**: Use HuggingFace `datasets` library directly with streaming mode for >100GB corpora. Parquet is standard in ML pipelines (DuckDB, Spark, Delta Lake compatible).

### Phase 2: Preprocessing & Tokenization
- **What**: Word-level regex tokenizer (`\b\w+\b|[^\s\w]`), stratified 70/15/15 split, vocab built on train only.
- **Why**: Friend's initial implementation used word-level. Kept it to avoid breaking Phase 3. Stratification preserves 50/50 buggy/clean ratio. Train-only vocab prevents data leakage.
- **Real-World Alternative**: 
  - **Sub-word (BPE/WordPiece)**: Handles OOV code symbols (`->`, `::`, `__init__`) better. Standard for LLMs.
  - **Tree-sitter AST parsing**: Extracts syntactic structure instead of raw text. Used in GitHub Copilot, CodeQL.
  - **Decision Trade-off**: Word-level is faster to implement and easier to visualize, but hurts generalization to unseen identifiers.

### Phase 3: TextCNN Training
- **What**: Conv1D(3,4,5) → max-pool → BN → Dropout(0.5) → FC. BCEWithLogitsLoss + Adam. Early stopping (patience=2).
- **Why**: Matches rubric exactly. Max-pool extracts strongest signal regardless of position. BN stabilizes gradients. Dropout(0.5) per rubric overrides config(0.3) for grading safety.
- **Real-World Alternative**: 
  - **Transformers (CodeBERT, GraphCodeBERT)**: 10-15% higher accuracy but 100x slower, needs GPU cluster.
  - **GNNs over AST/CFG**: Captures dataflow, but requires complex graph construction.
  - **Decision Trade-off**: TextCNN is lightweight, interpretable, and trains in minutes on CPU/GPU. Ideal for academic baselines and latency-sensitive SaaS tools.

### Phase 4: Filter Visualization
- **What**: Forward hooks capture Conv1D outputs → mean across filters → heatmap aligned to tokens.
- **Why**: Rubric requires interpretability. Hooks are PyTorch-native, zero-training-overhead.
- **Real-World Alternative**: 
  - **Captum/SHAP**: Gradient-based attribution. More accurate but slower.
  - **Attention rollout**: If using Transformers. Not applicable to CNNs.
  - **Decision Trade-off**: Activation magnitude is a proxy, not causation. Good for demos, but production bug detectors use taint analysis + symbolic execution alongside ML.

---

## 🔑 KEY TECHNICAL DECISIONS & FIXES

| Decision | Rationale | Production Impact |
|----------|-----------|-------------------|
| `dtype=torch.float32` for labels | `BCEWithLogitsLoss` requires floats. Original code used `long`. | Prevents silent NaN gradients. Standard in binary classification. |
| `float(model_cfg["lr"])` cast | YAML parser treated `1e-3` as string. | Always cast numeric configs. Use `pydantic` or `omegaconf` in prod for schema validation. |
| Dropout=0.5 (hardcoded) | Rubric explicitly states 0.5. Config said 0.3. | Follow grading rubrics strictly. In prod, tune via Optuna/Ray. |
| Max sequence length = 512 | From config. Covers ~99% of functions. | Padding wastes compute. Prod systems use dynamic batching + bucketing. |
| Word-level tokenizer for CNN | Maintains pipeline compatibility with Phase 2. | Will switch to BPE for NLP track. Prod code analyzers use language-specific lexers. |

---

## 🌍 REAL-WORLD CHOICES YOU'D FACE

### 1. Tokenization Strategy
- **Word-level**: Fast, interpretable, high OOV rate. Good for baselines.
- **BPE (GPT2/CodeT5)**: Lower OOV, handles symbols, larger vocab (~50k). Standard for LLMs.
- **AST-based**: Parses code into syntax trees. Captures semantics, not just text. Used in static analysis tools.
- **Recommendation**: For SaaS code review, use **BPE + AST features** fused in a transformer.

### 2. Model Architecture
- **TextCNN**: Fast, local pattern detection. Misses long-range dependencies (e.g., variable declared in func A, used in func B).
- **LSTM/GRU**: Sequential memory. Slow on CPU, vanishing gradients on long code.
- **Transformers**: Global attention. State-of-the-art but heavy.
- **Recommendation**: Hybrid approach: CNN for fast triage, Transformer for deep review. Exactly what your project demonstrates.

### 3. Evaluation Metrics
- **Accuracy**: Misleading on imbalanced data. Devign is balanced, so acceptable.
- **F1-Score**: Better for bug detection (precision/recall trade-off).
- **AUC-ROC**: Threshold-independent. Good for ranking.
- **BLEU-4**: Weak for code comments (low lexical overlap). Used here for academic compliance.
- **Recommendation**: In prod, use **exact match + semantic similarity (CodeBLEU) + human-in-the-loop scoring**.

### 4. Deployment & Scaling
- **Current**: Local Python + RTX 3050.
- **Prod**: 
  - Model served via **TorchServe** or **Triton Inference Server**
  - Rate-limited LLM API calls cached with **Redis**
  - CI/CD pipeline runs unit tests on tokenizer + model drift detection
  - Monitoring: Prometheus + Grafana for latency/throughput

---

## 📝 HOW TO REPRODUCE & EXTEND

### Reproduce Results
```bash
source venv/bin/activate
python scripts/phase1_inspect.py
python scripts/phase2_preprocess.py
python scripts/phase3_train_cnn.py
python scripts/phase4_visualize_activations.py