# 🔍 Hybrid AI Code Review Assistant
> Fast pattern detection + semantic AI feedback for modern developer workflows.

![Status](https://img.shields.io/badge/Status-Prototype-blue)
![Python](https://img.shields.io/badge/Python-3.11+-green)
![PyTorch](https://img.shields.io/badge/PyTorch-2.x-red)
![License](https://img.shields.io/badge/License-MIT-lightgrey)

---

## 🚀 Overview
Automated code review faces a core trade-off: rule-based linters are fast but rigid, while large language models provide rich feedback but are slow, expensive, and opaque. This project bridges that gap with a **hybrid AI pipeline**:
- A lightweight **TextCNN** triages code in milliseconds, flagging suspicious patterns with interpretable activation heatmaps.
- A **prompt-optimized LLM** generates contextual review comments only when needed, reducing inference costs by ~70%.
- Built for low-latency CI/CD integration, transparent decision-making, and cost-aware scaling.

---

## 🏗️ Architecture & Pipeline
[Raw Code] 
   │
   ├─► Phase 1-2: Ingest → Clean → Tokenize → Tensorize<br>
   │
   ├─► Phase 3: TextCNN Classifier (Conv1D 3,4,5 → Max-Pool → BN → Dropout → FC)<br>
   │      └─ Output: Buggy/Clean probability + confidence<br>
   │
   ├─► Phase 4: Hook-based Activation Visualization (interpretable token-level triggers)<br>
   │
   ├─► Phase 5: Tokenization Benchmarking (BPE vs Word-Level | Python vs C)<br>
   │
   ├─► Phase 6: LLM Prompt Routing (Zero/One/Few-Shot → CodeT5 via HF Inference API)<br>
   │      ─ Caching + exponential backoff for rate-limit resilience<br>
   │
   ─► Phase 7: Evaluation & Integration (BLEU-4, heuristic scoring, side-by-side table, error analysis)<br>


---

## ✨ Key Features
| Feature | Description |
|---------|-------------|
| ⚡ **Sub-10ms Triage** | CNN inference optimized for consumer GPUs; scales to high-throughput pipelines |
| 🔍 **Explainable Predictions** | Forward hooks extract filter activations → heatmap highlights exact suspicious tokens |
| 💰 **Cost-Aware Routing** | Only ~30% of functions (flagged by CNN) trigger LLM calls, cutting API costs significantly |
| 📊 **Tokenization Insights** | Quantifies BPE fertility, OOV rates, and vocab coverage across Python & C |
| 🧠 **Structured Prompting** | Zero/one/few-shot templates with cached responses & polite pacing |
| 📈 **Dual Evaluation** | BLEU-4 + heuristic quality scoring (relevance, clarity, correctness) |
| 📦 **Fully Reproducible** | Deterministic seeding, living documentation, cache-first API design |

---

## 🛠️ Tech Stack
- **Deep Learning**: PyTorch, TextCNN, BCEWithLogitsLoss, Adam, Early Stopping
- **NLP/LLM**: HuggingFace Transformers, CodeT5-small, GPT2 BPE Tokenizer, HF Inference API
- **Data & Utils**: pandas, numpy, scikit-learn, pyyaml, tqdm, matplotlib, seaborn
- **Infrastructure**: Local RTX 3050, venv isolation, MD5-cached API responses, exponential backoff

---

