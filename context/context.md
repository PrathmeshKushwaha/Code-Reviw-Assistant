You're absolutely right. The current `CONTEXT.md` is too high-level. Let me create a **truly self-contained, AI-agent portable version** that includes actual code snippets, implementation details, and exact state documentation.

Here's a **completely rewritten CONTEXT.md** that another AI agent could use to resume work without any prior conversation history:

---

```markdown
# 📋 P04 CODE REVIEW ASSISTANT - PROJECT CONTEXT
*Last Updated: 2026-04-25 | Status: DL Track ✅ | NLP Track 🚧 | For AI Agent Handoff*

---

## 🎯 PROJECT OVERVIEW
**Goal**: Hybrid code review assistant with CNN bug classifier + LLM prompt engineering  
**Dataset**: Devign (27,316 C functions, 50/50 buggy/clean)  
**Hardware**: Local Python + RTX 3050 GPU  
**Timeline**: 8 days (~30 hrs)  
**Current Progress**: 60% complete (DL Track done, NLP Track pending)

---

## 🗂️ COMPLETE REPO STRUCTURE WITH FILE PURPOSES

```
P04_CodeReviewAssistant/
├── configs/
│   └── defaults.yaml          # SEE SECTION 3 BELOW FOR EXACT CONTENTS
├── data/
│   ├── raw/
│   │   ├── dataset.json       # Devign raw JSON (downloaded from GitHub)
│   │   └── devign_raw.parquet # Phase 1 output: cleaned DataFrame
│   └── processed/
│       ├── train_inputs.pt    # Shape: [19121, 512], dtype: torch.long
│       ├── train_labels.pt    # Shape: [19121], dtype: torch.float32
│       ├── val_inputs.pt      # Shape: [2732, 512]
│       ├── val_labels.pt      # Shape: [2732]
│       ├── test_inputs.pt     # Shape: [5463, 512]
│       ├── test_labels.pt     # Shape: [5463]
│       ├── vocab.json         # 124,645 tokens (word-level)
│       ├── best_textcnn.pt    # Trained model weights
│       └── metrics.json       # SEE SECTION 5 FOR EXACT METRICS
├── scripts/
│   ├── phase1_inspect.py      # SEE SECTION 4.1
│   ├── phase2_preprocess.py   # SEE SECTION 4.2
│   ├── phase3_train_cnn.py    # SEE SECTION 4.3
│   └── phase4_visualize_activations.py  # SEE SECTION 4.4
├── src/
│   ├── data_processor.py      # SEE SECTION 4.2 (CodePreprocessor class)
│   ├── embedding_utils.py     # Word2Vec helpers (unused)
│   ├── textcnn_model.py       # SEE SECTION 4.3 (TextCNN class)
│   └── __init__.py
├── outputs/
│   └── visualizations/
│       ├── activation_sample_10.png
│       ├── activation_sample_450.png
│       └── activation_sample_980.png
├── requirements.txt           # SEE SECTION 2
├── CONTEXT.md                 # ← THIS FILE
└── DECISION_LOG.md            # Rationale for all technical choices
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

**Installation**: `pip install -r requirements.txt`

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

## 3️⃣ KEY IMPLEMENTATIONS (CODE SNIPPETS)

### 3.1 Phase 1: `scripts/phase1_inspect.py`
**Purpose**: Load JSON → EDA → save Parquet  
**Key Functions**:
- `load_config()`: Loads YAML from `configs/defaults.yaml`
- `load_devign_json()`: Handles nested JSON structures, returns DataFrame
- `inspect_and_save()`: Prints EDA stats, saves to `devign_raw.parquet`

**Output**: `data/raw/devign_raw.parquet` (27,316 rows, columns: `func`, `target`, `id`, `project`)

---

### 3.2 Phase 2: `scripts/phase2_preprocess.py` + `src/data_processor.py`

**`CodePreprocessor` Class** (`src/data_processor.py`):
```python
class CodePreprocessor:
    def __init__(self, config):
        self.max_len = config['model']['max_len']
        self.vocab = {'<PAD>': 0, '<UNK>': 1}
        self.tokenizer_regex = re.compile(r'\b\w+\b|[^\s\w]')  # Word+symbol tokenizer
    
    def clean_code(self, code):
        # Strip C comments, normalize whitespace
        code = re.sub(r'/\*.*?\*/', '', code, flags=re.DOTALL)
        code = re.sub(r'//.*$', '', code, flags=re.MULTILINE)
        return ' '.join(code.split()).strip()
    
    def tokenize(self, code):
        return self.tokenizer_regex.findall(code)
    
    def build_vocab(self, train_texts):
        # Build vocab on TRAIN ONLY (prevents leakage)
        self.vocab = {'<PAD>': 0, '<UNK>': 1}
        for token in sorted(set(t for text in train_texts for t in self.tokenize(text))):
            self.vocab[token] = len(self.vocab)
        return self.vocab
    
    def encode(self, tokens):
        ids = [self.vocab.get(t, 1) for t in tokens]  # 1 = <UNK>
        ids = ids[:self.max_len]  # Truncate
        ids.extend([0] * (self.max_len - len(ids)))  # Pad
        return torch.tensor(ids, dtype=torch.long)
```

**Phase 2 Pipeline**:
1. Loads `devign_raw.parquet`
2. Stratified split: 70% train, 10% val, 20% test (per config)
3. Builds vocab on train set only
4. Encodes all splits → saves as `.pt` tensors
5. **CRITICAL**: Labels saved as `torch.float32` (not `long`) for BCEWithLogitsLoss

**Output Files**:
- `data/processed/{train,val,test}_inputs.pt` → Shape `[N, 512]`
- `data/processed/{train,val,test}_labels.pt` → Shape `[N]`, dtype `float32`
- `data/processed/vocab.json` → 124,645 tokens

---

### 3.3 Phase 3: `src/textcnn_model.py` + `scripts/phase3_train_cnn.py`

**`TextCNN` Class** (`src/textcnn_model.py`):
```python
class TextCNN(nn.Module):
    def __init__(self, vocab_size, embed_dim=128, num_filters=128, 
                 filter_sizes=(3, 4, 5), dropout=0.5):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.convs = nn.ModuleList([
            nn.Conv1d(embed_dim, num_filters, k) for k in filter_sizes
        ])
        self.bn = nn.BatchNorm1d(num_filters * len(filter_sizes))  # 384
        self.dropout = nn.Dropout(dropout)  # 0.5 per rubric
        self.fc = nn.Linear(num_filters * len(filter_sizes), 1)
    
    def forward(self, x):
        x = self.embedding(x).permute(0, 2, 1)  # [B, E, L]
        conv_out = [F.relu(conv(x)) for conv in self.convs]
        pooled = [F.max_pool1d(c, c.size(2)).squeeze(2) for c in conv_out]
        x = torch.cat(pooled, 1)
        x = self.dropout(self.bn(x))
        return self.fc(x).squeeze(1)
```

**Training Loop** (`scripts/phase3_train_cnn.py`):
- Loads tensors from Phase 2
- **Device**: Auto-detects GPU (RTX 3050)
- **Loss**: `BCEWithLogitsLoss()`
- **Optimizer**: `Adam(lr=float(model_cfg["lr"]))` ← cast to float!
- **Early Stopping**: Patience=2 epochs
- **Metrics**: Accuracy, F1, AUC-ROC computed per epoch
- **Saves**: `data/processed/best_textcnn.pt` (best validation loss)

**Final Metrics** (saved to `data/processed/metrics.json`):
```json
{
  "test_results": {
    "accuracy": 0.607,
    "f1": 0.501,
    "auc_roc": 0.661
  }
}
```

---

### 3.4 Phase 4: `scripts/phase4_visualize_activations.py`

**Purpose**: Extract CNN filter activations and generate heatmaps  
**Method**:
1. Registers forward hooks on each Conv1D layer
2. Runs inference on test samples
3. Captures activations: `[1, 128, seq_len']` per filter
4. Averages across 128 filters → `[seq_len']`
5. Aligns with original tokens
6. Plots heatmaps using seaborn

**Output**: `outputs/visualizations/activation_sample_{idx}.png`  
**Interpretation**: Yellow = high activation (buggy patterns like `malloc`, `if(!ptr)`, pointer arithmetic)

---

## 4️⃣ HOW TO VERIFY EACH COMPONENT

### Verify Phase 1:
```bash
python scripts/phase1_inspect.py
# Expected: data/raw/devign_raw.parquet exists, 27,316 rows
```

### Verify Phase 2:
```bash
python scripts/phase2_preprocess.py
# Expected: 6 .pt files in data/processed/, vocab.json with 124,645 entries
```

### Verify Phase 3:
```bash
python src/textcnn_model.py  # Tests forward pass
python scripts/phase3_train_cnn.py  # Full training
# Expected: best_textcnn.pt, metrics.json with Acc~0.60, F1~0.50
```

### Verify Phase 4:
```bash
python scripts/phase4_visualize_activations.py
# Expected: 3 PNG files in outputs/visualizations/
```

---

## 5️⃣ KNOWN ISSUES & FIXES APPLIED

| Issue | Symptom | Fix Applied |
|-------|---------|-------------|
| YAML `lr` as string | `TypeError: '<=' not supported between 'float' and 'str'` | Cast: `lr=float(model_cfg["lr"])` in phase3_train_cnn.py |
| Labels as `torch.long` | BCEWithLogitsLoss expects float32 | Changed dtype to `torch.float32` in data_processor.py line ~90 |
| Dropout mismatch | Config says 0.3, rubric says 0.5 | Hardcoded `dropout=0.5` in TextCNN __init__ |
| Path import error | `NameError: name 'Path' not defined` | Moved `from pathlib import Path` before usage in textcnn_model.py |

---

## 6️⃣ NEXT STEPS (NLP TRACK - NOT STARTED)

### Phase 5: Tokenization Comparison
- [ ] Load Python dataset (e.g., CodeSearchNet Python test split)
- [ ] Tokenize Python + C samples with:
  - GPT2 BPE tokenizer (`AutoTokenizer.from_pretrained("gpt2")`)
  - Current word-level tokenizer
- [ ] Compute metrics:
  - Fertility: `num_subword_tokens / num_word_tokens`
  - OOV rate: `% tokens not in base vocab`
  - Vocab coverage: `unique_tokens / total_tokens`
- [ ] Save comparison table

### Phase 6: LLM Prompting
- [ ] Design 3 prompt templates:
  - Zero-shot: `"Review this code for bugs:\n{code}\nComment:"`
  - One-shot: + 1 example
  - Few-shot: + 3 examples (buggy/clean pairs)
- [ ] Integrate HuggingFace Inference API:
  ```python
  from huggingface_hub import InferenceClient
  client = InferenceClient(model="Salesforce/codet5p-220m")
  comment = client.text_generation(prompt, max_new_tokens=128)
  ```
- [ ] Implement caching + rate limit handling
- [ ] Generate comments for 50 test samples

### Phase 7: Evaluation & Integration
- [ ] Compute BLEU-4: `evaluate.load("bleu")`
- [ ] Manual scoring rubric (1-5 scale):
  - Relevance: Does comment address actual bugs?
  - Clarity: Is explanation understandable?
  - Correctness: Is technical advice accurate?
- [ ] Build comparison table:
  ```
  | Code Sample | Ground Truth | CNN Pred | CNN Conf | LLM Prompt | LLM Comment | BLEU | Manual Score |
  ```
- [ ] Error analysis: CNN FP/FN vs LLM quality

---

## 7️⃣ HOW TO RESUME WORK (STEP-BY-STEP)

### If Starting Fresh:
```bash
# 1. Clone repo, activate venv
git clone <repo_url>
cd P04_CodeReviewAssistant
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows

# 2. Install dependencies
pip install -r requirements.txt

# 3. Download dataset
# Get dataset.json from https://github.com/epicosy/devign/blob/master/data/raw/dataset.json
# Place in data/raw/dataset.json

# 4. Run all phases sequentially
python scripts/phase1_inspect.py
python scripts/phase2_preprocess.py
python scripts/phase3_train_cnn.py
python scripts/phase4_visualize_activations.py

# 5. Start NLP track
# ← NEW AI AGENT SHOULD START HERE →
```

### If DL Track Already Done:
```bash
# Verify artifacts exist
ls -lh data/processed/*.pt data/processed/best_textcnn.pt

# Start NLP track immediately
python scripts/phase5_tokenize_compare.py  # ← Create this next
```

---

## 8️⃣ GRADING ALIGNMENT CHECKLIST

| Unit | Requirement | Status | Evidence |
|------|-------------|--------|----------|
| CSR311-I | CNN architecture | ✅ | `src/textcnn_model.py`: Conv1D(3,4,5) + max-pool |
| CSR311-III | Filter visualization | ✅ | `scripts/phase4_visualize_activations.py` + 3 heatmaps |
| CSR311 | Metrics (Acc, F1, AUC) | ✅ | `data/processed/metrics.json` |
| CSR322-I | Tokenization comparison | 🚧 | Phase 5 pending |
| CSR322-II | Prompt engineering | 🚧 | Phase 6 pending |
| CSR322-IV | BLEU-4 + manual eval | 🚧 | Phase 7 pending |
| Integration | Side-by-side table | 🚧 | Phase 7 pending |

---
