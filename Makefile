# =============================================================================
# P04 Code Review Assistant - Makefile
# =============================================================================
# Usage:
#   make help              # Show this help
#   make all              # Run full pipeline (Phases 1-7)
#   make dl               # Run DL track only (Phases 1-4)
#   make nlp              # Run NLP track only (Phases 5-7, assumes DL done)
#   make phase3           # Run specific phase
#   make clean            # Remove generated outputs (keep raw data)
#   make clean-all        # Remove everything except source code
#   make test             # Quick smoke test of critical components
# =============================================================================

# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------
PYTHON := python
PIP := pip
VENV := .venv
ACTIVATE := $(VENV)/bin/activate
REQS := requirements.txt
CONFIG := configs/defaults.yaml

# Output directories
DATA_RAW := data/raw
DATA_PROC := data/processed
OUTPUTS := outputs

# Dataset source
DATASET_URL := https://raw.githubusercontent.com/epicosy/devign/master/data/raw/dataset.json
DATASET_PATH := $(DATA_RAW)/dataset.json

# Ollama configuration (updated for your setup)
OLLAMA_MODEL ?= qwen2.5-coder:7b-instruct-q4_K_M
OLLAMA_URL ?= http://localhost:11434/api/generate

# API configuration (override via env or command line)
API_PROVIDER ?= auto
N_SAMPLES ?= 50
REQUEST_DELAY ?= 2.0

# Colors for output (Linux/Mac)
BLUE := \033[34m
GREEN := \033[32m
YELLOW := \033[33m
RED := \033[31m
NC := \033[0m  # No Color

# -----------------------------------------------------------------------------
# Default target
# -----------------------------------------------------------------------------
.PHONY: all
all: setup check-dataset phase1 phase2 phase3 phase4 phase5 phase6 phase7 report
	@echo "$(GREEN)✅ Full pipeline complete!$(NC)"
	@echo "📊 Final metrics: $(DATA_PROC)/metrics.json"
	@echo "📈 Evaluation table: $(OUTPUTS)/evaluation_table.csv"

# -----------------------------------------------------------------------------
# Help
# -----------------------------------------------------------------------------
.PHONY: help
help:
	@echo "$(BLUE)P04 Code Review Assistant - Available Commands$(NC)"
	@echo ""
	@echo "  $(GREEN)Pipeline:$(NC)"
	@echo "    make all              Run full pipeline (Phases 1-7)"
	@echo "    make dl               Run DL track only (Phases 1-4)"
	@echo "    make nlp              Run NLP track only (Phases 5-7)"
	@echo "    make phase[N]         Run specific phase (1-7)"
	@echo ""
	@echo "  $(GREEN)Setup & Cleanup:$(NC)"
	@echo "    make setup            Create venv + install dependencies"
	@echo "    make check-dataset    Download Devign dataset if missing"
	@echo "    make clean            Remove outputs (keep raw data)"
	@echo "    make clean-all        Remove all generated files"
	@echo ""
	@echo "  $(GREEN)Testing & Reporting:$(NC)"
	@echo "    make test             Quick smoke test of critical components"
	@echo "    make report           Generate final summary report"
	@echo ""
	@echo "  $(GREEN)Configuration:$(NC)"
	@echo "    API_PROVIDER=ollama make phase6   # Use Ollama (default: $(OLLAMA_MODEL))"
	@echo "    API_PROVIDER=groq make phase6     # Force Groq API"
	@echo "    API_PROVIDER=hf make phase6       # Force HuggingFace API"
	@echo "    N_SAMPLES=20 make phase6          # Process fewer samples"
	@echo "    OLLAMA_MODEL=name make phase6     # Override Ollama model"
	@echo ""

# -----------------------------------------------------------------------------
# Setup
# -----------------------------------------------------------------------------
.PHONY: setup
setup: $(VENV)/.installed

$(VENV)/.installed: $(REQS)
	@echo "$(BLUE)🔧 Setting up environment...$(NC)"
	@test -d $(VENV) || $(PYTHON) -m venv $(VENV)
	. $(ACTIVATE) && $(PIP) install --upgrade pip
	. $(ACTIVATE) && $(PIP) install -r $(REQS)
	. $(ACTIVATE) && $(PIP) install python-dotenv requests  # For .env + Ollama support
	touch $@
	@echo "$(GREEN)✅ Environment ready$(NC)"

# -----------------------------------------------------------------------------
# Dataset
# -----------------------------------------------------------------------------
.PHONY: check-dataset
check-dataset: $(DATASET_PATH)

$(DATASET_PATH):
	@echo "$(BLUE)📥 Downloading Devign dataset...$(NC)"
	@mkdir -p $(DATA_RAW)
	@curl -L -o $(DATASET_PATH) $(DATASET_URL) || wget -O $(DATASET_PATH) $(DATASET_URL)
	@echo "$(GREEN)✅ Dataset downloaded: $(DATASET_PATH)$(NC)"

# -----------------------------------------------------------------------------
# Phase Targets
# -----------------------------------------------------------------------------
.PHONY: phase1 phase2 phase3 phase4 phase5 phase6 phase7

phase1: setup check-dataset
	@echo "$(BLUE)▶ Phase 1: Data Inspection$(NC)"
	. $(ACTIVATE) && $(PYTHON) scripts/phase1_inspect.py
	@echo "$(GREEN)✅ Phase 1 complete: $(DATA_RAW)/devign_raw.parquet$(NC)"

phase2: setup check-dataset
	@echo "$(BLUE)▶ Phase 2: Preprocessing$(NC)"
	. $(ACTIVATE) && $(PYTHON) scripts/phase2_preprocess.py
	@echo "$(GREEN)✅ Phase 2 complete: $(DATA_PROC)/*.pt$(NC)"

phase3: setup check-dataset
	@echo "$(BLUE)▶ Phase 3: CNN Training$(NC)"
	. $(ACTIVATE) && $(PYTHON) scripts/phase3_train_cnn.py
	@echo "$(GREEN)✅ Phase 3 complete: $(DATA_PROC)/best_textcnn.pt$(NC)"

phase4: setup
	@echo "$(BLUE)▶ Phase 4: Activation Visualization$(NC)"
	@mkdir -p $(OUTPUTS)/visualizations
	. $(ACTIVATE) && $(PYTHON) scripts/phase4_visualize_activations.py
	@echo "$(GREEN)✅ Phase 4 complete: $(OUTPUTS)/visualizations/*.png$(NC)"

phase5: setup
	@echo "$(BLUE)▶ Phase 5: Tokenization Comparison$(NC)"
	. $(ACTIVATE) && $(PYTHON) scripts/phase5_tokenize_compare.py
	@echo "$(GREEN)✅ Phase 5 complete: $(OUTPUTS)/tokenization_comparison.{json,csv}$(NC)"

phase6: setup
	@echo "$(BLUE)▶ Phase 6: LLM Prompting (Provider: $(API_PROVIDER))$(NC)"
	@echo "   Model: $(OLLAMA_MODEL)"
	. $(ACTIVATE) && \
		export API_PROVIDER=$(API_PROVIDER) && \
		export N_SAMPLES=$(N_SAMPLES) && \
		export REQUEST_DELAY=$(REQUEST_DELAY) && \
		export OLLAMA_MODEL=$(OLLAMA_MODEL) && \
		export OLLAMA_URL=$(OLLAMA_URL) && \
		$(PYTHON) scripts/phase6_llm_prompting.py
	@echo "$(GREEN)✅ Phase 6 complete: $(OUTPUTS)/llm_comments.{json,csv}$(NC)"

phase7: setup
	@echo "$(BLUE)▶ Phase 7: Evaluation & Integration$(NC)"
	@echo "   Using LLM provider: $(API_PROVIDER) | Model: $(OLLAMA_MODEL)"
	. $(ACTIVATE) && \
		API_PROVIDER=$(API_PROVIDER) \
		OLLAMA_MODEL=$(OLLAMA_MODEL) \
		$(PYTHON) scripts/phase7_evaluation.py
	@echo "$(GREEN)✅ Phase 7 complete: $(OUTPUTS)/evaluation_table.csv$(NC)"

# -----------------------------------------------------------------------------
# Track Targets
# -----------------------------------------------------------------------------
.PHONY: dl nlp

dl: phase1 phase2 phase3 phase4
	@echo "$(GREEN)✅ DL Track complete$(NC)"

nlp: phase5 phase6 phase7
	@echo "$(GREEN)✅ NLP Track complete$(NC)"

# -----------------------------------------------------------------------------
# Testing
# -----------------------------------------------------------------------------
.PHONY: test
test: setup
	@echo "$(BLUE)🧪 Running smoke tests...$(NC)"
	@echo "  • Checking Python version..."
	@. $(ACTIVATE) && $(PYTHON) --version
	@echo "  • Checking PyTorch + CUDA..."
	@. $(ACTIVATE) && $(PYTHON) -c "import torch; print(f'  PyTorch: {torch.__version__} | CUDA: {torch.cuda.is_available()}')"
	@echo "  • Checking TextCNN forward pass..."
	@. $(ACTIVATE) && $(PYTHON) src/textcnn_model.py
	@echo "  • Checking config load..."
	@. $(ACTIVATE) && $(PYTHON) -c "import yaml; c=yaml.safe_load(open('$(CONFIG)')); print(f'  Config loaded: max_len={c[\"model\"][\"max_len\"]}')"
	@echo "  • Checking Ollama connectivity..."
	@curl -s http://localhost:11434/api/tags > /dev/null && echo "  ✅ Ollama: responsive" || echo "  ⚠️ Ollama: not responding (optional for local fallback)"
	@echo "$(GREEN)✅ Smoke tests passed$(NC)"

# -----------------------------------------------------------------------------
# Reporting
# -----------------------------------------------------------------------------
.PHONY: report
report:
	@echo "$(BLUE)📊 Generating summary report...$(NC)"
	@. $(ACTIVATE) && $(PYTHON) -c "\
import json, os; \
m = json.load(open('$(DATA_PROC)/metrics.json')) if os.path.exists('$(DATA_PROC)/metrics.json') else {}; \
e = json.load(open('$(OUTPUTS)/evaluation_results.json')) if os.path.exists('$(OUTPUTS)/evaluation_results.json') else {}; \
print('\n=== P04 FINAL SUMMARY ==='); \
print(f\"CNN Metrics: Acc={m.get('test_results',{}).get('accuracy','N/A')}, F1={m.get('test_results',{}).get('f1','N/A')}, AUC={m.get('test_results',{}).get('auc_roc','N/A')}\"); \
print(f\"BLEU-4 (few-shot): {e.get('bleu4_corpus',{}).get('few_shot','N/A')}\"); \
print(f\"Manual Score (few-shot): {e.get('manual_scores_avg',{}).get('few_shot',{}).get('overall','N/A')}/5\"); \
print('✅ Report generated')"
	@echo "$(GREEN)✅ Summary printed$(NC)"

# -----------------------------------------------------------------------------
# Cleanup
# -----------------------------------------------------------------------------
.PHONY: clean clean-all

clean:
	@echo "$(YELLOW)🧹 Cleaning outputs (keeping raw data)...$(NC)"
	rm -rf $(OUTPUTS)/*
	rm -rf $(DATA_PROC)/*.pt $(DATA_PROC)/vocab.json $(DATA_PROC)/metrics.json $(DATA_PROC)/best_textcnn.pt
	rm -rf $(DATA_RAW)/devign_raw.parquet
	@echo "$(GREEN)✅ Cleaned. Raw dataset preserved at $(DATASET_PATH)$(NC)"

clean-all: clean
	@echo "$(YELLOW)🧹 Cleaning everything except source...$(NC)"
	rm -rf $(VENV)
	rm -rf __pycache__ */__pycache__
	rm -rf .pytest_cache
	rm -f .env
	@echo "$(GREEN)✅ Full clean complete. Run 'make setup' to restart.$(NC)"

# -----------------------------------------------------------------------------
# Utility
# -----------------------------------------------------------------------------
.PHONY: shell
shell: setup
	@echo "$(GREEN)🐚 Activating virtual environment...$(NC)"
	@echo "Type 'exit' to leave the shell."
	. $(ACTIVATE) && exec $(SHELL)

.PHONY: env
env:
	@echo "$(BLUE)🔍 Current environment:$(NC)"
	@echo "  Python: $$($(PYTHON) --version 2>&1)"
	@echo "  PIP: $$($(PIP) --version 2>&1)"
	@if [ -f "$(ACTIVATE)" ]; then \
		echo "  Venv: $(GREEN)active$(NC) ($(VENV))"; \
	else \
		echo "  Venv: $(RED)not found$(NC)"; \
	fi
	@echo "  API_PROVIDER: $(API_PROVIDER)"
	@echo "  OLLAMA_MODEL: $(OLLAMA_MODEL)"
	@echo "  N_SAMPLES: $(N_SAMPLES)"
	@echo "  REQUEST_DELAY: $(REQUEST_DELAY)s"