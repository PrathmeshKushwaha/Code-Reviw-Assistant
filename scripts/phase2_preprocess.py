import logging
import json
import yaml
import torch
import pandas as pd
from pathlib import Path
from src.data_processor import CodePreprocessor

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s"
)

def run_phase2():
    config_path = Path("configs/defaults.yaml")
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

    processed_dir = Path(cfg["dataset"]["processed_dir"])
    processed_dir.mkdir(parents=True, exist_ok=True)

    # Load artifact from Phase 1
    raw_parquet = Path(cfg["dataset"]["raw_path"]).parent / "devign_raw.parquet"
    if not raw_parquet.exists():
        raise FileNotFoundError("❌ devign_raw.parquet missing. Please run Phase 1 first.")

    logging.info(f"📥 Loading data from {raw_parquet}")
    df = pd.read_parquet(raw_parquet)
    logging.info(f"📊 Loaded {len(df)} samples")

    preprocessor = CodePreprocessor(cfg)
    result = preprocessor.preprocess_and_split(df, cfg)

    # Save tensors & vocab
    logging.info("💾 Saving processed artifacts...")
    for split in ['train', 'val', 'test']:
        torch.save(result['inputs'][split], processed_dir / f"{split}_inputs.pt")
        torch.save(result['labels'][split], processed_dir / f"{split}_labels.pt")

    with open(processed_dir / "vocab.json", "w", encoding="utf-8") as f:
        json.dump(preprocessor.vocab, f)

    logging.info(f"✅ Phase 2 complete.")
    logging.info(f"📦 Saved to: {processed_dir}")
    logging.info(f"📊 Sizes: Train={len(result['inputs']['train'])} | Val={len(result['inputs']['val'])} | Test={len(result['inputs']['test'])}")

if __name__ == "__main__":
    run_phase2()