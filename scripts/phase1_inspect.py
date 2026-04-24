import os
import json
import logging
from pathlib import Path
import pandas as pd
import yaml

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s"
)

def load_config(config_path: str = "configs/defaults.yaml") -> dict:
    BASE_DIR = Path(__file__).resolve().parents[1]  # project root
    config_path = BASE_DIR / config_path

    if not config_path.exists():
        raise FileNotFoundError(f"Config not found at: {config_path}")

    with open(config_path, "r") as f:
        return yaml.safe_load(f)
        
def load_devign_json(file_path: str) -> pd.DataFrame:
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(
            f"❌ Dataset not found at {file_path}.\n"
            "Please download the Devign dataset (JSON format) and place it there.\n"
            "Example repo: https://github.com/saikatroy27/Devign/tree/master/dataset"
        )
    
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    # Handle nested structures common in research datasets
    if isinstance(data, dict):
        data = data.get("data", data.get("functions", data))
    if not isinstance(data, list):
        raise ValueError("❌ Invalid format. Expected a JSON list of records or a dict wrapping a list.")
        
    return pd.DataFrame(data)

def inspect_and_save(df: pd.DataFrame, cfg: dict) -> None:
    raw_dir = Path(cfg["dataset"]["raw_path"]).parent
    raw_dir.mkdir(parents=True, exist_ok=True)
    
    # Basic EDA
    logging.info("✅ Dataset loaded successfully.")
    logging.info(f"📊 Shape: {df.shape[0]} samples × {df.shape[1]} columns")
    logging.info(f"🔑 Columns: {list(df.columns)}")
    logging.info(f"🏷️ Label distribution:\n{df['target'].value_counts().to_frame()}")
    logging.info(f"📝 Sample row:\n{df.iloc[0].to_dict()}")
    
    # Save as Parquet (faster I/O for later phases)
    parquet_path = raw_dir / "devign_raw.parquet"
    df.to_parquet(parquet_path, index=False)
    logging.info(f"💾 Saved raw parquet to {parquet_path}")

if __name__ == "__main__":
    cfg = load_config()
    try:
        df = load_devign_json(cfg["dataset"]["raw_path"])
        inspect_and_save(df, cfg)
    except Exception as e:
        logging.error(e)
        exit(1)