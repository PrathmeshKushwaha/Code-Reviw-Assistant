import os
import sys
import json
import yaml
import torch
import logging
import numpy as np
from pathlib import Path
from sklearn.metrics import classification_report

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.model import VulnerabilityGRU
from src.trainer import create_loader, train_epoch, evaluate, compute_metrics

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s"
)

def main():
    # Reproducibility
    SEED = 42
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)

    cfg_path = Path("configs/defaults.yaml")
    with open(cfg_path, "r") as f:
        cfg = yaml.safe_load(f)

    processed_dir = Path(cfg["dataset"]["processed_dir"])
    batch_size = cfg["model"]["batch_size"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load tensors
    train_in = torch.load(processed_dir / "train_inputs.pt", weights_only=True)
    train_lab = torch.load(processed_dir / "train_labels.pt", weights_only=True)
    val_in = torch.load(processed_dir / "val_inputs.pt", weights_only=True)
    val_lab = torch.load(processed_dir / "val_labels.pt", weights_only=True)
    test_in = torch.load(processed_dir / "test_inputs.pt", weights_only=True)
    test_lab = torch.load(processed_dir / "test_labels.pt", weights_only=True)

    train_loader = create_loader(train_in, train_lab, batch_size, shuffle=True)
    val_loader = create_loader(val_in, val_lab, batch_size, shuffle=False)
    test_loader = create_loader(test_in, test_lab, batch_size, shuffle=False)

    vocab_size = len(json.load(open(processed_dir / "vocab.json")))

    model = VulnerabilityGRU(
        vocab_size=vocab_size,
        embed_dim=cfg["model"]["embed_dim"],
        hidden_dim=cfg["model"]["hidden_dim"],
        dropout=cfg["model"]["dropout"]
    ).to(device)

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["model"]["lr"], weight_decay=1e-4)

    logging.info(f"🚀 Training on {device} | Vocab: {vocab_size} | Params: {sum(p.numel() for p in model.parameters()):,}")

    best_val_f1 = 0.0
    patience, patience_counter = 3, 0
    best_model_path = processed_dir / "baseline_model.pt"

    for epoch in range(1, cfg["model"]["epochs"] + 1):
        loss = train_epoch(model, train_loader, criterion, optimizer, device)
        val_preds, val_labels = evaluate(model, val_loader, device)
        val_metrics = compute_metrics(val_preds, val_labels)

        logging.info(f"Epoch {epoch}/{cfg['model']['epochs']} | Loss: {loss:.4f} | "
                     f"Val Acc: {val_metrics['accuracy']:.4f} | Val F1: {val_metrics['f1']:.4f}")

        if val_metrics["f1"] > best_val_f1:
            best_val_f1 = val_metrics["f1"]
            torch.save(model.state_dict(), best_model_path)
            patience_counter = 0
            logging.info("💾 Saved best model checkpoint.")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                logging.info(f"⏹️ Early stopping triggered at epoch {epoch}")
                break

    # Final Test Evaluation
    logging.info("📊 Evaluating on held-out Test Set...")
    model.load_state_dict(torch.load(best_model_path, weights_only=True))
    test_preds, test_labels = evaluate(model, test_loader, device)
    test_metrics = compute_metrics(test_preds, test_labels)

    logging.info("✅ Final Test Results:")
    logging.info(f"Accuracy : {test_metrics['accuracy']:.4f}")
    logging.info(f"Precision: {test_metrics['precision']:.4f}")
    logging.info(f"Recall   : {test_metrics['recall']:.4f}")
    logging.info(f"F1-Score : {test_metrics['f1']:.4f}")

    print("\n📋 Detailed Classification Report:")
    print(classification_report(test_labels, test_preds, target_names=["Safe (0)", "Vulnerable (1)"], zero_division=0))

if __name__ == "__main__":
    main()