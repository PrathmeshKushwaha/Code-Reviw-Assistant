import os
import sys
import json
from xml.parsers.expat import model
import yaml
import logging
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from pathlib import Path
from tqdm import tqdm
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

# Add project root to sys.path so we can import src modules
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.textcnn_model import TextCNN, get_device

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s"
)

def load_config(config_path="configs/defaults.yaml"):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

def compute_metrics(preds, labels):
    preds_bin = (preds >= 0.5).astype(int)
    return {
        "accuracy": accuracy_score(labels, preds_bin),
        "f1": f1_score(labels, preds_bin),
        "auc_roc": roc_auc_score(labels, preds)
    }

def train():
    cfg = load_config()
    model_cfg = cfg["model"]
    device = get_device(model_cfg.get("device", "auto"))
    logging.info(f"🚀 Training on: {device}")

    # 1. Load processed tensors from Phase 2
    data_dir = Path(cfg["dataset"]["processed_dir"])
    logging.info(f"📥 Loading processed tensors from {data_dir}")
    
    train_inputs = torch.load(data_dir / "train_inputs.pt", weights_only=True)
    train_labels = torch.load(data_dir / "train_labels.pt", weights_only=True)
    val_inputs = torch.load(data_dir / "val_inputs.pt", weights_only=True)
    val_labels = torch.load(data_dir / "val_labels.pt", weights_only=True)
    test_inputs = torch.load(data_dir / "test_inputs.pt", weights_only=True)
    test_labels = torch.load(data_dir / "test_labels.pt", weights_only=True)

    # 2. DataLoaders
    batch_size = model_cfg["batch_size"]
    train_loader = DataLoader(TensorDataset(train_inputs, train_labels), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(TensorDataset(val_inputs, val_labels), batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(TensorDataset(test_inputs, test_labels), batch_size=batch_size, shuffle=False)

    # 3. Model Init (dynamic vocab size from data)
    vocab_size = train_inputs.max().item() + 1
    model = TextCNN(
        vocab_size=vocab_size,
        embed_dim=model_cfg["embed_dim"],
        num_filters=model_cfg.get("hidden_dim", 128),
        dropout=0.5  # Rubric requirement overrides config 0.3
    ).to(device)

    # 4. Optimizer & Loss
    optimizer = torch.optim.Adam(model.parameters(), lr=float(model_cfg["lr"]), weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=model_cfg["epochs"])
    criterion = nn.BCEWithLogitsLoss()

    # 5. Training Loop
    best_val_loss = float('inf')
    patience = 2
    patience_counter = 0
    epochs = model_cfg["epochs"]
    history = []

    logging.info(f"📊 Vocab size: {vocab_size} | Epochs: {epochs} | Batch size: {batch_size}")

    for epoch in range(1, epochs + 1):
        # Train
        model.train()
        train_loss = 0.0
        for inputs, labels in tqdm(train_loader, desc=f"Epoch {epoch}/{epochs} [Train]", leave=False):
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            logits = model(inputs)
            loss = criterion(logits, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # ADD THIS LINE
            optimizer.step()
            train_loss += loss.item() * inputs.size(0)

        avg_train_loss = train_loss / len(train_loader.dataset)

        # Validate
        model.eval()
        val_loss = 0.0
        val_preds, val_labels_list = [], []
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                logits = model(inputs)
                val_loss += criterion(logits, labels).item() * inputs.size(0)
                val_preds.extend(torch.sigmoid(logits).cpu().numpy())
                val_labels_list.extend(labels.cpu().numpy())

        avg_val_loss = val_loss / len(val_loader.dataset)
        val_metrics = compute_metrics(np.array(val_preds), np.array(val_labels_list))

        history.append({
            "epoch": epoch,
            "train_loss": avg_train_loss,
            "val_loss": avg_val_loss,
            **val_metrics
        })

        logging.info(f"Epoch {epoch:2d}/{epochs} | "
                    f"Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | "
                    f"Acc: {val_metrics['accuracy']:.3f} | F1: {val_metrics['f1']:.3f} | AUC: {val_metrics['auc_roc']:.3f}")
        scheduler.step()

        # Early Stopping
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            torch.save(model.state_dict(), "data/processed/best_textcnn.pt")
            logging.info("💾 Saved best model checkpoint.")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                logging.info(f"⏹️ Early stopping triggered after {epoch} epochs.")
                break

    # 6. Final Test Evaluation
    logging.info("\n🔍 Evaluating on TEST set...")
    model.load_state_dict(torch.load("data/processed/best_textcnn.pt", weights_only=True))
    model.eval()
    test_preds, test_labels_list = [], []
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(device)
            logits = model(inputs)
            test_preds.extend(torch.sigmoid(logits).cpu().numpy())
            test_labels_list.extend(labels.cpu().numpy())

    test_metrics = compute_metrics(np.array(test_preds), np.array(test_labels_list))
    logging.info(f"✅ TEST RESULTS | Acc: {test_metrics['accuracy']:.3f} | F1: {test_metrics['f1']:.3f} | AUC-ROC: {test_metrics['auc_roc']:.3f}")

    # Save metrics
    results_path = Path("data/processed/metrics.json")
    with open(results_path, "w") as f:
        json.dump({"history": history, "test_results": test_metrics}, f, indent=2)
    logging.info(f"📊 Metrics saved to {results_path}")

if __name__ == "__main__":
    try:
        train()
    except Exception as e:
        logging.error(f"❌ Training failed: {e}", exc_info=True)
        sys.exit(1)