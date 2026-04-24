import os, sys, json, yaml, torch, logging, numpy as np, pandas as pd
from pathlib import Path
from itertools import product
from sklearn.metrics import classification_report

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.model import VulnerabilityGRU
from src.trainer import create_loader, train_epoch, evaluate, compute_metrics
from src.embedding_utils import load_and_tokenize_for_w2v, train_word2vec, build_embedding_matrix

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)-8s | %(message)s")

def run_tuning():
    cfg_path = Path("configs/defaults.yaml")
    with open(cfg_path, "r") as f:
        cfg = yaml.safe_load(f)

    processed_dir = Path(cfg["dataset"]["processed_dir"])
    raw_parquet = Path(cfg["dataset"]["raw_path"]).parent / "devign_raw.parquet"
    vocab = json.load(open(processed_dir / "vocab.json"))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load pre-split tensors to map back to indices for W2V training
    train_indices = np.arange(0, int(len(pd.read_parquet(raw_parquet)) * (1 - cfg['dataset']['val_split'] - cfg['dataset']['test_split'])))
    train_in = torch.load(processed_dir / "train_inputs.pt", weights_only=True)
    train_lab = torch.load(processed_dir / "train_labels.pt", weights_only=True)
    val_in = torch.load(processed_dir / "val_inputs.pt", weights_only=True)
    val_lab = torch.load(processed_dir / "val_labels.pt", weights_only=True)

    # 1. Train Word2Vec on training corpus
    train_tokens = load_and_tokenize_for_w2v(raw_parquet, train_indices)
    w2v_model = train_word2vec(train_tokens, embed_dim=cfg['model']['embed_dim'])
    emb_matrix = build_embedding_matrix(w2v_model, vocab)

    # 2. Hyperparameter Grid
    param_grid = {
        'hidden_dim': [64, 128],
        'lr': [1e-3, 5e-4],
        'dropout': [0.2, 0.3]
    }
    keys, values = zip(*param_grid.items())
    grid = [dict(zip(keys, v)) for v in product(*values)]

    best_f1 = 0.0
    best_cfg = None
    results = []

    logging.info(f"🔍 Starting grid search over {len(grid)} configurations...")

    for params in grid:
        run_cfg = {**cfg['model'], **params}
        torch.manual_seed(42)
        
        model = VulnerabilityGRU(
            vocab_size=len(vocab), embed_dim=cfg['model']['embed_dim'],
            hidden_dim=run_cfg['hidden_dim'], dropout=run_cfg['dropout'],
            embedding_weights=emb_matrix
        ).to(device)

        optimizer = torch.optim.AdamW(model.parameters(), lr=run_cfg['lr'])
        criterion = torch.nn.CrossEntropyLoss()

        train_loader = create_loader(train_in, train_lab, cfg['model']['batch_size'])
        val_loader = create_loader(val_in, val_lab, cfg['model']['batch_size'], shuffle=False)

        # Fast tuning: 3 epochs per config
        for ep in range(3):
            train_epoch(model, train_loader, criterion, optimizer, device)
        
        v_preds, v_labs = evaluate(model, val_loader, device)
        v_f1 = compute_metrics(v_preds, v_labs)['f1']
        results.append({**params, 'val_f1': v_f1})
        
        if v_f1 > best_f1:
            best_f1 = v_f1
            best_cfg = {**cfg, 'model': run_cfg}
            torch.save(model.state_dict(), processed_dir / "tuned_best_model.pt")
            logging.info(f"🌟 New best: {params} | Val F1: {v_f1:.4f}")

    # Print results table
    import pandas as pd
    res_df = pd.DataFrame(results).sort_values('val_f1', ascending=False)
    print("\n📊 Hyperparameter Search Results:")
    print(res_df.to_string(index=False))

    # Save best config
    with open(processed_dir / "best_config.yaml", "w") as f:
        yaml.dump(best_cfg, f)

    logging.info(f"✅ Tuning complete. Best Val F1: {best_f1:.4f}")
    logging.info(f"📁 Saved best model & config to {processed_dir}")

if __name__ == "__main__":
    run_tuning()