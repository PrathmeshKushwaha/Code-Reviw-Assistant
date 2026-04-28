# scripts/phase4_visualize_activations.py
import os
import sys
import yaml
import logging
import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.textcnn_model import TextCNN, get_device
from src.data_processor import CodePreprocessor

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)-8s | %(message)s")

class ActivationHook:
    """Captures forward activations from Conv layers"""
    def __init__(self):
        self.activations = []
    def __call__(self, module, input, output):
        # output shape: [batch, num_filters, seq_len']
        self.activations.append(output.detach().cpu())

def run_visualization():
    cfg_path = "configs/defaults.yaml"
    with open(cfg_path, "r") as f:
        cfg = yaml.safe_load(f)

    device = get_device(cfg["model"].get("device", "auto"))
    logging.info(f"🚀 Visualization on: {device}")

    # 1. Load trained model
    data_dir = Path(cfg["dataset"]["processed_dir"])
    vocab_path = data_dir / "vocab.json"
    model_path = data_dir / "best_textcnn.pt"

    # Load vocab size
    import json
    with open(vocab_path, 'r') as f:
        vocab = json.load(f)
    vocab_size = len(vocab)

    model = TextCNN(
        vocab_size=vocab_size,
        embed_dim=cfg["model"]["embed_dim"],
        num_filters=cfg["model"].get("hidden_dim", 128),
        dropout=0.5
    ).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model.eval()

    # 2. Register hooks on Conv layers
    hooks = [ActivationHook() for _ in model.convs]
    hook_handles = [conv.register_forward_hook(hook) for conv, hook in zip(model.convs, hooks)]

    # 3. Load raw parquet + test tensors
    raw_parquet = Path(cfg["dataset"]["raw_path"]).parent / "devign_raw.parquet"
    df_raw = pd.read_parquet(raw_parquet)
    test_inputs = torch.load(data_dir / "test_inputs.pt", weights_only=True)
    test_labels = torch.load(data_dir / "test_labels.pt", weights_only=True)

    # Preprocessor to align tokens
    prep = CodePreprocessor(cfg)

    # 4. Select 3 diverse test samples (1 buggy, 2 clean or mixed)
    sample_indices = [10, 450, 980]  # Adjust if out of bounds
    sample_indices = [i for i in sample_indices if i < len(test_inputs)]
    logging.info(f"🔍 Visualizing {len(sample_indices)} samples...")

    out_dir = Path("outputs/visualizations")
    out_dir.mkdir(parents=True, exist_ok=True)

    for idx in sample_indices:
        # Get raw code & label
        code_str = df_raw.iloc[idx]["func"]
        true_label = test_labels[idx].item()

        # Re-tokenize to get human-readable tokens
        clean_code = prep.clean_code(code_str)
        tokens = prep.tokenize(clean_code)[:cfg["model"]["max_len"]]

        # Forward pass with hooks
        with torch.no_grad():
            model.eval()
            input_tensor = test_inputs[idx:idx+1].to(device)
            _ = model(input_tensor)  # Triggers hooks

        # Extract activations per conv layer
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))
        fig.suptitle(f"Sample #{idx} | True: {'BUGGY' if true_label else 'CLEAN'} | Tokens: {len(tokens)}", fontsize=12)

        for layer_idx, hook in enumerate(hooks):
            # activation shape: [1, num_filters, seq_len_after_conv]
            act = hook.activations[0].squeeze(0)  # [F, L']
            # Collapse filters into mean activation per position for visualization
            mean_act = act.mean(dim=0).numpy()  # [L']

            # Align with original tokens (conv shortens sequence)
            k_size = [3,4,5][layer_idx]
            valid_len = min(len(mean_act), len(tokens))
            tokens_aligned = tokens[:valid_len]

            # Plot heatmap
            sns.heatmap(
                mean_act[:valid_len].reshape(1, -1),
                xticklabels=tokens_aligned,
                yticklabels=[f"Filter k={k_size}"],
                ax=axes[layer_idx],
                cmap="viridis",
                cbar=True
            )
            axes[layer_idx].set_title(f"Activation (Mean across {act.shape[0]} filters)")
            axes[layer_idx].set_xticklabels(axes[layer_idx].get_xticklabels(), rotation=45, ha="right", fontsize=8)

        plt.tight_layout()
        save_path = out_dir / f"activation_sample_{idx}.png"
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        logging.info(f"💾 Saved: {save_path}")
        plt.close()

        # Clear hook cache for next iteration
        for h in hooks: h.activations.clear()

    # Cleanup handles
    for h in hook_handles: h.remove()

    logging.info("✅ Step 7 complete. Check outputs/visualizations/ for heatmaps.")

if __name__ == "__main__":
    try:
        run_visualization()
    except Exception as e:
        logging.error(f"❌ Visualization failed: {e}", exc_info=True)
        sys.exit(1)