# src/textcnn_model.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import logging

logger = logging.getLogger(__name__)

class TextCNN(nn.Module):
    """
    TextCNN for binary code bug classification.
    Architecture per rubric:
      - Embedding layer (vocab_size × embed_dim)
      - 3 parallel Conv1d layers: kernel sizes [3, 4, 5]
      - Max-over-time pooling per filter
      - Concatenate → BatchNorm1d → Dropout(0.5) → Linear → sigmoid
    """
    def __init__(self, vocab_size: int, embed_dim: int = 128, num_filters: int = 128, 
                 filter_sizes: tuple = (3, 4, 5), dropout: float = 0.5):
        super().__init__()
        
        # Embedding layer: token IDs → dense vectors
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        
        # Parallel Conv1d layers for n-gram feature extraction
        self.convs = nn.ModuleList([
            nn.Conv1d(in_channels=embed_dim, out_channels=num_filters, kernel_size=k)
            for k in filter_sizes
        ])
        
        # BatchNorm + Dropout (rubric: dropout=0.5)
        total_features = num_filters * len(filter_sizes)  # 128 * 3 = 384
        self.bn = nn.BatchNorm1d(total_features)
        self.dropout = nn.Dropout(dropout)
        
        # Final classifier: binary output (logit for BCEWithLogitsLoss)
        self.fc = nn.Linear(total_features, 1)
        
        logger.info(f"✅ TextCNN initialized: vocab={vocab_size}, embed={embed_dim}, "
                   f"filters={num_filters}×{filter_sizes}, dropout={dropout}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.embedding(x).permute(0, 2, 1)  # [B, E, L]
        
        conv_outputs = []
        for conv in self.convs:
            conv_out = F.relu(conv(x))
            # ✅ REVERTED to original working max pooling
            pooled = F.max_pool1d(conv_out, kernel_size=conv_out.size(2)).squeeze(2)
            conv_outputs.append(pooled)
            
        x = torch.cat(conv_outputs, dim=1)
        x = self.bn(x)
        x = self.dropout(x)
        return self.fc(x).squeeze(1)

    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """Convenience: returns probability (sigmoid applied)"""
        logits = self.forward(x)
        return torch.sigmoid(logits)


def load_vocab_size(vocab_path: str) -> int:
    """Helper: load vocab size from vocab.json saved by Phase 2"""
    import json
    from pathlib import Path
    vocab_file = Path(vocab_path)
    if not vocab_file.exists():
        raise FileNotFoundError(f"Vocab not found: {vocab_path}")
    with open(vocab_file, 'r', encoding='utf-8') as f:
        vocab = json.load(f)
    return len(vocab)


def get_device(config_device: str = "auto") -> torch.device:
    """Auto-detect device per config.yaml"""
    if config_device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    elif config_device == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    else:
        return torch.device("cpu")


# ================= Quick Test Block (Run when executed directly) =================
if __name__ == "__main__":
    import sys
    from pathlib import Path  # ✅ Import moved BEFORE usage
    
    # Add project root to sys.path for relative imports
    sys.path.append(str(Path(__file__).resolve().parents[1]))
    
    # Mock config for testing
    VOCAB_PATH = "data/processed/vocab.json"
    BATCH_SIZE = 4
    SEQ_LEN = 512  # From your config.yaml
    
    print("🧪 Running TextCNN sanity check...")
    
    # Load vocab size
    vocab_size = load_vocab_size(VOCAB_PATH)
    print(f"📚 Vocab size: {vocab_size}")
    
    # Initialize model
    device = get_device("auto")
    print(f"🚀 Device: {device}")
    
    model = TextCNN(vocab_size=vocab_size).to(device)
    
    # Dummy input: batch of token IDs
    dummy_input = torch.randint(0, vocab_size, (BATCH_SIZE, SEQ_LEN)).to(device)
    
    # Forward pass
    with torch.no_grad():
        logits = model(dummy_input)
        probs = model.predict_proba(dummy_input)
    
    print(f"✅ Forward pass successful:")
    print(f"   Input shape: {dummy_input.shape}")
    print(f"   Logits shape: {logits.shape} (range: [{logits.min():.3f}, {logits.max():.3f}])")
    print(f"   Probs shape: {probs.shape} (range: [{probs.min():.3f}, {probs.max():.3f}])")
    
    # Verify architecture components
    print(f"\n🔍 Architecture verification:")
    print(f"   Embedding: {model.embedding}")
    print(f"   Conv layers: {[f'k={c.kernel_size[0]}' for c in model.convs]}")
    print(f"   BN + Dropout: {model.bn}, {model.dropout}")
    print(f"   FC output: {model.fc}")
    
    print("\n✅ TextCNN module ready for training.")