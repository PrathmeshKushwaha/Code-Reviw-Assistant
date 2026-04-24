import torch
import torch.nn as nn
import numpy as np

class VulnerabilityGRU(nn.Module):
    def __init__(self, vocab_size: int, embed_dim: int, hidden_dim: int, num_classes: int = 2, 
                 dropout: float = 0.3, embedding_weights: np.ndarray = None):
        super().__init__()
        
        # Use pretrained weights if provided, else initialize randomly
        if embedding_weights is not None:
            assert embedding_weights.shape[1] == embed_dim, "Embedding dimension mismatch"
            self.embedding = nn.Embedding.from_pretrained(
                torch.FloatTensor(embedding_weights), 
                freeze=False,  # Allow fine-tuning during training
                padding_idx=0
            )
        else:
            self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
            
        self.gru = nn.GRU(
            input_size=embed_dim,
            hidden_size=hidden_dim,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
            dropout=dropout
        )
        
        effective_hidden = hidden_dim * 2
        self.classifier = nn.Sequential(
            nn.LayerNorm(effective_hidden),
            nn.Linear(effective_hidden, effective_hidden // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(effective_hidden // 2, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        embedded = self.embedding(x)
        gru_out, _ = self.gru(embedded)
        pooled, _ = torch.max(gru_out, dim=1)
        return self.classifier(pooled)