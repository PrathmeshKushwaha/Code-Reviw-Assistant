import logging
import numpy as np
import pandas as pd
import re
from pathlib import Path
from gensim.models import Word2Vec

def load_and_tokenize_for_w2v(raw_parquet: Path, train_indices: np.ndarray) -> list[list[str]]:
    df = pd.read_parquet(raw_parquet)
    train_codes = df.iloc[train_indices]['func'].dropna().astype(str).tolist()
    tok_re = re.compile(r'\b\w+\b|[^\s\w]')
    
    tokens = []
    for code in train_codes:
        cleaned = re.sub(r'/\*.*?\*/', '', code, flags=re.DOTALL)
        cleaned = re.sub(r'//.*$', '', cleaned, flags=re.MULTILINE)
        cleaned = ' '.join(cleaned.split())
        tokens.append(tok_re.findall(cleaned))
    return tokens

def train_word2vec(tokens: list[list[str]], embed_dim: int, min_count: int = 2, epochs: int = 10):
    logging.info(f"🔨 Training Word2Vec on {len(tokens)} functions (dim={embed_dim})...")
    model = Word2Vec(
        sentences=tokens, vector_size=embed_dim, window=5, min_count=min_count,
        workers=4, epochs=epochs, sg=0  # CBOW for faster training
    )
    return model

def build_embedding_matrix(model: Word2Vec, vocab: dict, init_std: float = 0.1):
    vocab_size = len(vocab)
    emb_matrix = np.random.normal(0.0, init_std, (vocab_size, model.vector_size))
    matched = 0
    for word, idx in vocab.items():
        if word in model.wv:
            emb_matrix[idx] = model.wv[word]
            matched += 1
    logging.info(f"✅ Word2Vec matched {matched}/{vocab_size} tokens ({matched/vocab_size:.1%})")
    return emb_matrix.astype(np.float32)