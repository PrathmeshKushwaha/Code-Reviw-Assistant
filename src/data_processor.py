import re
import logging
import numpy as np
import pandas as pd
import torch
from pathlib import Path
from sklearn.model_selection import train_test_split
from tqdm import tqdm

class CodePreprocessor:
    def __init__(self, config: dict):
        self.max_len = config['model']['max_len']
        self.vocab = {'<PAD>': 0, '<UNK>': 1}
        # Basic tokenizer: captures words/identifiers + individual punctuation/symbols
        self.tokenizer_regex = re.compile(r'\b\w+\b|[^\s\w]')

    def clean_code(self, code: str) -> str:
        if pd.isna(code) or not isinstance(code, str):
            return ""
        # Strip C-style comments (block first, then line)
        code = re.sub(r'/\*.*?\*/', '', code, flags=re.DOTALL)
        code = re.sub(r'//.*$', '', code, flags=re.MULTILINE)
        # Normalize all whitespace/newlines to single space
        return ' '.join(code.split()).strip()

    def tokenize(self, code: str) -> list[str]:
        return self.tokenizer_regex.findall(code)

    def build_vocab(self, train_texts: list[str]) -> dict:
        self.vocab = {'<PAD>': 0, '<UNK>': 1}
        idx = 2
        unique_tokens = set()
        for text in train_texts:
            unique_tokens.update(self.tokenize(text))
        # Sort for deterministic mapping across runs
        for token in sorted(list(unique_tokens)):
            self.vocab[token] = idx
            idx += 1
        return self.vocab

    def encode(self, tokens: list[str]) -> torch.Tensor:
        ids = [self.vocab.get(t, self.vocab['<UNK>']) for t in tokens]
        # Truncate
        ids = ids[:self.max_len]
        # Pad
        if len(ids) < self.max_len:
            ids.extend([self.vocab['<PAD>']] * (self.max_len - len(ids)))
        return torch.tensor(ids, dtype=torch.long)

    def preprocess_and_split(self, df: pd.DataFrame, config: dict) -> dict:
        required_cols = {'func', 'target'}
        if not required_cols.issubset(df.columns):
            raise KeyError(f"❌ Missing columns. Expected: {required_cols}, Found: {set(df.columns)}")

        logging.info("🧹 Cleaning code & preparing splits...")
        df = df.copy()
        df['clean_code'] = df['func'].apply(self.clean_code)

        # 70% Train, 15% Val, 15% Test (stratified)
        val_test_ratio = config['dataset']['val_split'] + config['dataset']['test_split']
        train_idx, temp_idx = train_test_split(
            np.arange(len(df)),
            test_size=val_test_ratio,
            stratify=df['target'],
            random_state=config['dataset']['random_state']
        )

        val_ratio = config['dataset']['val_split'] / val_test_ratio
        val_idx, test_idx = train_test_split(
            temp_idx,
            test_size=1.0 - val_ratio,
            stratify=df.iloc[temp_idx]['target'],
            random_state=config['dataset']['random_state']
        )

        # Build vocabulary on TRAIN ONLY (prevents data leakage)
        train_texts = df.iloc[train_idx]['clean_code'].tolist()
        logging.info("📚 Building vocabulary from training data only...")
        self.build_vocab(train_texts)
        logging.info(f"🔤 Vocabulary size: {len(self.vocab)}")

        # Encode all splits
        def encode_split(indices: np.ndarray) -> torch.Tensor:
            codes = df.iloc[indices]['clean_code'].tolist()
            token_lists = [self.tokenize(c) for c in codes]
            return torch.stack([self.encode(tokens) for tokens in tqdm(token_lists, desc="Encoding", unit="samples")])

        logging.info("🔄 Encoding splits...")
        inputs = {
            'train': encode_split(train_idx),
            'val': encode_split(val_idx),
            'test': encode_split(test_idx)
        }
        labels = {
            'train': torch.tensor(df.iloc[train_idx]['target'].values, dtype=torch.float32),
            'val': torch.tensor(df.iloc[val_idx]['target'].values, dtype=torch.float32),
            'test': torch.tensor(df.iloc[test_idx]['target'].values, dtype=torch.float32)
        }

        return {'inputs': inputs, 'labels': labels}