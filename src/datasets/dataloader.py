# src/datasets/preprocess_sequences.py

import pandas as pd
from collections import defaultdict
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
import torch

class MovieDataset(Dataset):
    """Dataset cho BERT4Rec"""
    def __init__(self, sequences, max_len):
        self.sequences = sequences
        self.max_len = max_len

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        seq = self.sequences[idx][-self.max_len:]
        # pad left với 0
        seq = [0] * (self.max_len - len(seq)) + seq
        return torch.tensor(seq, dtype=torch.long)


def preprocess_sequences(path, min_seq_len=4):
    df = pd.read_csv(path, dtype={'user_id': str, 'movie_id': str})
    df = df.sort_values(['user_id', 'ts'])

    unique_users = df['user_id'].unique()
    user2idx = {uid: idx for idx, uid in enumerate(unique_users)}
    idx2user = {idx: uid for uid, idx in user2idx.items()}

    unique_movies = df['movie_id'].unique()
    movie2idx = {mid: idx+1 for idx, mid in enumerate(unique_movies)}
    idx2movie = {idx: mid for mid, idx in movie2idx.items()}

    from collections import defaultdict
    user_sequences = defaultdict(list)
    for _, row in df.iterrows():
        uid = row['user_id']
        mid = row['movie_id']
        user_sequences[uid].append(movie2idx[mid])

    sequences = [seq for seq in user_sequences.values() if len(seq) >= min_seq_len]

    # NEW: full history for inference
    user_history = {uid: seq for uid, seq in user_sequences.items()}

    return sequences, movie2idx, idx2movie, user2idx, idx2user, user_history



def train_val_split(sequences, val_ratio=0.2, random_state=42):
    """
    Chia sequences thành train/val

    Args:
        sequences (List[List[int]])
        val_ratio (float)

    Returns:
        train_seq, val_seq
    """
    train_seq, val_seq = train_test_split(sequences, test_size=val_ratio, random_state=random_state)
    return train_seq, val_seq


def get_dataloaders(sequences, max_seq_len, batch_size, val_ratio=0.2):
    """
    Tạo DataLoader train/val
    """
    train_seq, val_seq = train_val_split(sequences, val_ratio)
    train_loader = torch.utils.data.DataLoader(
        MovieDataset(train_seq, max_seq_len),
        batch_size=batch_size,
        shuffle=True
    )
    val_loader = torch.utils.data.DataLoader(
        MovieDataset(val_seq, max_seq_len),
        batch_size=batch_size,
        shuffle=False
    )
    return train_loader, val_loader
