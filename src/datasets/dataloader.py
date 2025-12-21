import pandas as pd
from collections import defaultdict
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
import torch


class MovieDataset(Dataset):
    """Dataset cho BERT4Rec (CHỈ xử lý movie_id sequence)"""
    def __init__(self, sequences, max_len):
        self.sequences = sequences
        self.max_len = max_len

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        seq = self.sequences[idx][-self.max_len:]
        # pad left với 0 (padding token)
        seq = [0] * (self.max_len - len(seq)) + seq
        return torch.tensor(seq, dtype=torch.long)


def preprocess_sequences(path, min_seq_len=5):
    """
    Preprocess dữ liệu recommender:
    - Tạo user sequence
    - Mapping movie_id -> index
    - LẤY description để dùng cho content embedding
    """

    df = pd.read_csv(
        path,
        dtype={
            'user_id': str,
            'movie_id': str
        }
    )
    df['description'] = df['description'].fillna("").astype(str)
    df = df.sort_values(['user_id', 'ts'])

    # ================= USER =================
    unique_users = df['user_id'].unique()
    user2idx = {uid: idx for idx, uid in enumerate(unique_users)}
    idx2user = {idx: uid for uid, idx in user2idx.items()}

    # ================= MOVIE =================
    unique_movies = df['movie_id'].unique()
    movie2idx = {mid: idx + 1 for idx, mid in enumerate(unique_movies)}  # 0 = padding
    idx2movie = {idx: mid for mid, idx in movie2idx.items()}

    # ================= USER SEQUENCES =================
    user_sequences = defaultdict(list)
    for _, row in df.iterrows():
        uid = row['user_id']
        mid = row['movie_id']
        user_sequences[uid].append(movie2idx[mid])

    sequences = [
        seq for seq in user_sequences.values()
        if len(seq) >= min_seq_len
    ]

    # ================= FULL HISTORY (INFERENCE) =================
    user_history = dict(user_sequences)

    # ================= NEW: MOVIE DESCRIPTIONS =================
    # key: movie_idx (int)
    # value: description (str)
    movie_descriptions = {}

    if 'description' in df.columns:
        for mid, desc in df[['movie_id', 'description']].drop_duplicates().values:
            movie_descriptions[movie2idx[mid]] = desc
    else:
        print("⚠️ Warning: CSV không có cột 'description'")

    return (
        sequences,
        movie2idx,
        idx2movie,
        user2idx,
        idx2user,
        user_history,
        movie_descriptions   # 👈 thêm vào tổng
    )


def train_val_split(sequences, val_ratio=0.2, random_state=42):
    """Chia sequences thành train/val"""
    train_seq, val_seq = train_test_split(
        sequences,
        test_size=val_ratio,
        random_state=random_state
    )
    return train_seq, val_seq


def get_dataloaders(sequences, max_seq_len, batch_size, val_ratio):
    """Tạo DataLoader train/val cho BERT4Rec"""
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
