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


class SASRecDataset(Dataset):
    """Dataset cho SASRec (input, target)"""

    def __init__(self, sequences, max_len):
        self.sequences = sequences
        self.max_len = max_len

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        seq = self.sequences[idx]

        input_seq = seq[:-1]
        target = seq[-1]

        input_seq = input_seq[-self.max_len:]
        input_seq = [0] * (self.max_len - len(input_seq)) + input_seq

        return (
            torch.tensor(input_seq, dtype=torch.long),
            torch.tensor(target, dtype=torch.long)
        )


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
    movie_descriptions = {}

    if 'description' in df.columns:
        for mid, desc in df[['movie_id', 'description']].drop_duplicates().values:
            movie_descriptions[movie2idx[mid]] = desc
    else:
        print("⚠️ Warning: CSV không có cột 'description'")

    # ================= FIX: TRAIN/VAL SPLIT =================
    train_sequences = []
    val_sequences = []

    for seq in user_sequences.values():
        if len(seq) < min_seq_len:
            continue

        # ✅ Train: bỏ 2 item cuối, Val: bỏ 1 item cuối
        # Tránh data leakage
        if len(seq) >= min_seq_len + 1:
            train_sequences.append(seq[:-2])  # Bỏ 2 item cuối
            val_sequences.append(seq[:-1])  # Bỏ 1 item cuối
        else:
            train_sequences.append(seq[:-1])
            val_sequences.append(seq[:-1])

    return (
        train_sequences,
        val_sequences,
        movie2idx,
        idx2movie,
        user2idx,
        idx2user,
        user_history,
        movie_descriptions
    )


def get_dataloaders(
        train_sequences,
        val_sequences,
        max_seq_len,
        batch_size,
        model_type="bert"
):
    if model_type == "bert":
        train_dataset = MovieDataset(train_sequences, max_seq_len)
        val_dataset = MovieDataset(val_sequences, max_seq_len)
    elif model_type == "sasrec":
        train_dataset = SASRecDataset(train_sequences, max_seq_len)
        val_dataset = SASRecDataset(val_sequences, max_seq_len)
    else:
        raise ValueError("model_type must be 'bert' or 'sasrec'")

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True
    )

    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False
    )

    return train_loader, val_loader