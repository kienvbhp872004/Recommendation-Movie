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
    """Dataset cho SASRec (input, target) - FIXED VERSION"""

    def __init__(self, input_sequences, targets, max_len):
        """
        Args:
            input_sequences: List of input sequences (seq[:-1])
            targets: List of target items (seq[-1])
            max_len: Maximum sequence length
        """
        self.input_sequences = input_sequences
        self.targets = targets
        self.max_len = max_len

    def __len__(self):
        return len(self.input_sequences)

    def __getitem__(self, idx):
        input_seq = self.input_sequences[idx][-self.max_len:]
        target = self.targets[idx]

        # Pad input sequence
        input_seq = [0] * (self.max_len - len(input_seq)) + input_seq

        return (
            torch.tensor(input_seq, dtype=torch.long),
            torch.tensor(target, dtype=torch.long)
        )


def preprocess_sequences(path, min_seq_len=5):
    """
    Preprocess dữ liệu recommender với ĐÚNG cách chia train/val:
    - Train: sequence[:-1] (bỏ item cuối)
    - Val: predict sequence[-1] given sequence[:-1]
    - KHÔNG có data leakage
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

    # ================= FULL HISTORY (INFERENCE) =================
    user_history = dict(user_sequences)

    # ================= MOVIE DESCRIPTIONS =================
    movie_descriptions = {}
    if 'description' in df.columns:
        for mid, desc in df[['movie_id', 'description']].drop_duplicates().values:
            movie_descriptions[movie2idx[mid]] = desc
    else:
        print("⚠️ Warning: CSV không có cột 'description'")

    # ================= ✅ FIXED: TRAIN/VAL SPLIT (Leave-One-Out) =================
    train_sequences = []
    val_input_sequences = []
    val_targets = []

    for seq in user_sequences.values():
        # Cần ít nhất min_seq_len + 1 items (min_seq_len cho train, 1 cho val)
        if len(seq) < min_seq_len + 1:
            continue

        # ✅ Train: toàn bộ sequence TRỪ item cuối
        # Đây là data model học
        train_sequences.append(seq[:-1])

        # ✅ Val: predict item cuối GIVEN sequence[:-1]
        # Input: seq[:-1], Target: seq[-1]
        val_input_sequences.append(seq[:-1])
        val_targets.append(seq[-1])

    print(f"📊 Dataset Statistics:")
    print(f"   - Total users: {len(unique_users)}")
    print(f"   - Total movies: {len(unique_movies)}")
    print(f"   - Train sequences: {len(train_sequences)}")
    print(f"   - Val sequences: {len(val_input_sequences)}")
    print(f"   - Avg train seq length: {sum(len(s) for s in train_sequences) / len(train_sequences):.1f}")

    return (
        train_sequences,
        val_input_sequences,
        val_targets,
        movie2idx,
        idx2movie,
        user2idx,
        idx2user,
        user_history,
        movie_descriptions
    )


def get_dataloaders(
        train_sequences,
        val_input_sequences,
        val_targets,
        max_seq_len,
        batch_size,
        model_type="bert"
):
    """
    ✅ FIXED: Tạo dataloader cho cả BERT4Rec và SASRec

    Args:
        train_sequences: Training sequences (seq[:-1])
        val_input_sequences: Validation input sequences (seq[:-1])
        val_targets: Validation target items (seq[-1])
        max_seq_len: Maximum sequence length
        batch_size: Batch size
        model_type: "bert" or "sasrec"
    """

    if model_type == "bert":
        # BERT4Rec: chỉ cần sequence, tự mask random trong training
        train_dataset = MovieDataset(train_sequences, max_seq_len)
        val_dataset = MovieDataset(val_input_sequences, max_seq_len)

    elif model_type == "sasrec":
        # SASRec: cần (input, target) pairs
        # Training: tạo pairs từ train_sequences
        train_inputs = []
        train_targets = []
        for seq in train_sequences:
            if len(seq) >= 2:  # Cần ít nhất 2 items để tạo (input, target)
                train_inputs.append(seq[:-1])
                train_targets.append(seq[-1])

        train_dataset = SASRecDataset(train_inputs, train_targets, max_seq_len)
        val_dataset = SASRecDataset(val_input_sequences, val_targets, max_seq_len)

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


