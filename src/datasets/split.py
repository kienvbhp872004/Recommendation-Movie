from sklearn.model_selection import train_test_split


def train_val_split(sequences, val_ratio=0.1):
    train_seq, val_seq = train_test_split(sequences, test_size=val_ratio, random_state=42)
    return train_seq, val_seq