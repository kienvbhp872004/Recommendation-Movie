import torch
from torch.utils.data import Dataset


class MovieDataset(Dataset):
    def __init__(self, sequences, max_len):
        self.sequences = sequences
        self.max_len = max_len


    def __len__(self):
        return len(self.sequences)


    def __getitem__(self, idx):
        seq = self.sequences[idx][-self.max_len:]
        seq = [0]*(self.max_len-len(seq)) + seq
        return torch.tensor(seq, dtype=torch.long)