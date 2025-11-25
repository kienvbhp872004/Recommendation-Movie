import torch
import torch.nn as nn


class BERT4Rec(nn.Module):
    def __init__(self, vocab_size, embedding_dim, max_seq_len, num_layers, num_heads, hidden_dim):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        encoder_layer = nn.TransformerEncoderLayer(
        d_model=embedding_dim,
        nhead=num_heads,
        dim_feedforward=hidden_dim,
        batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc = nn.Linear(embedding_dim, vocab_size)


    def forward(self, x):
        x_emb = self.embedding(x)
        x_enc = self.encoder(x_emb)
        logits = self.fc(x_enc)
        return logits