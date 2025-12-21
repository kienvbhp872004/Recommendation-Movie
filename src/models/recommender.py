import torch
import torch.nn as nn


class BERT4Rec(nn.Module):
    def __init__(
        self,
        vocab_size,
        embedding_dim,
        max_seq_len,
        num_layers,
        num_heads,
        hidden_dim,
        text_emb_dim,
        pretrained_text_emb
    ):
        super().__init__()

        # ===== Item embedding (collaborative) =====
        self.item_embedding = nn.Embedding(
            vocab_size,
            embedding_dim,
            padding_idx=0
        )

        # ===== Text embedding (content) =====
        self.text_embedding = nn.Embedding.from_pretrained(
            pretrained_text_emb,   # shape: [vocab_size, text_emb_dim]
            freeze=True
        )
        self.text_proj = nn.Linear(text_emb_dim, embedding_dim)

        # ===== Transformer encoder =====
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embedding_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim,
            batch_first=True
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers
        )

        # ===== Output =====
        self.fc = nn.Linear(embedding_dim, vocab_size)

    def forward(self, x):
        """
        x: [batch_size, seq_len] (movie_id sequence)
        """

        # Collaborative embedding
        item_emb = self.item_embedding(x)        # [B, T, D]

        # Content embedding
        text_emb = self.text_embedding(x)         # [B, T, text_dim]
        text_emb = self.text_proj(text_emb)       # [B, T, D]
        # Fusion
        fused_emb = item_emb + text_emb           # [B, T, D]

        # Transformer
        x_enc = self.encoder(fused_emb)

        # Prediction
        logits = self.fc(x_enc)                   # [B, T, vocab_size]
        return logits
