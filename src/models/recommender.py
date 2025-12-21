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
            pretrained_text_emb,  # shape: [vocab_size, text_emb_dim]
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
        item_emb = self.item_embedding(x)  # [B, T, D]

        # Content embedding
        text_emb = self.text_embedding(x)  # [B, T, text_dim]
        text_emb = self.text_proj(text_emb)  # [B, T, D]

        # Fusion
        fused_emb = item_emb + text_emb  # [B, T, D]

        # Transformer
        x_enc = self.encoder(fused_emb)

        # Prediction
        logits = self.fc(x_enc)  # [B, T, vocab_size]
        return logits


class SASRec(nn.Module):
    def __init__(
            self,
            vocab_size,
            embedding_dim,
            max_seq_len,
            num_heads,
            num_layers,
            dropout,
            text_emb_dim=None,
            pretrained_text_emb=None
    ):
        super().__init__()

        self.max_seq_len = max_seq_len

        self.item_emb = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.pos_emb = nn.Embedding(max_seq_len, embedding_dim)

        # ✅ Text embedding (content-based)
        if pretrained_text_emb is not None:
            self.text_emb = nn.Embedding.from_pretrained(
                pretrained_text_emb, freeze=True
            )
            self.text_proj = nn.Linear(text_emb_dim, embedding_dim)
        else:
            self.text_emb = None

        # ✅ Transformer với causal mask
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embedding_dim,
            nhead=num_heads,
            dim_feedforward=embedding_dim * 4,
            dropout=dropout,
            batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers)

        self.fc = nn.Linear(embedding_dim, vocab_size)

    def forward(self, x):
        """
        x: [batch_size, seq_len]
        return: [batch_size, vocab_size] (chỉ dự đoán next item)
        """
        batch_size, seq_len = x.size()

        # ✅ Causal mask: không cho model nhìn tương lai
        mask = torch.triu(
            torch.ones(seq_len, seq_len, device=x.device) * float('-inf'),
            diagonal=1
        )

        # Positional encoding
        pos = torch.arange(seq_len, device=x.device).unsqueeze(0)

        # Item + Position embedding
        emb = self.item_emb(x) + self.pos_emb(pos)

        # ✅ Text embedding (nếu có)
        if self.text_emb is not None:
            text = self.text_proj(self.text_emb(x))
            emb = emb + text

        # ✅ Transformer với causal mask
        h = self.encoder(emb, mask=mask)

        # ✅ CHỈ lấy hidden state cuối cùng để dự đoán next item
        return self.fc(h[:, -1])  # [B, vocab_size]