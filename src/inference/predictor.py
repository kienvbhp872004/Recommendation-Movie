# src/inference/predictor.py

import torch
import pandas as pd
from tqdm import tqdm


class Predictor:
    def __init__(self, model, movie2idx, idx2movie, max_seq_len, device):
        self.model = model
        self.movie2idx = movie2idx
        self.idx2movie = idx2movie
        self.max_seq_len = max_seq_len
        self.device = device

    def _prepare_sequence(self, seq):
        """pad/truncate sequence để feed cho model"""
        seq = seq[-self.max_seq_len:]
        seq = [0] * (self.max_seq_len - len(seq)) + seq
        return torch.tensor(seq, dtype=torch.long).unsqueeze(0).to(self.device)

    # src/inference/predictor.py
    def predict_topk(self, history, k=5):
        """
        Lấy top-k movie id dự đoán từ history sequence.
        """
        self.model.eval()
        seq = history[-self.max_seq_len:]
        seq = [0] * (self.max_seq_len - len(seq)) + seq
        seq_tensor = torch.tensor(seq, dtype=torch.long).unsqueeze(0).to(self.device)

        with torch.no_grad():
            logits = self.model(seq_tensor)  # (1, seq_len, vocab_size)
            last_logits = logits[0, -1]  # lấy token cuối
            probs = torch.softmax(last_logits, dim=-1)

            # loại bỏ padding idx = 0
            probs[0] = 0.0

            topk = torch.topk(probs, k=k).indices.cpu().tolist()
            movies = [self.idx2movie[i] for i in topk]
            return movies

    def predict_submission(self, submission_path, output_path, user_full_sequences):
        """
        submission_path: file .csv chứa user_id
        user_full_sequences: dict {user_id_str: [movie_ids_idx]}
        """
        df = pd.read_csv(submission_path, dtype={"user_id": str})

        results = []

        for uid in tqdm(df["user_id"], desc="Predicting"):
            if uid not in user_full_sequences:
                # user chưa có history → trả top phổ biến nhất (tùy bạn)
                results.append("0 0 0 0 0")
                continue

            history = user_full_sequences[uid]
            top5 = self.predict_topk(history, k=5)
            results.append(" ".join(top5))

        df["movie_id"] = results
        df.to_csv(output_path, index=False)
        print(f"Saved submission file to: {output_path}")
