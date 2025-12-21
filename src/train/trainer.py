import torch
from tqdm import tqdm


class Trainer:
    def __init__(self, model, optimizer, criterion, model_type="bert", callbacks=[]):
        """
        model_type: "bert" | "sasrec"
        """
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.model_type = model_type
        self.callbacks = callbacks
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'precision@5': [],
            'recall@5': [],
            'ndcg@5': [],
            'precision@10': [],
            'recall@10': [],
            'ndcg@10': []
        }

    def train_one_epoch(self, loader, device):
        self.model.train()
        total_loss = 0

        for batch in tqdm(loader, desc="Training"):
            self.optimizer.zero_grad()

            # ================= BERT4Rec =================
            if self.model_type == "bert":
                seq = batch.to(device)
                logits = self.model(seq)

                loss = self.criterion(
                    logits.view(-1, logits.size(-1)),
                    seq.view(-1)
                )

            # ================= SASRec =================
            elif self.model_type == "sasrec":
                seq, target = batch
                seq = seq.to(device)
                target = target.to(device)

                logits = self.model(seq)  # [B, vocab_size]

                # ✅ Chỉ tính loss cho next item prediction
                loss = self.criterion(logits, target)

            loss.backward()
            self.optimizer.step()
            total_loss += loss.item()

        return total_loss / len(loader)

    def compute_metrics(self, loader, device, k_list=[5, 10]):
        """
        Tính Precision@K, Recall@K, NDCG@K
        """
        self.model.eval()

        all_precisions = {k: [] for k in k_list}
        all_recalls = {k: [] for k in k_list}
        all_ndcgs = {k: [] for k in k_list}

        with torch.no_grad():
            for batch in tqdm(loader, desc="Computing metrics"):

                if self.model_type == "bert":
                    seq = batch.to(device)
                    logits = self.model(seq)  # [B, T, vocab_size]

                    # Lấy prediction của vị trí cuối cùng
                    logits = logits[:, -1, :]  # [B, vocab_size]
                    targets = seq[:, -1]  # [B]

                elif self.model_type == "sasrec":
                    seq, targets = batch
                    seq = seq.to(device)
                    targets = targets.to(device)

                    logits = self.model(seq)  # [B, vocab_size]

                # Top-K predictions
                for k in k_list:
                    _, topk_indices = torch.topk(logits, k, dim=-1)  # [B, K]

                    for i in range(logits.size(0)):
                        target = targets[i].item()
                        preds = topk_indices[i].cpu().tolist()

                        # Skip padding
                        if target == 0:
                            continue

                        # Precision@K
                        hit = 1 if target in preds else 0
                        all_precisions[k].append(hit / k)

                        # Recall@K
                        all_recalls[k].append(hit)

                        # NDCG@K
                        if hit:
                            # Tìm vị trí của target trong predictions
                            rank = preds.index(target) + 1
                            ndcg = 1.0 / (torch.log2(torch.tensor(rank + 1.0)).item())
                        else:
                            ndcg = 0.0
                        all_ndcgs[k].append(ndcg)

        metrics = {}
        for k in k_list:
            metrics[f'precision@{k}'] = sum(all_precisions[k]) / len(all_precisions[k]) if all_precisions[k] else 0.0
            metrics[f'recall@{k}'] = sum(all_recalls[k]) / len(all_recalls[k]) if all_recalls[k] else 0.0
            metrics[f'ndcg@{k}'] = sum(all_ndcgs[k]) / len(all_ndcgs[k]) if all_ndcgs[k] else 0.0

        return metrics

    def validate(self, loader, device):
        self.model.eval()
        total_loss = 0

        with torch.no_grad():
            for batch in tqdm(loader, desc="Validation"):

                if self.model_type == "bert":
                    seq = batch.to(device)
                    logits = self.model(seq)
                    loss = self.criterion(
                        logits.view(-1, logits.size(-1)),
                        seq.view(-1)
                    )

                elif self.model_type == "sasrec":
                    seq, target = batch
                    seq = seq.to(device)
                    target = target.to(device)

                    logits = self.model(seq)  # [B, vocab_size]

                    # ✅ Chỉ tính loss cho next item (không tính toàn bộ sequence)
                    loss = self.criterion(logits, target)

                total_loss += loss.item()

        return total_loss / len(loader)

    def fit(self, train_loader, val_loader, epochs, device):
        for epoch in range(epochs):
            print(f"\n{'=' * 60}")
            print(f"Epoch {epoch + 1}/{epochs}")
            print(f"{'=' * 60}")

            # Training
            train_loss = self.train_one_epoch(train_loader, device)

            # Validation loss
            val_loss = self.validate(val_loader, device)

            # Metrics
            metrics = self.compute_metrics(val_loader, device, k_list=[5, 10])

            # Update history
            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_loss)
            for k, v in metrics.items():
                self.history[k].append(v)

            # Print results
            print(f"\n📊 Results:")
            print(f"  Train Loss: {train_loss:.4f}")
            print(f"  Val Loss:   {val_loss:.4f}")
            print(f"\n  Precision@5:  {metrics['precision@5']:.4f}")
            print(f"  Recall@5:     {metrics['recall@5']:.4f}")
            print(f"  NDCG@5:       {metrics['ndcg@5']:.4f}")
            print(f"\n  Precision@10: {metrics['precision@10']:.4f}")
            print(f"  Recall@10:    {metrics['recall@10']:.4f}")
            print(f"  NDCG@10:      {metrics['ndcg@10']:.4f}")

            # ✅ Callback (early stopping)
