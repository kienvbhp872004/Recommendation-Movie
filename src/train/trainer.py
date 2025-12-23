import torch
import numpy as np
from tqdm import tqdm


class Trainer:
    def __init__(self, model, optimizer, criterion, model_type="bert", callbacks=[], mask_prob=0.15):
        """
        Args:
            model_type: "bert" | "sasrec"
            mask_prob: Probability of masking items for BERT4Rec (default 0.15)
        """
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.model_type = model_type
        self.callbacks = callbacks
        self.mask_prob = mask_prob
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

    def mask_sequence(self, seq, mask_prob=0.15, mask_token=0):
        """
        ✅ BERT4Rec masking strategy:
        - Randomly mask `mask_prob` of items
        - 80% replace with [MASK] (0)
        - 10% replace with random item
        - 10% keep original

        Returns:
            masked_seq: Input sequence with masks
            labels: Original sequence (targets)
            mask_positions: Boolean mask of positions to predict
        """
        device = seq.device  # ✅ Get device from input tensor
        seq = seq.clone()
        labels = seq.clone()

        # Create mask: 1 for items to predict, 0 for others
        mask_positions = torch.rand(seq.shape, device=device) < mask_prob  # ✅ Same device
        # Don't mask padding (0)
        mask_positions = mask_positions & (seq != 0)

        # 80% -> [MASK] token
        mask_80 = torch.rand(seq.shape, device=device) < 0.8  # ✅ Same device
        seq[mask_positions & mask_80] = mask_token

        # 10% -> random item
        mask_10 = torch.rand(seq.shape, device=device) < 0.5  # ✅ Same device
        vocab_size = self.model.vocab_size if hasattr(self.model, 'vocab_size') else seq.max().item() + 1
        random_items = torch.randint(1, vocab_size, seq.shape, device=device)  # ✅ Same device
        seq[mask_positions & ~mask_80 & mask_10] = random_items[mask_positions & ~mask_80 & mask_10]

        # 10% -> keep original (do nothing)

        return seq, labels, mask_positions

    def train_one_epoch(self, loader, device):
        self.model.train()
        total_loss = 0

        for batch in tqdm(loader, desc="Training"):
            self.optimizer.zero_grad()

            # ================= ✅ BERT4Rec - FIXED =================
            if self.model_type == "bert":
                seq = batch.to(device)  # [B, T]

                # ✅ Mask random items
                masked_seq, labels, mask_positions = self.mask_sequence(seq, self.mask_prob)

                # Forward pass
                logits = self.model(masked_seq)  # [B, T, vocab_size]

                # ✅ Chỉ tính loss cho masked positions
                loss = self.criterion(
                    logits[mask_positions],  # [num_masked, vocab_size]
                    labels[mask_positions]  # [num_masked]
                )

            # ================= ✅ SASRec - CORRECT =================
            elif self.model_type == "sasrec":
                seq, target = batch
                seq = seq.to(device)
                target = target.to(device)

                logits = self.model(seq)  # [B, vocab_size]

                # ✅ Predict next item
                loss = self.criterion(logits, target)

            else:
                raise ValueError(f"Unknown model_type: {self.model_type}")

            loss.backward()
            self.optimizer.step()
            total_loss += loss.item()

        return total_loss / len(loader)

    def compute_metrics(self, loader, device, k_list=[5, 10]):
        """
        ✅ FIXED: Tính Precision@K, Recall@K, NDCG@K
        """
        self.model.eval()

        all_precisions = {k: [] for k in k_list}
        all_recalls = {k: [] for k in k_list}
        all_ndcgs = {k: [] for k in k_list}

        with torch.no_grad():
            for batch in tqdm(loader, desc="Computing metrics"):

                # ================= ✅ BERT4Rec =================
                if self.model_type == "bert":
                    seq = batch.to(device)  # [B, T]

                    # ✅ Predict next item: mask vị trí cuối + 1
                    # Tạo input sequence bằng cách append [MASK] token
                    mask_token = 0
                    masked_seq = torch.cat([
                        seq,
                        torch.full((seq.size(0), 1), mask_token, device=device, dtype=torch.long)
                    ], dim=1)

                    logits = self.model(masked_seq)  # [B, T+1, vocab_size]
                    logits = logits[:, -1, :]  # [B, vocab_size] - prediction for masked position

                    # Target = last item in original sequence
                    targets = seq[:, -1]  # [B]

                # ================= ✅ SASRec =================
                elif self.model_type == "sasrec":
                    seq, targets = batch
                    seq = seq.to(device)
                    targets = targets.to(device)

                    logits = self.model(seq)  # [B, vocab_size]

                # ================= Compute metrics =================
                for k in k_list:
                    _, topk_indices = torch.topk(logits, k, dim=-1)  # [B, K]

                    for i in range(logits.size(0)):
                        target = targets[i].item()
                        preds = topk_indices[i].cpu().tolist()

                        # Skip padding
                        if target == 0:
                            continue

                        # Hit or not
                        hit = 1 if target in preds else 0

                        # Precision@K
                        all_precisions[k].append(hit / k)

                        # Recall@K (binary: 1 target)
                        all_recalls[k].append(hit)

                        # NDCG@K
                        if hit:
                            rank = preds.index(target) + 1
                            ndcg = 1.0 / np.log2(rank + 1)
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
        """
        ✅ FIXED: Validation với proper masking cho BERT4Rec
        """
        self.model.eval()
        total_loss = 0

        with torch.no_grad():
            for batch in tqdm(loader, desc="Validation"):

                # ================= ✅ BERT4Rec =================
                if self.model_type == "bert":
                    seq = batch.to(device)

                    # ✅ Mask last position để predict
                    masked_seq = seq.clone()
                    masked_seq[:, -1] = 0  # Mask last item
                    targets = seq[:, -1]  # Target = last item

                    logits = self.model(masked_seq)  # [B, T, vocab_size]
                    logits = logits[:, -1, :]  # [B, vocab_size]

                    loss = self.criterion(logits, targets)

                # ================= ✅ SASRec =================
                elif self.model_type == "sasrec":
                    seq, target = batch
                    seq = seq.to(device)
                    target = target.to(device)

                    logits = self.model(seq)  # [B, vocab_size]
                    loss = self.criterion(logits, target)

                total_loss += loss.item()

        return total_loss / len(loader)

    def fit(self, train_loader, val_loader, epochs, device):
        """
        ✅ FIXED: Complete training loop với callbacks
        """
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

            # ✅ Callbacks (early stopping, checkpointing, etc.)
            # for callback in self.callbacks:
            #     callback.on_epoch_end(epoch, val_loss, self.model)
            #
            #     # Check if early stopping triggered
            #     if hasattr(callback, 'early_stop') and callback.early_stop:
            #         print(f"\n⚠️ Early stopping triggered at epoch {epoch + 1}")
            #         return

        print("\n✅ Training completed!")