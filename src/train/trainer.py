import torch
from tqdm import tqdm

class Trainer:
    def __init__(self, model, optimizer, criterion, callbacks=[]):
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.callbacks = callbacks
        self.history = {'train_loss': [], 'val_loss': []}

    def train_one_epoch(self, loader, device):
        self.model.train()
        total_loss = 0
        for seq in tqdm(loader):
            seq = seq.to(device)
            logits = self.model(seq)
            loss = self.criterion(logits.view(-1, logits.size(-1)), seq.view(-1))
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            total_loss += loss.item()
        return total_loss / len(loader)

    def validate(self, loader, device):
        self.model.eval()
        total_loss = 0
        with torch.no_grad():
            for seq in loader:
                seq = seq.to(device)
                logits = self.model(seq)
                loss = self.criterion(logits.view(-1, logits.size(-1)), seq.view(-1))
                total_loss += loss.item()
        return total_loss / len(loader)

    def fit(self, train_loader, val_loader, epochs, device):
        for epoch in range(epochs):
            train_loss = self.train_one_epoch(train_loader, device)
            val_loss = self.validate(val_loader, device)
            print(f"Epoch {epoch+1}: train_loss={train_loss:.4f}, val_loss={val_loss:.4f}")
            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_loss)
