import torch

class EarlyStopping:
    def __init__(self, patience, save_path=None):
        self.patience = patience
        self.wait = 0
        self.best_loss = float('inf')
        self.save_path = save_path

    def on_epoch_end(self, model, val_loss):
        if val_loss < self.best_loss:
            self.best_loss = val_loss
            self.wait = 0
            if self.save_path:
                torch.save(model.state_dict(), self.save_path)
        else:
            self.wait += 1
            if self.wait >= self.patience:
                print('Early stopping triggered')
                return True
        return False
