# main.py
import os
import torch

from src.utils.config import load_config
from src.utils.gpu import get_device
from src.datasets.dataloader import preprocess_sequences, get_dataloaders
from src.models.recommender import BERT4Rec
from src.train.trainer import Trainer
from src.train.callbacks import EarlyStopping
from src.utils.visualize import plot_loss, plot_metrics
from src.inference.predictor import Predictor   # <<< NEW


def main():
    # --- Load config + device ---
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.join(BASE_DIR, 'config.yaml')
    config = load_config(config_path)
    device = get_device()
    data_path = os.path.join(BASE_DIR, config['data']['input_path'])
    print("Using device:", device)

    # --- Load sequences + user history ---
    sequences, movie2idx, idx2movie, user2idx, idx2user, user_history = preprocess_sequences(
        data_path
    )

    # --- DataLoader ---
    train_loader, val_loader = get_dataloaders(
        sequences,
        max_seq_len=config['model']['max_seq_len'],
        batch_size=config['train']['batch_size'],
        val_ratio=config['train']['val_ratio']
    )

    # --- Model ---
    model = BERT4Rec(
        vocab_size=config['model']['vocab_size'],
        embedding_dim=config['model']['embedding_dim'],
        max_seq_len=config['model']['max_seq_len'],
        num_layers=config['model']['num_layers'],
        num_heads=config['model']['num_heads'],
        hidden_dim=config['model']['hidden_dim']
    ).to(device)

    # --- Optimizer, criterion, callbacks ---
    optimizer = torch.optim.Adam(model.parameters(), lr=float(config['train']['lr']))
    criterion = torch.nn.CrossEntropyLoss()
    callbacks = [
        EarlyStopping(
            patience=config['callbacks']['early_stopping_patience'],
            save_path=config['callbacks']['checkpoint_path']
        )
    ]

    # --- Trainer ---
    trainer = Trainer(model, optimizer, criterion, callbacks)
    trainer.fit(train_loader, val_loader, config['train']['epochs'], device)

    # --- Visualization ---
    print("Plotting loss curve...")
    plot_loss(trainer.history['train_loss'], trainer.history['val_loss'],
              save_path=os.path.join(BASE_DIR, 'visualize/loss_train/loss_curve.png'))

    metrics_keys = ('precision@5', 'recall@5', 'ndcg@5')
    if all(k in trainer.history for k in metrics_keys):
        print("Plotting metrics...")
        plot_metrics(trainer.history, keys=metrics_keys,
                     save_path=os.path.join(BASE_DIR, 'metrics.png'))

    # ---------- INFERENCE / SUBMISSION ----------
    print("\n=== Running Submission Prediction ===")
    predictor = Predictor(
        model=model,
        movie2idx=movie2idx,
        idx2movie=idx2movie,
        max_seq_len=config['model']['max_seq_len'],
        device=device
    )

    submission_input = os.path.join(BASE_DIR, "data/raw/submission.csv")
    submission_output = os.path.join(BASE_DIR, "submission_result.csv")

    predictor.predict_submission(
        submission_input,
        submission_output,
        user_history
    )


if __name__ == '__main__':
    main()
