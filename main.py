# main.py
import os
import torch

from sentence_transformers import SentenceTransformer

from src.utils.config import load_config
from src.utils.gpu import get_device
from src.datasets.dataloader import preprocess_sequences, get_dataloaders
from src.models.recommender import BERT4Rec
from src.train.trainer import Trainer
from src.train.callbacks import EarlyStopping
from src.utils.visualize import plot_loss, plot_metrics
from src.inference.predictor import Predictor


def build_text_embedding(movie_descriptions, vocab_size, device):
    """
    Encode movie descriptions -> pretrained_text_emb
    """
    print("Encoding movie descriptions...")
    text_model = SentenceTransformer("all-MiniLM-L6-v2")

    text_dim = text_model.get_sentence_embedding_dimension()
    pretrained_text_emb = torch.zeros(vocab_size, text_dim)

    for movie_idx, desc in movie_descriptions.items():
        emb = text_model.encode(desc, convert_to_tensor=True)
        pretrained_text_emb[movie_idx] = emb.cpu()

    return pretrained_text_emb


def main():
    # ================= LOAD CONFIG + DEVICE =================
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.join(BASE_DIR, 'config.yaml')
    config = load_config(config_path)

    device = get_device()
    print("Using device:", device)

    data_path = os.path.join(BASE_DIR, config['data']['input_path'])

    # ================= PREPROCESS =================
    (
        sequences,
        movie2idx,
        idx2movie,
        user2idx,
        idx2user,
        user_history,
        movie_descriptions
    ) = preprocess_sequences(data_path)

    vocab_size = max(movie2idx.values()) + 1

    # ================= TEXT EMBEDDING =================
    pretrained_text_emb = build_text_embedding(
        movie_descriptions,
        vocab_size=vocab_size,
        device=device
    )

    # ================= DATALOADER =================
    train_loader, val_loader = get_dataloaders(
        sequences,
        max_seq_len=config['model']['max_seq_len'],
        batch_size=config['train']['batch_size'],
        val_ratio=config['train']['val_ratio']
    )

    # ================= MODEL =================
    model = BERT4Rec(
        vocab_size=vocab_size,
        embedding_dim=config['model']['embedding_dim'],
        max_seq_len=config['model']['max_seq_len'],
        num_layers=config['model']['num_layers'],
        num_heads=config['model']['num_heads'],
        hidden_dim=config['model']['hidden_dim'],
        text_emb_dim=pretrained_text_emb.size(1),
        pretrained_text_emb=pretrained_text_emb
    ).to(device)

    # ================= OPTIMIZER / LOSS / CALLBACKS =================
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=float(config['train']['lr'])
    )

    criterion = torch.nn.CrossEntropyLoss()

    callbacks = [
        EarlyStopping(
            patience=config['callbacks']['early_stopping_patience'],
            save_path=config['callbacks']['checkpoint_path']
        )
    ]

    # ================= TRAIN =================
    trainer = Trainer(model, optimizer, criterion, callbacks)
    trainer.fit(
        train_loader,
        val_loader,
        config['train']['epochs'],
        device
    )

    # ================= VISUALIZATION =================
    print("Plotting loss curve...")
    plot_loss(
        trainer.history['train_loss'],
        trainer.history['val_loss'],
        save_path=os.path.join(BASE_DIR, 'visualize/loss_train/loss_curve.png')
    )

    metrics_keys = ('precision@5', 'recall@5', 'ndcg@5')
    if all(k in trainer.history for k in metrics_keys):
        print("Plotting metrics...")
        plot_metrics(
            trainer.history,
            keys=metrics_keys,
            save_path=os.path.join(BASE_DIR, 'metrics.png')
        )

    # ================= INFERENCE =================
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
