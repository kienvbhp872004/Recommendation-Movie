import torch
from torch.utils.data import DataLoader

from src.utils.config import load_config
from src.utils.gpu import get_device
from src.datasets.dataloader import MovieDataset
from src.datasets.preprocess import load_sequences
from src.datasets.split import train_val_split
from src.models.recommender import BERT4Rec
from src.train.trainer import Trainer
from src.train.callbacks import EarlyStopping

# Load config và device
config = load_config('config.yaml')
device = get_device()

# Load sequences và split train/val
sequences = load_sequences(config['data']['input_path'])
train_seq, val_seq = train_val_split(sequences, config['train']['val_ratio'])

# DataLoader
train_loader = DataLoader(
    MovieDataset(train_seq, config['model']['max_seq_len']),
    batch_size=config['train']['batch_size'],
    shuffle=True
)
val_loader = DataLoader(
    MovieDataset(val_seq, config['model']['max_seq_len']),
    batch_size=config['train']['batch_size']
)

# Model
model = BERT4Rec(
    vocab_size=config['model']['vocab_size'],
    embedding_dim=config['model']['embedding_dim'],
    max_seq_len=config['model']['max_seq_len'],
    num_layers=config['model']['num_layers'],
    num_heads=config['model']['num_heads'],
    hidden_dim=config['model']['hidden_dim']
).to(device)

# Optimizer, criterion, callbacks
optimizer = torch.optim.Adam(model.parameters(), lr=config['train']['lr'])
criterion = torch.nn.CrossEntropyLoss()
callbacks = [
    EarlyStopping(
        patience=config['callbacks']['early_stopping_patience'],
        save_path=config['callbacks']['checkpoint_path']
    )
]

# Trainer
trainer = Trainer(model, optimizer, criterion, callbacks)
trainer.fit(train_loader, val_loader, config['train']['epochs'], device)
