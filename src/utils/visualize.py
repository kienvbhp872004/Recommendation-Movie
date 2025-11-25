# src/utils/visualize.py

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


def plot_loss(train_losses, val_losses, save_path=None):
    """
    Vẽ curve loss của quá trình training.

    Args:
        train_losses (list[float]): loss mỗi epoch của train.
        val_losses (list[float]): loss mỗi epoch của validation.
        save_path (str, optional): nếu có, lưu ảnh ra file.
    """
    plt.figure()
    plt.plot(train_losses, label='Train Loss', marker='o')
    plt.plot(val_losses, label='Val Loss', marker='o')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Loss Curve')
    plt.legend()
    plt.grid(True)
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
    plt.show()
    plt.close()


def plot_metrics(history, keys=('precision@5', 'recall@5', 'ndcg@5'), save_path=None):
    """
    Vẽ các metric theo epoch.

    Args:
        history (dict): dict chứa danh sách metrics theo epoch.
        keys (tuple): metrics cần vẽ.
        save_path (str, optional): nếu có, lưu ảnh ra file.
    """
    plt.figure()
    for k in keys:
        if k in history:
            plt.plot(history[k], label=k, marker='o')
    plt.xlabel('Epoch')
    plt.ylabel('Score')
    plt.title('Metrics per Epoch')
    plt.legend()
    plt.grid(True)
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
    plt.show()
    plt.close()


def plot_similarity_heatmap(sim, indices=None, figsize=(10, 8), save_path=None):
    """
    Vẽ heatmap ma trận similarity của embeddings.

    Args:
        sim (np.array): ma trận similarity (V,V).
        indices (list, optional): nếu muốn xem subset của items.
        figsize (tuple): size figure.
        save_path (str, optional): nếu có, lưu ảnh ra file.
    """
    if indices is not None:
        sim_sample = sim[np.ix_(indices, indices)]
    else:
        sim_sample = sim
    plt.figure(figsize=figsize)
    sns.heatmap(sim_sample, cmap='viridis')
    plt.title('Similarity Heatmap')
    plt.xlabel('Item Index')
    plt.ylabel('Item Index')
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
    plt.show()
    plt.close()


def plot_topk_distribution(predictions, top_n=50, save_path=None):
    """
    Vẽ tần suất xuất hiện của các item dự đoán top-k.

    Args:
        predictions (list or np.array): predicted item ids.
        top_n (int): số item top frequent muốn hiển thị.
        save_path (str, optional): nếu có, lưu ảnh ra file.
    """
    vals, counts = np.unique(predictions, return_counts=True)
    order = np.argsort(-counts)[:top_n]
    plt.figure(figsize=(12, 4))
    plt.bar(range(len(order)), counts[order], tick_label=vals[order])
    plt.xlabel('Item ID')
    plt.ylabel('Frequency')
    plt.title(f'Top-{top_n} Predicted Item Frequency')
    plt.xticks(rotation=90)
    plt.grid(axis='y')
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
    plt.show()
    plt.close()
