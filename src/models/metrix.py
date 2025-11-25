import numpy as np
import math

def precision_recall_at_k(preds, targets, k=10):
    precisions, recalls = [], []
    for p, t in zip(preds, targets):
        p_set = set(p[:k])
        hit = 1 if int(t) in p_set else 0
        precisions.append(hit / k)
        recalls.append(hit)
    return np.mean(precisions), np.mean(recalls)


def ndcg_at_k(preds, targets, k=10):
    scores = []
    for p, t in zip(preds, targets):
        dcg = 0.0
        idcg = 1.0  # chỉ 1 item relevant
        for i, pid in enumerate(p[:k]):
            if int(pid) == int(t):
                dcg = 1.0 / math.log2(i + 2)
                break
        scores.append(dcg / idcg)
    return np.mean(scores)
