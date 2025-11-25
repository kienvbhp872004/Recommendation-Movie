import numpy as np
from sklearn.metrics.pairwise import cosine_similarity


def get_movie_embeddings(model):
    return model.embedding.weight.detach().cpu().numpy()


def compute_similarity_matrix(embeddings, top_k=None):
    sim = cosine_similarity(embeddings)
    if top_k:
        out = np.zeros_like(sim)
        for i in range(sim.shape[0]):
            idx = np.argsort(-sim[i])[:top_k]
            out[i, idx] = sim[i, idx]
        return out
    return sim