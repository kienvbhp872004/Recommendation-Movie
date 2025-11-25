import pandas as pd


def load_sequences(path):
    df = pd.read_csv(path)
    sequences = df['sequence'].apply(eval).tolist()
    return sequences