from sentence_transformers import SentenceTransformer
import numpy as np

class Embedder:
    def __init__(self, name_or_path: str):
        self.model = SentenceTransformer(name_or_path)

    def encode(self, texts: list[str]) -> np.ndarray:
        return self.model.encode(texts, normalize_embeddings=True)

def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    return float((a * b).sum())
