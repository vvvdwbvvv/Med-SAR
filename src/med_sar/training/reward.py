from dataclasses import dataclass
from med_sar.models.embeddings import Embedder, cosine_sim

@dataclass
class CycleConsistencyReward:
    embedder: Embedder
    threshold: float = 0.75

    def score(self, clean: str, adv: str) -> float:
        a, b = self.embedder.encode([clean, adv])
        sim = cosine_sim(a, b)
        # Reward shaped to heavily penalize semantic drift
        return sim if sim >= self.threshold else -1.0 * (self.threshold - sim)
