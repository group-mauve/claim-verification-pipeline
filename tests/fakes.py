"""Stand-ins for the neural models, so every script can run in the tests without a GPU or downloads."""
import re
import zlib

import numpy as np

_WORD = re.compile(r"\w+")


class FakeST:
    """Stands in for SentenceTransformer: bag-of-hashed-words vectors, L2-normalised."""

    def __init__(self, dim=64):
        self.dim = dim

    def get_sentence_embedding_dimension(self):
        return self.dim

    def encode(self, texts, normalize_embeddings=True, **_):
        out = np.zeros((len(texts), self.dim), np.float32)
        for r, t in enumerate(texts):
            for w in t.lower().replace(":", " ").split():
                out[r, zlib.crc32(w.encode()) % self.dim] += 1
        n = np.linalg.norm(out, axis=1, keepdims=True)
        return out / np.maximum(n, 1e-9) if normalize_embeddings else out


def overlap(a: str, b: str) -> float:
    """Share of the words of b that also occur in a."""
    wa, wb = set(_WORD.findall(a.lower())), set(_WORD.findall(b.lower()))
    return len(wa & wb) / max(len(wb), 1)


class FakeNLIScorer:
    """Stands in for NLIScorer: entailment grows with word overlap; 'never' in the claim means contradiction."""

    def __init__(self, name, fp16=True, max_length=256, batch_size=8, log=None):
        self.name = name

    def logits(self, premises, hypotheses):
        out = []
        for p, h in zip(premises, hypotheses):
            o = overlap(p, h)
            out.append([5 * o - 1, 3.0 if "never" in h.lower() else -1.0, 1 - 2 * o])
        return np.array(out, np.float32).reshape(-1, 3)

    def sanity_check(self):
        pass
