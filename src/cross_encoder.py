"""Score (text A, text B) pairs with a Hugging Face sequence-classification model.

Used by the cross-encoder reranker (Prompt 10: A = claim, B = sentence, one relevance logit) and by the
NLI verifier (Prompt 12: A = evidence, B = claim, three logits). Both read the two texts together, so
every word of one can attend to every word of the other; that is why they are slower but more precise
than the bi-encoder, which encodes the two texts separately.
"""
from __future__ import annotations

import numpy as np


class PairModel:
    def __init__(self, name: str, fp16: bool = True, max_length: int = 256, batch_size: int = 128):
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        self.name = name
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.tokenizer = AutoTokenizer.from_pretrained(name)
        self.model = AutoModelForSequenceClassification.from_pretrained(name).to(self.device).eval()
        self.id2label = {int(k): v for k, v in self.model.config.id2label.items()}
        self.use_amp = fp16 and self.device == "cuda"
        self.max_length = max_length
        self.batch_size = batch_size

    def _forward(self, a: list[str], b: list[str], amp: bool) -> np.ndarray:
        import torch

        enc = self.tokenizer(a, b, truncation=True, max_length=self.max_length, padding=True, return_tensors="pt")
        enc = {k: v.to(self.device) for k, v in enc.items()}
        with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.float16, enabled=amp):
            return self.model(**enc).logits.float().cpu().numpy()

    def logits(self, first: list[str], second: list[str], progress: bool = False) -> np.ndarray:
        """Logits of shape (n_pairs, n_labels), in the input order.

        Pairs are sorted by length before batching so each batch holds similar lengths (less padding,
        often 2x faster). A batch that overflows in fp16 (NaN/inf) is recomputed in fp32.
        """
        n = len(first)
        if n == 0:
            return np.zeros((0, len(self.id2label)), np.float32)
        order = np.argsort([len(x) + len(y) for x, y in zip(first, second)], kind="stable")
        out = np.zeros((n, len(self.id2label)), np.float32)
        steps = range(0, n, self.batch_size)
        if progress:
            try:
                from tqdm.auto import tqdm

                steps = tqdm(steps, desc=self.name.split("/")[-1], mininterval=10)
            except ImportError:
                pass
        for i in steps:
            idx = order[i : i + self.batch_size]
            a, b = [first[j] for j in idx], [second[j] for j in idx]
            res = self._forward(a, b, self.use_amp)
            if self.use_amp and not np.isfinite(res).all():
                res = self._forward(a, b, False)
            out[idx] = res
        return out
