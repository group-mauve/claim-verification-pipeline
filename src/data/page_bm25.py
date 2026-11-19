"""A small in-memory BM25 built on scipy sparse matrices.

Used only by build_corpus.py to find look-alike "decoy" pages among all 5.4M Wikipedia pages
(document = page title + first sentence). The main sentence retriever in Prompt 5 is Pyserini.

How it works:
1. Turn every document into a row of term counts (HashingVectorizer: no vocabulary to store,
   English stop words removed so very common words do not make every document match).
2. Replace each count tf with its BM25 weight:
       idf(t) * tf * (k1 + 1) / (tf + k1 * (1 - b + b * doc_len / avg_doc_len))
   with idf(t) = log(1 + (N - df + 0.5) / (df + 0.5)).
3. A query is a 0/1 vector over the same terms, so the BM25 score of every document is one
   sparse matrix product: scores = query_vector @ weights.T.
"""
from __future__ import annotations

import numpy as np
import scipy.sparse as sp
from sklearn.feature_extraction.text import HashingVectorizer

N_FEATURES = 2**23  # ~8M hash buckets; collisions are rare enough for this purpose
TOKEN_PATTERN = r"(?u)\b\w\w+\b"


def _vectorizer(binary: bool) -> HashingVectorizer:
    return HashingVectorizer(
        n_features=N_FEATURES,
        alternate_sign=False,
        norm=None,
        binary=binary,
        lowercase=True,
        stop_words="english",
        token_pattern=TOKEN_PATTERN,
        dtype=np.float32,
    )


def clean(text: str) -> str:
    """Drop FEVER's bracket tokens (-LRB-, -RRB-, ...) and underscores in titles."""
    for tok in ("-LRB-", "-RRB-", "-LSB-", "-RSB-", "-LCB-", "-RCB-", "-COLON-"):
        text = text.replace(tok, " ")
    return text.replace("_", " ")


class SparseBM25:
    def __init__(self, k1: float = 0.9, b: float = 0.4):
        self.k1, self.b = k1, b
        self.weights_t: sp.csr_matrix | None = None  # terms x docs

    def fit(self, docs: list[str], chunk: int = 500_000) -> "SparseBM25":
        vec = _vectorizer(binary=False)
        # Vectorise in chunks to keep peak memory low on 5.4M documents.
        tf = sp.vstack([vec.transform([clean(d) for d in docs[i : i + chunk]]) for i in range(0, len(docs), chunk)]).tocsr()
        n_docs = tf.shape[0]
        doc_len = np.asarray(tf.sum(axis=1)).ravel()
        avg_len = max(doc_len.mean(), 1e-9)
        df = np.bincount(tf.indices, minlength=N_FEATURES)
        idf = np.log1p((n_docs - df + 0.5) / (df + 0.5)).astype(np.float32)

        rows = np.repeat(np.arange(n_docs), np.diff(tf.indptr))
        t = tf.data
        norm = self.k1 * (1 - self.b + self.b * doc_len[rows] / avg_len)
        tf.data = (idf[tf.indices] * t * (self.k1 + 1) / (t + norm)).astype(np.float32)
        self.weights_t = tf.T.tocsr()
        self.n_docs = n_docs
        return self

    def search(self, queries: list[str], k: int, batch_size: int = 64) -> list[list[tuple[int, float]]]:
        """Return, for each query, the top-k (doc_index, score) pairs, best first."""
        vec = _vectorizer(binary=True)
        results = []
        for i in range(0, len(queries), batch_size):
            q = vec.transform([clean(x) for x in queries[i : i + batch_size]])
            scores = (q @ self.weights_t).tocsr()
            for r in range(scores.shape[0]):
                start, end = scores.indptr[r], scores.indptr[r + 1]
                data, idx = scores.data[start:end], scores.indices[start:end]
                if len(data) > k:
                    top = np.argpartition(-data, k)[:k]
                    data, idx = data[top], idx[top]
                order = np.argsort(-data, kind="stable")
                results.append([(int(idx[j]), float(data[j])) for j in order])
        return results
