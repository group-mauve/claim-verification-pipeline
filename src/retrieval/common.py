"""Shared pieces for every retriever (BM25, dense, DPR, later hybrid / rerank).

Run file format (one JSON line per claim), the same for every retriever so they can be compared,
fused (Prompt 9) and fed to the verifier (Prompt 12):
    {"id": 137334, "hits": [["Soul_Food_-LRB-film-RRB-", 0, 12.3], [page, sent_id, score], ...]}
Hits are sorted best first.
"""
from __future__ import annotations

import time
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd

from src.eval.metrics import retrieval_report
from src.utils import (
    clean_text,
    evidence_sets,
    iter_jsonl,
    locate_file,
    read_jsonl,
    readable_title,
    resolve,
    write_jsonl,
)

Run = dict  # claim id -> list of (page_id, sent_id, score)


# --------------------------------------------------------------------------- locating inputs
def find_corpus_dir(cfg: dict) -> Path:
    """Folder holding sentences.parquet: data/corpus locally, or the fever-corpus-v1 Kaggle input."""
    hit = locate_file("sentences.parquet", cfg["paths"]["search_dirs"])
    if hit is None:
        raise FileNotFoundError("corpus/sentences.parquet not found. Add the fever-corpus-v1 dataset as an input.")
    return hit.parent


def find_splits_dir(cfg: dict) -> Path:
    hit = locate_file("test_full.jsonl", cfg["paths"]["search_dirs"])
    if hit is None:
        raise FileNotFoundError("splits/test_full.jsonl not found. Add the fever-corpus-v1 dataset as an input.")
    return hit.parent


def find_output(cfg: dict, run_name: str, filename: str) -> Path | None:
    """Find outputs/<run_name>/<filename> from this session or from an earlier notebook added as input.

    filename may include subfolders, e.g. "embeddings/meta.json".
    """
    local = resolve(cfg["paths"]["outputs_dir"]) / run_name / filename
    if local.exists():
        return local
    tail = (run_name, *Path(filename).parts)
    for d in cfg["paths"]["search_dirs"]:
        root = resolve(d)
        if root.exists():
            for hit in root.rglob(Path(filename).name):
                if hit.parts[-len(tail):] == tail:
                    return hit
    return None


# --------------------------------------------------------------------------- corpus and claims
def load_corpus(corpus_dir: Path) -> pd.DataFrame:
    """Corpus sentences in a fixed order (page_id, sent_id). Row number = document index everywhere."""
    df = pd.read_parquet(corpus_dir / "sentences.parquet", columns=["page_id", "sent_id", "text"])
    return df.sort_values(["page_id", "sent_id"]).reset_index(drop=True)


def doc_text(page_id: str, text: str, sep: str = ": ") -> str:
    """What a retriever sees for one sentence: 'Soul Food (film): Soul Food is a 1997 ...'.

    The title matters because sentences often say 'He' or 'The film' instead of the name.
    """
    return readable_title(page_id) + sep + clean_text(text)


def test_split_name(cfg: dict) -> str:
    return "test_5k" if str(cfg.get("test_subset", "full")) == "5k" else "test_full"


def load_claims(splits_dir: Path, split: str) -> list[dict]:
    return read_jsonl(splits_dir / f"{split}.jsonl")


def gold_map(claims: list[dict]) -> dict:
    """claim id -> list of evidence sets (empty for NEI claims, which retrieval metrics skip)."""
    return {c["id"]: evidence_sets(c) for c in claims}


# --------------------------------------------------------------------------- run files
def write_run(run: Run, path: Path) -> None:
    write_jsonl(({"id": cid, "hits": [[p, int(s), round(float(sc), 6)] for p, s, sc in hits]} for cid, hits in run.items()), path)


def read_run(path: Path) -> Run:
    return {row["id"]: [(p, int(s), float(sc)) for p, s, sc in row["hits"]] for row in iter_jsonl(path)}


def ranked_ids(hits) -> list[tuple[str, int]]:
    return [(p, s) for p, s, *_ in hits]


# --------------------------------------------------------------------------- evaluation
def evaluate_run(run: Run, claims: list[dict], ks=(5, 10, 20), ndcg_k: int = 10) -> dict:
    """Recall@K, MRR, nDCG@K at sentence and page level over the S/R claims of `claims`."""
    runs = {cid: ranked_ids(h) for cid, h in run.items()}
    return retrieval_report(runs, gold_map(claims), ks=ks, ndcg_k=ndcg_k)


def time_per_query(search_one: Callable[[str], object], queries: list[str], n: int = 200) -> float:
    """Milliseconds per query when queries are answered one at a time (what a user would see)."""
    sample = queries[:n]
    if not sample:
        return float("nan")
    search_one(sample[0])  # warm-up (JIT, caches, CUDA kernels)
    t0 = time.perf_counter()
    for q in sample:
        search_one(q)
    return 1000 * (time.perf_counter() - t0) / len(sample)


def format_report(report: dict, extra: dict | None = None) -> str:
    """Readable two-line table of a retrieval_report for the logs."""
    lines = []
    for level in ("page", "sentence"):
        row = report.get(level, {})
        lines.append(f"{level:9s} " + "  ".join(f"{k} {v:.4f}" for k, v in row.items()))
    if extra:
        lines.append("          " + "  ".join(f"{k} {v}" for k, v in extra.items()))
    return "\n".join(lines)


# --------------------------------------------------------------------------- exact vector search
def exact_search(index_vectors: np.ndarray, queries: np.ndarray, k: int, batch: int = 1024) -> tuple[np.ndarray, np.ndarray]:
    """Exact inner-product top-k with FAISS IndexFlatIP when installed, numpy otherwise.

    Returns (scores, indices), each of shape (n_queries, k), best first. Both paths give the
    same result; numpy is only for machines without faiss (e.g. the unit tests).
    """
    k = min(k, len(index_vectors))
    try:
        import faiss

        index = faiss.IndexFlatIP(index_vectors.shape[1])
        index.add(np.ascontiguousarray(index_vectors, dtype=np.float32))
        return index.search(np.ascontiguousarray(queries, dtype=np.float32), k)
    except ImportError:
        all_s, all_i = [], []
        for i in range(0, len(queries), batch):
            sims = queries[i : i + batch] @ index_vectors.T
            top = np.argpartition(-sims, k - 1, axis=1)[:, :k]
            part = np.take_along_axis(sims, top, axis=1)
            order = np.argsort(-part, axis=1, kind="stable")
            all_i.append(np.take_along_axis(top, order, axis=1))
            all_s.append(np.take_along_axis(part, order, axis=1))
        return np.vstack(all_s), np.vstack(all_i)


def hits_from_indices(corpus: pd.DataFrame, scores: np.ndarray, indices: np.ndarray) -> list[list[tuple[str, int, float]]]:
    pages, sids = corpus["page_id"].to_numpy(), corpus["sent_id"].to_numpy()
    return [[(pages[j], int(sids[j]), float(s)) for s, j in zip(srow, irow) if j >= 0] for srow, irow in zip(scores, indices)]


# --------------------------------------------------------------------------- side-by-side examples
def per_claim_rr(run: Run, claims: list[dict]) -> dict:
    """Sentence-level reciprocal rank per S/R claim (used to find where one system beats another)."""
    from src.eval.metrics import reciprocal_rank

    out = {}
    for c in claims:
        sets = evidence_sets(c)
        if sets and c["id"] in run:
            out[c["id"]] = reciprocal_rank(ranked_ids(run[c["id"]]), {x for g in sets for x in g})
    return out


def biggest_wins(rr_a: dict, rr_b: dict, n: int = 3) -> list:
    """Claim ids where system A beats system B by the largest reciprocal-rank margin."""
    common = [cid for cid in rr_a if cid in rr_b]
    diffs = sorted(common, key=lambda cid: rr_a[cid] - rr_b[cid], reverse=True)
    return [cid for cid in diffs if rr_a[cid] - rr_b[cid] > 0][:n]


def show_hits(hits, gold: set, texts: dict, n: int = 3) -> list[str]:
    """Top-n hits as printable lines; correct evidence marked with [GOLD]."""
    lines = []
    for rank, (p, s, sc) in enumerate(hits[:n], start=1):
        mark = "[GOLD]" if (p, s) in gold else "      "
        lines.append(f"   {rank}. {mark} ({readable_title(p)}, {s}) {sc:.3f} | {texts.get((p, s), '')[:150]}")
    return lines
