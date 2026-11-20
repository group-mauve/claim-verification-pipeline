"""Prompt 6 (M2): dense retrieval with a sentence-transformers bi-encoder and an exact FAISS index.

Run (GPU session):
    python -m src.retrieval.dense --config configs/dense_offshelf.yaml

The same code serves the fine-tuned model in Prompt 8: src/retrieval/finetune.py calls run_dense with
model_name = models/minilm_ft and run_name = dense_ft.

Steps
1. Encode every corpus sentence as "title: sentence" with all-MiniLM-L6-v2 (384-d), L2-normalised,
   float32. Encoding runs in shards of 50k sentences saved as .npy files, so a session timeout only
   loses the current shard; a rerun skips finished shards.
2. Put all vectors in a FAISS IndexFlatIP. On unit-length vectors inner product = cosine similarity,
   and Flat means exact search (every sentence is scored, no approximation).
3. Encode the claims the same way and take the top 100 sentences per claim, for val and test.
4. Report the BM25 metrics plus index size (GB) and ms/query (encode one claim + search).
5. Print 3 claims where dense beats BM25 and 3 where BM25 beats dense.
Outputs in outputs/<run_name>/: run_test.jsonl, run_val.jsonl, metrics.json, examples.json,
embeddings/emb_NNN.npy + meta.json, index.faiss.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Callable

import numpy as np

from src.retrieval.common import (
    biggest_wins,
    doc_text,
    evaluate_run,
    find_corpus_dir,
    find_model_dir,
    find_output,
    find_splits_dir,
    format_report,
    hits_from_indices,
    load_claims,
    load_corpus,
    per_claim_rr,
    read_run,
    show_hits,
    test_split_name,
    time_per_query,
    write_run,
)
from src.utils import clean_text, evidence_sets, load_config, save_json, set_seed, start_run


# --------------------------------------------------------------------------- sharded encoding
def encode_in_shards(texts: list[str], encode: Callable[[list[str]], np.ndarray], out_dir: Path, shard_size: int,
                     meta: dict, log) -> np.ndarray:
    """Encode `texts` shard by shard into out_dir/emb_NNN.npy; reuse finished shards. Returns all vectors.

    meta (model name, number of texts, ...) is stored in meta.json; if it changes (other model or
    corpus) the old shards are not reused.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    meta = {**meta, "n": len(texts), "shard_size": shard_size}
    meta_path = out_dir / "meta.json"
    if meta_path.exists() and json.loads(meta_path.read_text()) != meta:
        log.warning("Existing embeddings were made with different settings; re-encoding.")
        for f in out_dir.glob("emb_*.npy"):
            f.unlink()
    meta_path.write_text(json.dumps(meta, indent=2))

    shards = []
    n_shards = (len(texts) + shard_size - 1) // shard_size
    t0 = time.time()
    for i in range(n_shards):
        path = out_dir / f"emb_{i:03d}.npy"
        chunk = texts[i * shard_size : (i + 1) * shard_size]
        if path.exists():
            arr = np.load(path)
            if len(arr) == len(chunk):
                shards.append(arr)
                log.info(f"  shard {i + 1}/{n_shards}: reused {path.name}")
                continue
        arr = encode(chunk).astype(np.float32)
        np.save(path, arr)
        shards.append(arr)
        log.info(f"  shard {i + 1}/{n_shards}: encoded {len(chunk):,} in {time.time() - t0:.0f}s total")
    return np.vstack(shards)


def load_bi_encoder(model_name: str, fp16: bool):
    import torch
    from sentence_transformers import SentenceTransformer

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = SentenceTransformer(model_name, device=device)
    if fp16 and device == "cuda":
        model.half()
    return model, device


def embedding_dim(model) -> int:
    """Vector size; newer sentence-transformers renamed get_sentence_embedding_dimension."""
    fn = getattr(model, "get_embedding_dimension", None) or model.get_sentence_embedding_dimension
    return int(fn())


def load_embeddings(cfg: dict, run_name: str, n_expected: int) -> np.ndarray:
    """Corpus vectors saved by run_dense (outputs/<run_name>/embeddings/emb_NNN.npy), in corpus row order."""
    from src.retrieval.common import require_output

    emb_dir = require_output(cfg, run_name, "embeddings/meta.json").parent
    vectors = np.vstack([np.load(p) for p in sorted(emb_dir.glob("emb_*.npy"))]).astype(np.float32)
    if len(vectors) != n_expected:
        raise ValueError(f"{emb_dir} holds {len(vectors):,} vectors but the corpus has {n_expected:,} sentences")
    return vectors


def st_encoder(model, batch_size: int) -> Callable[[list[str]], np.ndarray]:
    def encode(texts: list[str]) -> np.ndarray:
        return model.encode(texts, batch_size=batch_size, convert_to_numpy=True, normalize_embeddings=True,
                            show_progress_bar=len(texts) > 10_000).astype(np.float32)
    return encode


class FlatIndex:
    """FAISS IndexFlatIP (exact inner product); numpy fallback if faiss is missing."""

    def __init__(self, vectors: np.ndarray):
        self.vectors = np.ascontiguousarray(vectors, dtype=np.float32)
        try:
            import faiss

            self.index = faiss.IndexFlatIP(self.vectors.shape[1])
            self.index.add(self.vectors)
        except ImportError:
            self.index = None

    def search(self, queries: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
        q = np.ascontiguousarray(queries, dtype=np.float32)
        if self.index is not None:
            return self.index.search(q, k)
        from src.retrieval.common import exact_search

        return exact_search(self.vectors, q, k)

    def save(self, path: Path) -> None:
        if self.index is not None:
            import faiss

            faiss.write_index(self.index, str(path))

    @property
    def size_gb(self) -> float:
        return self.vectors.nbytes / 1024**3


def dense_retrieve(index: FlatIndex, encode, corpus, claims: list[dict], k: int) -> dict:
    q = encode([clean_text(c["claim"]) for c in claims])
    scores, idx = index.search(q, k)
    return {c["id"]: hits for c, hits in zip(claims, hits_from_indices(corpus, scores, idx))}


# --------------------------------------------------------------------------- comparison with BM25
def compare_runs(name_a: str, run_a: dict, name_b: str, run_b: dict, claims: list[dict], corpus, log, n: int = 3) -> list[dict]:
    texts = {(p, s): t for p, s, t in zip(corpus["page_id"], corpus["sent_id"], corpus["text"])}
    rr_a, rr_b = per_claim_rr(run_a, claims), per_claim_rr(run_b, claims)
    by_id = {c["id"]: c for c in claims}
    examples = []
    for winner, loser, wins in ((name_a, name_b, biggest_wins(rr_a, rr_b, n)), (name_b, name_a, biggest_wins(rr_b, rr_a, n))):
        log.info("=" * 90)
        log.info(f"Claims where {winner} beats {loser} (sentence-level reciprocal rank):")
        for cid in wins:
            c = by_id[cid]
            gold = {x for g in evidence_sets(c) for x in g}
            log.info(f"[{c['label']}] {c['claim']}   RR {name_a}={rr_a[cid]:.2f} {name_b}={rr_b[cid]:.2f}")
            log.info(f"  gold: {sorted(gold)[:3]}")
            for name, run in ((name_a, run_a), (name_b, run_b)):
                log.info(f"  {name} top 3:")
                for line in show_hits(run[cid], gold, texts):
                    log.info(line)
            examples.append({"winner": winner, "id": cid, "claim": c["claim"], "label": c["label"],
                             f"rr_{name_a}": rr_a[cid], f"rr_{name_b}": rr_b[cid],
                             f"top3_{name_a}": run_a[cid][:3], f"top3_{name_b}": run_b[cid][:3]})
    return examples


# --------------------------------------------------------------------------- main
def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--config", default="configs/dense_offshelf.yaml")
    args = parser.parse_args(argv)
    run_dense(load_config(args.config))
    return 0


def run_dense(cfg: dict) -> dict:
    """Encode the corpus, build the index, retrieve for val and test, evaluate. Returns the metrics."""
    set_seed(cfg["seed"])
    run_dir, log = start_run(cfg)
    dc = cfg["dense"]

    corpus_dir, splits_dir = find_corpus_dir(cfg), find_splits_dir(cfg)
    corpus = load_corpus(corpus_dir)
    log.info(f"Corpus: {len(corpus):,} sentences from {corpus_dir}")

    model_path = find_model_dir(cfg, dc["model_name"])
    model, device = load_bi_encoder(model_path, dc["fp16"])
    log.info(f"Model {model_path} on {device} (fp16={dc['fp16'] and device == 'cuda'}), dim {embedding_dim(model)}")
    encode = st_encoder(model, dc["batch_size"])

    texts = [doc_text(p, t) for p, t in zip(corpus["page_id"], corpus["text"])]
    log.info(f"Example input: {texts[0][:120]}")
    t0 = time.time()
    vectors = encode_in_shards(texts, encode, run_dir / "embeddings", dc["shard_size"],
                               {"model": dc["model_name"], "text_format": dc["text_format"]}, log)
    encode_s = time.time() - t0
    norms = np.linalg.norm(vectors[:1000], axis=1)
    log.info(f"Embeddings {vectors.shape} {vectors.dtype}; norms of first 1000 in [{norms.min():.3f}, {norms.max():.3f}]")
    corpus[["page_id", "sent_id"]].to_parquet(run_dir / "embeddings" / "ids.parquet", index=False)

    index = FlatIndex(vectors)
    if dc["save_faiss_index"]:
        index.save(run_dir / "index.faiss")
    log.info(f"Index: FAISS IndexFlatIP, {index.size_gb:.3f} GB")

    val = load_claims(splits_dir, "val")
    val_run = dense_retrieve(index, encode, corpus, val, dc["top_k"])
    write_run(val_run, run_dir / "run_val.jsonl")

    split = test_split_name(cfg)
    test = load_claims(splits_dir, split)
    t0 = time.time()
    test_run = dense_retrieve(index, encode, corpus, test, dc["top_k"])
    batch_ms = 1000 * (time.time() - t0) / len(test)
    write_run(test_run, run_dir / "run_test.jsonl")
    ms = time_per_query(lambda q: index.search(encode([q]), dc["top_k"]), [clean_text(c["claim"]) for c in test], dc["latency_queries"])

    metrics = {
        "model": dc["model_name"], "device": device, "corpus_sentences": len(corpus), "dim": int(vectors.shape[1]),
        "corpus_encode_seconds": round(encode_s, 1), "index_size_gb": round(index.size_gb, 3),
        "val": evaluate_run(val_run, val), "test_split": split, "test": evaluate_run(test_run, test),
        "ms_per_query": round(ms, 2), "batch_ms_per_query": round(batch_ms, 2),
    }
    log.info(f"VAL:\n" + format_report(metrics["val"]))
    log.info(f"TEST ({split}):\n" + format_report(metrics["test"], {"ms/query": metrics["ms_per_query"], "index GB": metrics["index_size_gb"]}))

    other = dc.get("compare_with")
    other_path = find_output(cfg, other, "run_test.jsonl") if other else None
    if other_path:
        log.info(f"Comparing with {other} run {other_path}")
        examples = compare_runs("dense", test_run, other, read_run(other_path), test, corpus, log)
        save_json(examples, run_dir / "examples.json")
    elif other:
        log.warning(f"No {other}/run_test.jsonl found; skipping the side-by-side examples. "
                    "On Kaggle, add the BM25 notebook's output as an input.")
    save_json(metrics, run_dir / "metrics.json")
    log.info(f"Outputs written to {run_dir}")
    return metrics


if __name__ == "__main__":
    sys.exit(main())
