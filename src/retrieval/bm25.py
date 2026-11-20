"""Prompt 5 (M1): BM25 sentence retrieval with Pyserini (Lucene), plus a rank_bm25 fallback.

Run:
    python -m src.retrieval.bm25 --config configs/bm25.yaml            # full run (Pyserini, needs Java 21)
    python -m src.retrieval.bm25 --config configs/bm25.yaml --quick    # rank_bm25 on 50k sentences, no Java

Steps
1. Write every corpus sentence as a Lucene document: id "page_id:sent_id", text "page title sentence".
2. Build the Lucene index once (Pyserini's default English analyzer: lowercase, stop words, Porter stemming).
3. Tune k1 (term-frequency saturation) and b (document-length normalisation) on the validation claims
   by sentence-level Recall@10.
4. With the best k1, b retrieve the top 100 sentences for every test claim and evaluate at sentence and
   page level (a page's rank = rank of its best sentence). Also time single queries (ms/query).
5. Also save BM25 runs for val (Prompt 9 fusion tuning) and train (Prompt 8 hard negatives).
Outputs in outputs/bm25/: run_test.jsonl, run_val.jsonl, run_train.jsonl, metrics.json, lucene_index/.
"""
from __future__ import annotations

import argparse
import random
import re
import shutil
import subprocess
import sys
import time
from collections import namedtuple
from pathlib import Path

import numpy as np

from src.retrieval.common import (
    doc_text,
    evaluate_run,
    find_corpus_dir,
    find_splits_dir,
    format_report,
    load_claims,
    load_corpus,
    test_split_name,
    time_per_query,
    write_run,
)
from src.utils import evidence_pages, load_config, save_json, set_seed, start_run, write_jsonl

Hit = namedtuple("Hit", ["docid", "score"])


def make_docid(page_id: str, sent_id: int) -> str:
    return f"{page_id}:{sent_id}"


def parse_docid(docid: str) -> tuple[str, int]:
    """'Telemundo:4' -> ('Telemundo', 4). rsplit, so a ':' inside a title cannot break it."""
    page, sid = docid.rsplit(":", 1)
    return page, int(sid)


# --------------------------------------------------------------------------- backends
class LuceneBM25:
    """Thin wrapper around Pyserini's LuceneSearcher with a uniform (docid, score) interface."""

    def __init__(self, index_dir: Path):
        from pyserini.search.lucene import LuceneSearcher

        self.searcher = LuceneSearcher(str(index_dir))

    def set_bm25(self, k1: float, b: float) -> None:
        self.searcher.set_bm25(k1, b)

    def search(self, query: str, k: int) -> list[Hit]:
        return [Hit(h.docid, h.score) for h in self.searcher.search(query, k)]

    def batch_search(self, queries: list[str], qids: list[str], k: int, threads: int) -> dict[str, list[Hit]]:
        res = self.searcher.batch_search(queries, qids, k=k, threads=threads)
        return {q: [Hit(h.docid, h.score) for h in hits] for q, hits in res.items()}


_TOKEN = re.compile(r"\w+")


def tokenize(text: str) -> list[str]:
    from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

    return [t for t in _TOKEN.findall(text.lower()) if t not in ENGLISH_STOP_WORDS]


class RankBM25:
    """Pure-Python BM25 (rank_bm25 package) for quick tests on <= ~50k sentences. Slow on more."""

    def __init__(self, docids: list[str], texts: list[str]):
        self.docids = np.array(docids)
        self.tokens = [tokenize(t) for t in texts]
        self.model = None

    def set_bm25(self, k1: float, b: float) -> None:
        from rank_bm25 import BM25Okapi

        self.model = BM25Okapi(self.tokens, k1=k1, b=b)

    def search(self, query: str, k: int) -> list[Hit]:
        scores = self.model.get_scores(tokenize(query))
        k = min(k, len(scores))
        top = np.argpartition(-scores, k - 1)[:k]
        top = top[np.argsort(-scores[top], kind="stable")]
        return [Hit(self.docids[i], float(scores[i])) for i in top if scores[i] > 0]

    def batch_search(self, queries, qids, k, threads=1):
        return {q: self.search(text, k) for q, text in zip(qids, queries)}


# --------------------------------------------------------------------------- index building
def build_lucene_index(corpus, run_dir: Path, threads: int, log, rebuild: bool = False) -> Path:
    index_dir = run_dir / "lucene_index"
    if index_dir.exists() and any(index_dir.iterdir()) and not rebuild:
        log.info(f"Using existing Lucene index {index_dir}")
        return index_dir
    coll_dir = run_dir / "collection"
    coll_dir.mkdir(parents=True, exist_ok=True)
    n = write_jsonl(
        ({"id": make_docid(p, s), "contents": doc_text(p, t, sep=" ")} for p, s, t in zip(corpus["page_id"], corpus["sent_id"], corpus["text"])),
        coll_dir / "docs.jsonl",
    )
    log.info(f"Wrote {n:,} documents to {coll_dir}; indexing with Pyserini ...")
    t0 = time.time()
    cmd = [
        sys.executable, "-m", "pyserini.index.lucene",
        "--collection", "JsonCollection", "--input", str(coll_dir), "--index", str(index_dir),
        "--generator", "DefaultLuceneDocumentGenerator", "--threads", str(threads),
    ]
    subprocess.run(cmd, check=True)
    shutil.rmtree(coll_dir, ignore_errors=True)  # the index holds everything we need
    log.info(f"Index built in {time.time() - t0:.0f}s at {index_dir}")
    return index_dir


# --------------------------------------------------------------------------- retrieval helpers
def retrieve(searcher, claims: list[dict], k: int, threads: int) -> dict:
    """claim id -> [(page, sent_id, score)], best first."""
    qids = [str(c["id"]) for c in claims]
    res = searcher.batch_search([c["claim"] for c in claims], qids, k=k, threads=threads)
    return {c["id"]: [(*parse_docid(h.docid), h.score) for h in res.get(str(c["id"]), [])] for c in claims}


def quick_sample(corpus, claims: list[dict], n_docs: int, n_claims: int, seed: int):
    """50k-sentence sample that contains every evidence page of the first n_claims claims."""
    claims = claims[:n_claims]
    needed = set().union(*(evidence_pages(c) for c in claims))
    keep = corpus["page_id"].isin(needed)
    rest = corpus.index[~keep].tolist()
    random.Random(seed).shuffle(rest)
    extra = rest[: max(n_docs - int(keep.sum()), 0)]
    sample = corpus[keep | corpus.index.isin(extra)].reset_index(drop=True)
    return sample, claims


# --------------------------------------------------------------------------- main
def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--config", default="configs/bm25.yaml")
    parser.add_argument("--quick", action="store_true", help="rank_bm25 on a 50k-sentence sample, validation only")
    parser.add_argument("--rebuild-index", action="store_true")
    args = parser.parse_args(argv)

    cfg = load_config(args.config)
    if args.quick:
        cfg["run_name"] += "_quick"
        cfg["bm25"]["backend"] = "rank_bm25"
    set_seed(cfg["seed"])
    run_dir, log = start_run(cfg)
    bc = cfg["bm25"]

    corpus_dir, splits_dir = find_corpus_dir(cfg), find_splits_dir(cfg)
    log.info(f"Corpus: {corpus_dir} | splits: {splits_dir}")
    corpus = load_corpus(corpus_dir)
    val = load_claims(splits_dir, "val")
    log.info(f"Corpus sentences: {len(corpus):,} | val claims: {len(val):,}")

    if bc["backend"] == "rank_bm25":
        q = cfg["quick_test"]
        corpus, val = quick_sample(corpus, val, q["n_docs"], q["n_claims"], cfg["seed"])
        log.info(f"Quick test: {len(corpus):,} sentences, {len(val)} val claims")
        texts = [doc_text(p, t, sep=" ") for p, t in zip(corpus["page_id"], corpus["text"])]
        searcher = RankBM25([make_docid(p, s) for p, s in zip(corpus["page_id"], corpus["sent_id"])], texts)
    else:
        searcher = LuceneBM25(build_lucene_index(corpus, run_dir, bc["threads"], log, args.rebuild_index))

    # ---- tune k1, b on validation
    level, metric = bc["tune_metric"]
    grid = []
    for k1 in bc["grid"]["k1"]:
        for b in bc["grid"]["b"]:
            searcher.set_bm25(k1, b)
            rep = evaluate_run(retrieve(searcher, val, bc["top_k"], bc["threads"]), val)
            grid.append({"k1": k1, "b": b, **{f"{lv}_{m}": v for lv in ("page", "sentence") for m, v in rep[lv].items()}})
            log.info(f"k1={k1:<4} b={b:<4} val {level} {metric} = {rep[level][metric]:.4f}  (page recall@10 {rep['page']['recall@10']:.4f})")
    best = max(grid, key=lambda r: r[f"{level}_{metric}"])
    k1, b = best["k1"], best["b"]
    log.info(f"Best on validation: k1={k1}, b={b} ({level} {metric} = {best[f'{level}_{metric}']:.4f})")
    searcher.set_bm25(k1, b)

    metrics = {"backend": bc["backend"], "best_k1": k1, "best_b": b, "tuning_grid": grid, "corpus_sentences": len(corpus)}
    val_run = retrieve(searcher, val, bc["top_k"], bc["threads"])
    write_run(val_run, run_dir / "run_val.jsonl")
    metrics["val"] = evaluate_run(val_run, val)

    if bc["backend"] == "rank_bm25":
        log.info("Quick test done (validation only):\n" + format_report(metrics["val"]))
        save_json(metrics, run_dir / "metrics.json")
        return 0

    # ---- test
    split = test_split_name(cfg)
    test = load_claims(splits_dir, split)
    t0 = time.time()
    test_run = retrieve(searcher, test, bc["top_k"], bc["threads"])
    batch_ms = 1000 * (time.time() - t0) / len(test)
    write_run(test_run, run_dir / "run_test.jsonl")
    ms = time_per_query(lambda qq: searcher.search(qq, bc["top_k"]), [c["claim"] for c in test], bc["latency_queries"])
    metrics["test_split"] = split
    metrics["test"] = evaluate_run(test_run, test)
    metrics["ms_per_query"] = round(ms, 2)
    metrics["batch_ms_per_query"] = round(batch_ms, 2)
    log.info(f"TEST ({split}, {metrics['test']['n_claims']:,} S/R claims scored, {len(test):,} retrieved):\n"
             + format_report(metrics["test"], {"ms/query": metrics["ms_per_query"], f"batch ms/query ({bc['threads']} threads)": metrics["batch_ms_per_query"]}))

    # ---- train run for Prompt 8 hard negatives
    train = load_claims(splits_dir, "train")
    train_run = retrieve(searcher, train, bc["train_top_k"], bc["threads"])
    write_run(train_run, run_dir / "run_train.jsonl")
    log.info(f"Train run: {len(train_run):,} claims x top {bc['train_top_k']}")

    example = test[0]
    log.info(f"Example: [{example['label']}] {example['claim']}")
    for p, s, sc in test_run[example["id"]][:3]:
        log.info(f"   {sc:7.3f}  {p}:{s}")
    save_json(metrics, run_dir / "metrics.json")
    save_json({"k1": k1, "b": b}, run_dir / "best_params.json")
    log.info(f"Outputs written to {run_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
