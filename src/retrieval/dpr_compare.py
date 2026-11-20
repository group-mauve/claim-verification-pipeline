"""Prompt 7 (M2): compare the DPR checkpoint with MiniLM on the 5k test subset.

Run (GPU session, after Prompts 5 and 6):
    python -m src.retrieval.dpr_compare --config configs/dpr_compare.yaml

Setup
- Claims: test_5k (class-balanced dev subset). Only its S/R claims are scored.
- Candidate pool: all sentences of (a) the evidence pages of those claims and (b) every page in their
  BM25 top-100. BM25 chooses the pool, so neither dense model picks its own candidates, and encoding
  this pool with BERT-base DPR is cheaper than the whole corpus.
- DPR: facebook/dpr-ctx_encoder-single-nq-base for sentences (input: title + sentence as a text pair,
  the way DPR was trained) and dpr-question_encoder-single-nq-base for claims. 768-d, raw dot product,
  no normalisation (DPR was trained with an unnormalised dot product).
- MiniLM: the Prompt 6 embeddings of the same sentences (384-d, normalised, cosine).
- Both search the same pool exactly (FAISS IndexFlatIP). Reported: Recall@10, MRR (page and sentence),
  ms/query (encode one claim + search) and index memory (for the pool and projected to the full corpus).
Outputs in outputs/dpr_compare/: metrics.json, comparison.md, run_dpr.jsonl, run_minilm.jsonl.
"""
from __future__ import annotations

import argparse
import sys
import time

import numpy as np

from src.retrieval.common import (
    evaluate_run,
    find_corpus_dir,
    find_output,
    find_splits_dir,
    format_report,
    hits_from_indices,
    load_claims,
    load_corpus,
    read_run,
    time_per_query,
    write_run,
)
from src.retrieval.dense import FlatIndex, encode_in_shards, load_bi_encoder, st_encoder
from src.utils import clean_text, evidence_pages, load_config, readable_title, save_json, set_seed, start_run


def dpr_encoders(cfg: dict):
    """Return (encode_contexts(titles, texts), encode_questions(texts), device) for the DPR checkpoints."""
    import torch
    from transformers import (
        DPRContextEncoder,
        DPRContextEncoderTokenizerFast,
        DPRQuestionEncoder,
        DPRQuestionEncoderTokenizerFast,
    )

    dc = cfg["dpr"]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    ctx_tok = DPRContextEncoderTokenizerFast.from_pretrained(dc["ctx_model"])
    ctx_model = DPRContextEncoder.from_pretrained(dc["ctx_model"]).to(device).eval()
    q_tok = DPRQuestionEncoderTokenizerFast.from_pretrained(dc["question_model"])
    q_model = DPRQuestionEncoder.from_pretrained(dc["question_model"]).to(device).eval()
    use_amp = dc["fp16"] and device == "cuda"

    @torch.no_grad()
    def run(model, enc_batch):
        enc_batch = {k: v.to(device) for k, v in enc_batch.items()}
        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=use_amp):
            return model(**enc_batch).pooler_output.float().cpu().numpy()

    def encode_contexts(pairs: list[str]) -> np.ndarray:
        # pairs are "title\tsentence" strings so they fit encode_in_shards' list-of-str interface
        out = []
        for i in range(0, len(pairs), dc["batch_size"]):
            titles, texts = zip(*(p.split("\t", 1) for p in pairs[i : i + dc["batch_size"]]))
            enc = ctx_tok(list(titles), list(texts), truncation=True, max_length=dc["max_length"], padding=True, return_tensors="pt")
            out.append(run(ctx_model, enc))
        return np.vstack(out).astype(np.float32)

    def encode_questions(texts: list[str]) -> np.ndarray:
        out = []
        for i in range(0, len(texts), dc["batch_size"]):
            enc = q_tok(texts[i : i + dc["batch_size"]], truncation=True, max_length=64, padding=True, return_tensors="pt")
            out.append(run(q_model, enc))
        return np.vstack(out).astype(np.float32)

    return encode_contexts, encode_questions, device


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--config", default="configs/dpr_compare.yaml")
    args = parser.parse_args(argv)

    cfg = load_config(args.config)
    set_seed(cfg["seed"])
    run_dir, log = start_run(cfg)
    k = cfg["top_k"]

    corpus = load_corpus(find_corpus_dir(cfg))
    claims = load_claims(find_splits_dir(cfg), "test_5k")
    ids = {c["id"] for c in claims}

    # ---- candidate pool
    pool_path = find_output(cfg, cfg["pool_from"], "run_test.jsonl")
    if pool_path is None:
        raise FileNotFoundError(f"{cfg['pool_from']}/run_test.jsonl not found; add the BM25 notebook output as an input.")
    pool_run = {cid: hits for cid, hits in read_run(pool_path).items() if cid in ids}
    missing = ids - set(pool_run)
    if missing:
        raise ValueError(f"{len(missing)} test_5k claims are not in {pool_path}; run BM25 with test_subset: full")
    pool_pages = set().union(*(evidence_pages(c) for c in claims)) | {p for hits in pool_run.values() for p, _, _ in hits}
    rows = np.flatnonzero(corpus["page_id"].isin(pool_pages).to_numpy())
    pool = corpus.iloc[rows].reset_index(drop=True)
    log.info(f"Pool: {len(pool_pages):,} pages, {len(pool):,} sentences ({100 * len(pool) / len(corpus):.1f}% of the corpus)")

    # ---- MiniLM: reuse Prompt 6 vectors
    emb_dir = find_output(cfg, cfg["minilm"]["run_name"], "embeddings/meta.json")
    if emb_dir is None:
        raise FileNotFoundError("dense_offshelf embeddings not found; run Prompt 6 first.")
    emb_dir = emb_dir.parent
    mini_all = np.vstack([np.load(p) for p in sorted(emb_dir.glob("emb_*.npy"))])
    assert len(mini_all) == len(corpus), f"MiniLM embeddings ({len(mini_all)}) do not match the corpus ({len(corpus)})"
    mini_index = FlatIndex(mini_all[rows])
    del mini_all
    mini_model, _ = load_bi_encoder(cfg["minilm"]["model_name"], True)
    mini_encode = st_encoder(mini_model, 256)

    # ---- DPR: encode the pool (sharded + cached)
    enc_ctx, enc_q, device = dpr_encoders(cfg)
    pairs = [f"{readable_title(p)}\t{clean_text(t)}" for p, t in zip(pool["page_id"], pool["text"])]
    t0 = time.time()
    dpr_vectors = encode_in_shards(pairs, enc_ctx, run_dir / "dpr_embeddings", cfg["dpr"]["shard_size"],
                                   {"model": cfg["dpr"]["ctx_model"], "pool_from": str(pool_path.name)}, log)
    log.info(f"DPR pool encoding: {dpr_vectors.shape} in {time.time() - t0:.0f}s on {device}")
    dpr_index = FlatIndex(dpr_vectors)

    # ---- retrieve and evaluate
    queries = [clean_text(c["claim"]) for c in claims]
    results, runs = {}, {}
    for name, index, encode, dim in (("minilm", mini_index, mini_encode, 384), ("dpr", dpr_index, enc_q, 768)):
        scores, idx = index.search(encode(queries), k)
        run = {c["id"]: h for c, h in zip(claims, hits_from_indices(pool, scores, idx))}
        runs[name] = run
        write_run(run, run_dir / f"run_{name}.jsonl")
        rep = evaluate_run(run, claims, ks=(5, 10, 20))
        ms = time_per_query(lambda q: index.search(encode([q]), k), queries, cfg["latency_queries"])
        results[name] = {
            "dim": dim, "report": rep, "ms_per_query": round(ms, 2),
            "pool_index_gb": round(len(pool) * dim * 4 / 1024**3, 3),
            "full_corpus_index_gb": round(len(corpus) * dim * 4 / 1024**3, 3),
        }
        log.info(f"{name.upper()}:\n" + format_report(rep, {"ms/query": results[name]["ms_per_query"], "pool index GB": results[name]["pool_index_gb"]}))

    m, d = results["minilm"], results["dpr"]
    r10 = lambda r: r["report"]["sentence"]["recall@10"]  # noqa: E731
    mrr = lambda r: r["report"]["sentence"]["mrr"]  # noqa: E731
    # Accuracy: Recall@10 first, MRR breaks ties. Speed is stated only as measured.
    better = "MiniLM" if (r10(m), mrr(m)) >= (r10(d), mrr(d)) else "DPR"
    speed = ("faster" if m["ms_per_query"] < d["ms_per_query"] else "slower" if m["ms_per_query"] > d["ms_per_query"] else "as fast")
    decision = (
        f"On {m['report']['n_claims']:,} S/R claims of the 5k subset, MiniLM reaches sentence Recall@10 {r10(m):.3f} and MRR {mrr(m):.3f}, "
        f"against {r10(d):.3f} and {mrr(d):.3f} for the DPR checkpoint, so {better} retrieves better off the shelf. "
        f"MiniLM is {speed} per query ({m['ms_per_query']:.1f} ms against {d['ms_per_query']:.1f} ms) and its full-corpus index needs "
        f"{m['full_corpus_index_gb']} GB against {d['full_corpus_index_gb']} GB, because its vectors have 384 instead of 768 dimensions. "
        + ("We keep MiniLM: it is at least as accurate with half the index size, and it is the model we fine-tune in Prompt 8."
           if better == "MiniLM" else
           "We still fine-tune MiniLM in Prompt 8 for its half-size index; the fine-tuned model must close this gap, and DPR stays as the reference.")
    )
    log.info("DECISION: " + decision)

    table = ["| Encoder | Dim | Sentence R@10 | Sentence MRR | Page R@10 | Page MRR | ms/query | Index GB (pool) | Index GB (full corpus) |",
             "|---|---|---|---|---|---|---|---|---|"]
    for name, r in (("MiniLM (all-MiniLM-L6-v2)", m), ("DPR (single-nq-base)", d)):
        s, p = r["report"]["sentence"], r["report"]["page"]
        table.append(f"| {name} | {r['dim']} | {s['recall@10']:.4f} | {s['mrr']:.4f} | {p['recall@10']:.4f} | {p['mrr']:.4f} | "
                     f"{r['ms_per_query']} | {r['pool_index_gb']} | {r['full_corpus_index_gb']} |")
    md = (f"# DPR vs MiniLM on test_5k\n\nPool: {len(pool_pages):,} pages / {len(pool):,} sentences "
          f"(evidence pages + BM25 top-100 pages of the 5k claims).\n\n" + "\n".join(table) + f"\n\n{decision}\n")
    (run_dir / "comparison.md").write_text(md, encoding="utf-8")
    save_json({"pool_pages": len(pool_pages), "pool_sentences": len(pool), "results": results, "decision": decision}, run_dir / "metrics.json")
    log.info(f"\n{md}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
