"""Prompt 10 (M2, optional): rerank the hybrid top 50 with a cross-encoder.

Run (GPU session, after Prompt 9):
    python -m src.retrieval.rerank --config configs/rerank.yaml

- Model: cross-encoder/ms-marco-MiniLM-L-6-v2. It reads (claim, "title: sentence") together and outputs
  one relevance score; the top 50 hybrid sentences of each claim are re-sorted by that score.
- Reported with and without reranking, on val and test: MRR, nDCG@10, Recall@K, ms per query.
- Decision rule: keep the reranker only if sentence MRR on validation improves by more than the noise
  between two seeds (the MRR difference between hybrid runs built from two fine-tuning seeds, measured in
  Prompt 9). The decision is made on validation so the test set does not choose the system. The downstream
  FEVER-score check is the extra "Hybrid + rerank + NLI" row of the Prompt 14 table.
Outputs in outputs/rerank/: run_val.jsonl, run_test.jsonl, metrics.json (with "keep": true/false).
Later steps use the reranked run only when keep is true (run name "auto" in their configs).
"""
from __future__ import annotations

import argparse
import json
import sys
import time

import numpy as np

from src.retrieval.common import (
    doc_text,
    evaluate_run,
    find_corpus_dir,
    find_output,
    find_splits_dir,
    format_report,
    load_claims,
    load_corpus,
    read_run,
    require_output,
    sentence_lookup,
    test_split_name,
    write_run,
)
from src.utils import clean_text, load_config, save_json, set_seed, start_run


def rerank_run(run: dict, claims: list[dict], texts: dict, score_fn, top_n: int) -> dict:
    """Re-sort the top_n hits of each claim by score_fn(claims_texts, sentence_texts) -> scores."""
    first, second, owners = [], [], []
    for c in claims:
        for p, s, _ in run.get(c["id"], [])[:top_n]:
            first.append(clean_text(c["claim"]))
            second.append(doc_text(p, texts.get((p, s), "")))
            owners.append((c["id"], p, s))
    scores = score_fn(first, second) if first else np.zeros(0)
    out: dict = {c["id"]: [] for c in claims}
    for (cid, p, s), sc in zip(owners, scores):
        out[cid].append((p, s, float(sc)))
    return {cid: sorted(h, key=lambda x: -x[2]) for cid, h in out.items()}


def cross_encoder_scorer(cfg: dict):
    from src.cross_encoder import PairModel

    rc = cfg["rerank"]
    model = PairModel(rc["model"], fp16=rc["fp16"], max_length=rc["max_length"], batch_size=rc["batch_size"])

    def score(first, second):
        return model.logits(first, second, progress=len(first) > 50_000)[:, 0]

    return score, model.device


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--config", default="configs/rerank.yaml")
    args = parser.parse_args(argv)

    cfg = load_config(args.config)
    set_seed(cfg["seed"])
    run_dir, log = start_run(cfg)
    rc = cfg["rerank"]
    ks = tuple(rc["ks"])

    texts = sentence_lookup(load_corpus(find_corpus_dir(cfg)))
    splits_dir = find_splits_dir(cfg)
    split = test_split_name(cfg)
    score_fn, device = cross_encoder_scorer(cfg)
    log.info(f"Reranker {rc['model']} on {device}; reranking the top {rc['top_n']} of {rc['input_run']}")

    results, timing = {}, {}
    for sp, name in (("val", "val"), ("test", split)):
        claims = load_claims(splits_dir, name)
        base = read_run(require_output(cfg, rc["input_run"], f"run_{sp}.jsonl"))
        base_top = {cid: hits[: rc["top_n"]] for cid, hits in base.items()}
        t0 = time.time()
        reranked = rerank_run(base, claims, texts, score_fn, rc["top_n"])
        timing[sp] = round(time.time() - t0, 1)
        write_run(reranked, run_dir / f"run_{sp}.jsonl")
        results[sp] = {"without": evaluate_run(base_top, claims, ks=ks), "with": evaluate_run(reranked, claims, ks=ks)}
        log.info(f"{sp.upper()} without rerank:\n" + format_report(results[sp]["without"]))
        log.info(f"{sp.upper()} with rerank ({timing[sp]}s for {len(claims):,} claims):\n" + format_report(results[sp]["with"]))
        if sp == "test":
            test_claims, test_base = claims, base

    # ---- latency of reranking one claim (50 cross-encoder pairs in one batch)
    sample = test_claims[: rc["latency_queries"]]
    rerank_run({sample[0]["id"]: test_base[sample[0]["id"]]}, sample[:1], texts, score_fn, rc["top_n"])  # warm-up
    t0 = time.perf_counter()
    for c in sample:
        rerank_run({c["id"]: test_base.get(c["id"], [])}, [c], texts, score_fn, rc["top_n"])
    rerank_ms = 1000 * (time.perf_counter() - t0) / len(sample)
    hm = find_output(cfg, rc["input_run"], "metrics.json")
    hybrid_ms = json.loads(hm.read_text(encoding="utf-8"))["ms_per_query"]["hybrid_total"] if hm else None
    lat = {"hybrid": hybrid_ms, "rerank_only": round(rerank_ms, 2),
           "hybrid_plus_rerank": round(hybrid_ms + rerank_ms, 2) if hybrid_ms is not None else None}
    log.info(f"ms/query: hybrid {lat['hybrid']} | rerank of {rc['top_n']} sentences {lat['rerank_only']} | total {lat['hybrid_plus_rerank']}")

    # ---- decision
    noise_src = "Prompt 9 seed noise"
    noise = None
    if hm:
        sn = json.loads(hm.read_text(encoding="utf-8")).get("seed_noise")
        noise = sn["val"]["sentence_mrr"] if sn else None
    if noise is None:
        noise, noise_src = rc["default_noise"], "configured default (second seed not run)"
    gain = {sp: results[sp]["with"]["sentence"]["mrr"] - results[sp]["without"]["sentence"]["mrr"] for sp in ("val", "test")}
    keep = gain["val"] > noise
    decision = (f"Reranking changes validation sentence MRR by {gain['val']:+.4f} (test {gain['test']:+.4f}); the noise between two "
                f"seeds is {noise:.4f} ({noise_src}). "
                + (f"The gain is larger than the noise, so we KEEP the reranker, at {lat['rerank_only']} ms extra per claim."
                   if keep else "The gain is not larger than the noise, so we DROP the reranker and use the hybrid ranking."))
    log.info("DECISION: " + decision)

    lines = ["| Split | System | Sentence MRR | Sentence nDCG@10 | Sentence R@10 | Page MRR |", "|---|---|---|---|---|---|"]
    for sp in ("val", "test"):
        for k in ("without", "with"):
            r = results[sp][k]
            lines.append(f"| {sp} | {k} rerank | {r['sentence']['mrr']:.4f} | {r['sentence']['ndcg@10']:.4f} | "
                         f"{r['sentence']['recall@10']:.4f} | {r['page']['mrr']:.4f} |")
    (run_dir / "table.md").write_text("# Cross-encoder rerank\n\n" + "\n".join(lines) + f"\n\n{decision}\n", encoding="utf-8")
    log.info("\n" + "\n".join(lines))
    save_json({"model": rc["model"], "top_n": rc["top_n"], "test_split": split, "results": results, "ms_per_query": lat,
               "rerank_seconds": timing, "mrr_gain": gain, "noise": noise, "noise_source": noise_src, "keep": keep,
               "decision": decision}, run_dir / "metrics.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
