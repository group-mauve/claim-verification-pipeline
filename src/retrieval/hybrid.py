"""Prompt 9 (M2): hybrid retrieval, fusing the BM25 and fine-tuned dense top-100 lists.

Run (CPU is enough; needs the run files of Prompts 5 and 8):
    python -m src.retrieval.hybrid --config configs/hybrid.yaml

Two fusion methods
- Reciprocal Rank Fusion (our design choice): RRF(d) = sum over the lists that contain d of 1 / (60 + rank).
  Only ranks are used, so the very different score scales of BM25 (0-40) and cosine (0-1) do not matter.
  A sentence ranked high by both lists wins; one ranked high by a single list still gets credit.
- Weighted score fusion: alpha * minmax(dense) + (1 - alpha) * minmax(bm25), scores min-max scaled per claim
  over that list's top 100 (a sentence missing from a list gets 0 from it). alpha is tuned on validation
  over {0.2, ..., 0.8} by sentence Recall@10.
Both are reported on test with Recall@{5, 10, 20, 50}, MRR and nDCG@10. RRF is written as run_test.jsonl
(the design choice); the weighted fusion run is saved next to it.

Seed noise: if the second-seed fine-tuned run (dense_ft_seed43) exists, RRF is also computed with it.
The difference between the two is the "noise between two seeds" used by Prompt 10's decision rule.
Outputs in outputs/hybrid/: run_val.jsonl, run_test.jsonl, run_*_weighted.jsonl, metrics.json, table.md.
"""
from __future__ import annotations

import argparse
import json
import sys
import time

from src.retrieval.common import (
    evaluate_run,
    find_output,
    find_splits_dir,
    format_report,
    load_claims,
    read_run,
    require_output,
    test_split_name,
    write_run,
)
from src.utils import load_config, save_json, set_seed, start_run


# --------------------------------------------------------------------------- fusion
def rrf_scores(ranked_lists: list[list], k: int = 60) -> list[tuple]:
    """Reciprocal Rank Fusion of ranked lists of hashable items (ranks start at 1).

    Returns [(item, score)] best first; ties keep the order in which items were first seen.
    """
    scores: dict = {}
    for ranked in ranked_lists:
        for rank, item in enumerate(ranked, start=1):
            scores[item] = scores.get(item, 0.0) + 1.0 / (k + rank)
    return sorted(scores.items(), key=lambda kv: -kv[1])  # sorted() is stable


def minmax(scores: dict) -> dict:
    """Scale scores to [0, 1] (min -> 0, max -> 1). If all scores are equal, every item gets 1."""
    if not scores:
        return {}
    lo, hi = min(scores.values()), max(scores.values())
    if hi == lo:
        return {d: 1.0 for d in scores}
    return {d: (s - lo) / (hi - lo) for d, s in scores.items()}


def weighted_scores(dense: dict, sparse: dict, alpha: float) -> list[tuple]:
    """alpha * minmax(dense) + (1 - alpha) * minmax(sparse) over the union; best first."""
    nd, ns = minmax(dense), minmax(sparse)
    items = list(dict.fromkeys([*dense, *sparse]))
    fused = {d: alpha * nd.get(d, 0.0) + (1 - alpha) * ns.get(d, 0.0) for d in items}
    return sorted(fused.items(), key=lambda kv: -kv[1])


def fuse_runs(bm25: dict, dense: dict, method: str, top_k: int, rrf_k: int = 60, alpha: float = 0.5) -> dict:
    """Fuse two run dicts (claim id -> [(page, sent_id, score)]) claim by claim."""
    out = {}
    for cid in dense.keys() | bm25.keys():
        b, d = bm25.get(cid, []), dense.get(cid, [])
        if method == "rrf":
            fused = rrf_scores([[(p, s) for p, s, _ in b], [(p, s) for p, s, _ in d]], rrf_k)
        else:
            fused = weighted_scores({(p, s): sc for p, s, sc in d}, {(p, s): sc for p, s, sc in b}, alpha)
        out[cid] = [(p, s, sc) for (p, s), sc in fused[:top_k]]
    return out


# --------------------------------------------------------------------------- main
def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--config", default="configs/hybrid.yaml")
    args = parser.parse_args(argv)

    cfg = load_config(args.config)
    set_seed(cfg["seed"])
    run_dir, log = start_run(cfg)
    hc = cfg["hybrid"]
    ks = tuple(hc["ks"])

    splits_dir = find_splits_dir(cfg)
    split = test_split_name(cfg)
    claims = {"val": load_claims(splits_dir, "val"), "test": load_claims(splits_dir, split)}
    runs = {}
    for name in (hc["sparse_run"], hc["dense_run"]):
        for sp in ("val", "test"):
            path = require_output(cfg, name, f"run_{sp}.jsonl")
            runs[(name, sp)] = read_run(path)
            log.info(f"{name} {sp}: {len(runs[(name, sp)]):,} claims from {path}")
    sparse = {sp: runs[(hc["sparse_run"], sp)] for sp in ("val", "test")}
    dense = {sp: runs[(hc["dense_run"], sp)] for sp in ("val", "test")}

    # ---- RRF
    t0 = time.perf_counter()
    rrf = {sp: fuse_runs(sparse[sp], dense[sp], "rrf", hc["top_k"], hc["rrf_k"]) for sp in ("val", "test")}
    fuse_ms = 1000 * (time.perf_counter() - t0) / (len(claims["val"]) + len(claims["test"]))

    # ---- weighted fusion: tune alpha on validation
    level, metric = hc["tune_metric"]
    grid = []
    for alpha in hc["alphas"]:
        rep = evaluate_run(fuse_runs(sparse["val"], dense["val"], "weighted", hc["top_k"], alpha=alpha), claims["val"], ks=ks)
        grid.append({"alpha": alpha, "val": rep})
        log.info(f"alpha={alpha:.1f}  val {level} {metric} = {rep[level][metric]:.4f}  (sentence MRR {rep['sentence']['mrr']:.4f})")
    best = max(grid, key=lambda g: g["val"][level][metric])
    alpha = best["alpha"]
    log.info(f"Best alpha on validation: {alpha} ({level} {metric} {best['val'][level][metric]:.4f})")
    weighted = {sp: fuse_runs(sparse[sp], dense[sp], "weighted", hc["top_k"], alpha=alpha) for sp in ("val", "test")}

    # ---- evaluate every system on val and test
    systems = {"BM25": sparse, "Dense (fine-tuned)": dense, "Hybrid RRF": rrf, f"Hybrid weighted (alpha={alpha})": weighted}
    results = {name: {sp: evaluate_run(run[sp], claims[sp], ks=ks) for sp in ("val", "test")} for name, run in systems.items()}
    for name, res in results.items():
        log.info(f"{name} TEST:\n" + format_report(res["test"]))

    for sp in ("val", "test"):
        write_run(rrf[sp], run_dir / f"run_{sp}.jsonl")
        write_run(weighted[sp], run_dir / f"run_{sp}_weighted.jsonl")

    # ---- latency: hybrid = BM25 query + dense query + fusion
    lat = {}
    for key, name in (("bm25", hc["sparse_run"]), ("dense", hc["dense_run"])):
        mp = find_output(cfg, name, "metrics.json")
        lat[key] = json.loads(mp.read_text(encoding="utf-8")).get("ms_per_query") if mp else None
    lat["fusion"] = round(fuse_ms, 3)
    lat["hybrid_total"] = round(sum(v for v in lat.values() if v is not None), 2)
    log.info(f"ms/query: BM25 {lat['bm25']} + dense {lat['dense']} + fusion {lat['fusion']} = {lat['hybrid_total']} "
             "(the two retrievers could also run in parallel)")

    # ---- seed noise (second fine-tuning seed)
    noise = None
    noise_name = hc.get("noise_dense_run")
    if noise_name and find_output(cfg, noise_name, "run_test.jsonl"):
        noise = {}
        for sp in ("val", "test"):
            other = read_run(require_output(cfg, noise_name, f"run_{sp}.jsonl"))
            rep_other = evaluate_run(fuse_runs(sparse[sp], other, "rrf", hc["top_k"], hc["rrf_k"]), claims[sp], ks=ks)
            rep_main = results["Hybrid RRF"][sp]
            noise[sp] = {f"{lv}_{m}": abs(rep_main[lv][m] - rep_other[lv][m]) for lv in ("page", "sentence") for m in ("mrr", "ndcg@10", "recall@10")}
        log.info(f"Seed noise (|RRF with seed 42 - RRF with {noise_name}|): val sentence MRR {noise['val']['sentence_mrr']:.4f}, "
                 f"test sentence MRR {noise['test']['sentence_mrr']:.4f}")
    else:
        log.warning(f"No {noise_name} run found; seed noise not measured (Prompt 10 then uses its configured default).")

    # ---- table
    header = ["System"] + [f"R@{k}" for k in ks] + ["MRR", "nDCG@10"]
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    for name, res in results.items():
        for lv in ("sentence", "page"):
            r = res["test"][lv]
            lines.append(f"| {name} ({lv}) | " + " | ".join(f"{r[f'recall@{k}']:.4f}" for k in ks) + f" | {r['mrr']:.4f} | {r['ndcg@10']:.4f} |")
    table = "\n".join(lines)
    (run_dir / "table.md").write_text(f"# Hybrid retrieval, test ({split})\n\n{table}\n", encoding="utf-8")
    log.info("\n" + table)

    better = "RRF" if results["Hybrid RRF"]["val"][level][metric] >= best["val"][level][metric] else "weighted fusion"
    log.info(f"On validation {better} has the higher {level} {metric}. run_test.jsonl holds RRF (the design choice).")
    save_json({"test_split": split, "best_alpha": alpha, "alpha_grid": grid, "results": results, "ms_per_query": lat,
               "seed_noise": noise, "better_on_val": better, "primary": "rrf"}, run_dir / "metrics.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
