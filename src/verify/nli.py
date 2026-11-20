"""Prompt 12 (M4): NLI verifier and aggregation.

Run (GPU session, after Prompts 9 and 11):
    python -m src.verify.nli --config configs/nli.yaml

- Models: cross-encoder/nli-deberta-v3-base (main) and roberta-large-mnli (comparison, on the 5k test subset).
  The column of each label is read from model.config.id2label, and a two-pair sanity check asserts that an
  obvious entailment comes out as E and an obvious contradiction as C.
- For each claim and each of its evidence sentences: premise = "page title: sentence", hypothesis = claim.
  The three probabilities (and logits, for calibration in Prompt 13) are kept.
- Aggregation rule: if max p(contradiction) > theta_c -> REFUTES; elif max p(entailment) > theta_e -> SUPPORTS;
  else NOT ENOUGH INFO. theta_c and theta_e are tuned on validation for macro-F1, separately for each
  model and evidence source, then applied once to test.
- Alternative: one premise made of all evidence sentences joined together (lets the model combine facts
  from two sentences, which multi-hop claims need); same rule and tuning.
- Oracle: annotated evidence instead of retrieved evidence (NEI claims get sentences sampled from their
  closest-matching page), the verifier's upper bound.
Evidence sources (configs/nli.yaml): "retrieved" = top 5 of the chosen retriever, "chain" = the two-hop chain,
"oracle" = annotated evidence.
Outputs in outputs/verify/: probs_<model>_<source>_<split>.parquet (one row per claim-sentence pair),
metrics.json, table.md. Logits are cached in outputs/nli_cache/ and reused by Prompts 13-14.
"""
from __future__ import annotations

import argparse
import sys

from src.retrieval.common import find_corpus_dir, find_splits_dir, load_claims, load_corpus, sentence_lookup, test_split_name
from src.utils import load_config, save_json, set_seed, start_run
from src.verify.evidence import (
    LABEL_OF,
    NLICache,
    NLIScorer,
    aggregate,
    classification_report,
    concat_table,
    gold_indices,
    load_evidence,
    page_index,
    rule_labels,
    save_probs,
    sentence_table,
    slug,
    tune_thresholds,
)


def evaluate_source(tables: dict, claims: dict, grid: list[float]) -> dict:
    """Tune the two thresholds on val, apply to test. tables/claims keyed by 'val' / 'test'."""
    agg = {sp: aggregate(tables[sp], claims[sp]) for sp in ("val", "test")}
    gold = {sp: gold_indices(claims[sp]) for sp in ("val", "test")}
    te, tc, f_val = tune_thresholds(agg["val"]["max_pe"], agg["val"]["max_pc"], gold["val"], grid)
    pred = rule_labels(agg["test"]["max_pe"], agg["test"]["max_pc"], te, tc)
    return {"theta_e": te, "theta_c": tc, "val_macro_f1": f_val, "test": classification_report(gold["test"], pred),
            "mean_sentences": float(agg["test"]["n_sentences"].mean())}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--config", default="configs/nli.yaml")
    args = parser.parse_args(argv)

    cfg = load_config(args.config)
    set_seed(cfg["seed"])
    run_dir, log = start_run(cfg)
    nc = cfg["nli"]
    grid = nc["threshold_grid"]

    corpus = load_corpus(find_corpus_dir(cfg))
    texts, pages = sentence_lookup(corpus), page_index(corpus)
    splits_dir = find_splits_dir(cfg)
    split = test_split_name(cfg)
    claims = {"val": load_claims(splits_dir, "val"), "test": load_claims(splits_dir, split)}
    subset_ids = {c["id"] for c in load_claims(splits_dir, nc["compare_split"])}

    evidence = {}
    for src, spec in nc["sources"].items():
        for sp in ("val", "test"):
            evid, _, desc = load_evidence(cfg, spec, sp, claims[sp], texts, pages)
            evidence[(src, sp)] = evid
        log.info(f"Evidence source '{src}' = {desc}")

    results, lines = {}, []
    for role, model_name in (("main", nc["main_model"]), ("compare", nc["compare_model"])):
        if not model_name:
            continue
        scorer = NLIScorer(model_name, nc["fp16"], nc["max_length"], nc["batch_size"], log)
        scorer.sanity_check()
        cache = NLICache(cfg, scorer, log)
        # The comparison model runs on the class-balanced 5k subset only (roberta-large is about 3x slower).
        cl = claims if role == "main" else {"val": claims["val"], "test": [c for c in claims["test"] if c["id"] in subset_ids]}
        test_name = split if role == "main" else nc["compare_split"]
        sources = list(nc["sources"]) if role == "main" else nc["compare_sources"]
        for src in sources:
            tables = {sp: sentence_table(evidence[(src, sp)], cl[sp], texts, cache) for sp in ("val", "test")}
            for sp in ("val", "test"):
                save_probs(tables[sp], run_dir / f"probs_{slug(model_name)}_{src}_{sp if sp == 'val' else test_name}.parquet")
            res = evaluate_source(tables, cl, grid)
            results[f"{role}|{src}|per_sentence"] = {"model": model_name, "source": src, "mode": "per sentence", "test_split": test_name, **res}
            if role == "main" and src in nc["concat_sources"]:
                ctab = {sp: concat_table(evidence[(src, sp)], cl[sp], texts, cache) for sp in ("val", "test")}
                results[f"{role}|{src}|concatenated"] = {"model": model_name, "source": src, "mode": "concatenated premise",
                                                         "test_split": test_name, **evaluate_source(ctab, cl, grid)}
        # Same claims as the comparison model, for a like-for-like row.
        if role == "main" and nc["compare_model"]:
            sub = {"val": claims["val"], "test": [c for c in claims["test"] if c["id"] in subset_ids]}
            for src in nc["compare_sources"]:
                tables = {sp: sentence_table(evidence[(src, sp)], sub[sp], texts, cache) for sp in ("val", "test")}
                results[f"main_on_subset|{src}|per_sentence"] = {"model": model_name, "source": src, "mode": "per sentence",
                                                                 "test_split": nc["compare_split"], **evaluate_source(tables, sub, grid)}
        del scorer, cache

    header = "| Model | Evidence | Mode | Test split | theta_e | theta_c | Accuracy | Macro-F1 | F1 S / R / NEI |"
    lines = [header, "|---|---|---|---|---|---|---|---|---|"]
    for r in results.values():
        t = r["test"]
        f = t["per_class_f1"]
        lines.append(f"| {r['model'].split('/')[-1]} | {r['source']} | {r['mode']} | {r['test_split']} | {r['theta_e']:.2f} | {r['theta_c']:.2f} | "
                     f"{t['accuracy']:.4f} | {t['macro_f1']:.4f} | {f[LABEL_OF[0]]:.3f} / {f[LABEL_OF[1]]:.3f} / {f[LABEL_OF[2]]:.3f} |")
    table = "\n".join(lines)
    log.info("\n" + table)
    main_rows = {k: v for k, v in results.items() if k.startswith("main|")}
    for k, r in main_rows.items():
        log.info(f"{k}: confusion (rows gold S/R/NEI, columns predicted) {r['test']['confusion']}")
    (run_dir / "table.md").write_text("# NLI verifier\n\n" + table + "\n", encoding="utf-8")
    save_json({"results": results, "threshold_grid": grid, "test_split": split}, run_dir / "metrics.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
