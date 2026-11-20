"""Prompt 14 (M1, integrating everyone's modules): end-to-end pipeline, FEVER score, results table.

Run:
    python -m src.pipeline --table verification                    # every row of the verification table
    python -m src.pipeline --config configs/pipeline_hybrid.yaml   # one pipeline

Pipeline per claim
    retriever (bm25 | dense | hybrid, reranked if Prompt 10 kept it) -> top-5 evidence sentences
    -> optional two-hop chain (Prompt 11; still 5 sentences) -> NLI per sentence (Prompt 12)
    -> aggregation rule with theta_e, theta_c tuned on validation
    -> optional temperature scaling + "NEI when unsure" (Prompt 13; T and tau fitted on validation).
Every tuned number is fitted on the validation split with that row's own evidence, then applied once to test.

Predictions are written in the official FEVER format ({"id", "predicted_label", "predicted_evidence":
[[page, sent_id], ...]}) and scored with the official sheffieldnlp scorer (and our own implementation, which
must agree). Gold evidence titles are converted with canonical_page first, because the scorer compares
strings exactly and FEVER mixes two Unicode forms (see README).

Table columns: label accuracy, macro-F1, FEVER score, supporting-fact F1 (predicted evidence vs annotated
evidence sentences, mean over SUPPORTS / REFUTES claims) and ECE (10 bins, confidence of the output label).
Outputs: outputs/pipeline/<row>/predictions_test.jsonl + metrics.json; outputs/tables/verification.csv and .md.
"""
from __future__ import annotations

import argparse
import glob
import os
import re
import sys

import numpy as np
import pandas as pd

from src.eval.metrics import expected_calibration_error, fever_score, mean_supporting_fact_prf
from src.retrieval.common import find_corpus_dir, find_output, find_splits_dir, load_claims, load_corpus, sentence_lookup, test_split_name
from src.utils import canonical_page, evidence_sets, load_config, resolve, save_json, set_seed, start_run, write_jsonl
from src.verify.calibrate import abstain, calibrate, claim_logits, confidence
from src.verify.evidence import (
    LABEL_OF,
    NLICache,
    NLIScorer,
    aggregate,
    gold_indices,
    load_evidence,
    macro_f1_idx,
    page_index,
    rule_labels,
    sentence_table,
    softmax,
    tune_thresholds,
)


# --------------------------------------------------------------------------- official scorer
def official_scorer():
    """fever.scorer.fever_score, or None. Looks for a fever-scorer clone if it is not importable yet."""
    try:
        from fever.scorer import fever_score as official

        return official
    except ImportError:
        pass
    except Exception:  # noqa: BLE001 - e.g. the clone is on the path but a dependency (six) is missing
        return None
    for root in filter(None, [os.environ.get("FEVER_SCORER_DIR"), "/kaggle/working/fever-scorer", str(resolve("fever-scorer"))]):
        hits = glob.glob(os.path.join(root, "**", "fever", "scorer.py"), recursive=True)
        if hits:
            sys.path.insert(0, os.path.dirname(os.path.dirname(hits[0])))
            try:
                from fever.scorer import fever_score as official

                return official
            except Exception:  # noqa: BLE001
                return None
    return None


def canonical_gold_evidence(evidence: list) -> list:
    return [[[a, b, canonical_page(p) if p is not None else None, s] for a, b, p, s in group] for group in evidence]


def score_predictions(preds: list[dict], claims: list[dict], log) -> dict:
    ours = fever_score(preds, claims)
    out = {"fever_score": ours["fever_score"], "label_accuracy": ours["label_accuracy"], "scorer": "ours"}
    official = official_scorer()
    if official is None:
        log.warning("Official FEVER scorer not found (clone sheffieldnlp/fever-scorer); using our implementation.")
        return out
    inst = [{"label": c["label"], "evidence": canonical_gold_evidence(c["evidence"]), **p} for c, p in zip(claims, preds)]
    strict, acc, prec, rec, f1 = official(inst, max_evidence=5)
    assert abs(strict - ours["fever_score"]) < 1e-9 and abs(acc - ours["label_accuracy"]) < 1e-9, \
        f"official scorer ({strict}, {acc}) and ours ({ours}) disagree"
    out.update({"fever_score": strict, "label_accuracy": acc, "scorer": "official (ours agrees)",
                "official_evidence_precision": prec, "official_evidence_recall": rec, "official_evidence_f1": f1})
    return out


# --------------------------------------------------------------------------- one pipeline
def run_pipeline(cfg: dict, shared: dict | None = None) -> dict:
    """Run one configured pipeline end to end; `shared` caches corpus, claims and NLI caches across rows."""
    shared = shared if shared is not None else {}
    set_seed(cfg["seed"])
    run_dir, log = start_run(cfg)
    pc = cfg["pipeline"]
    nc = cfg["nli"]

    if "texts" not in shared:
        corpus = load_corpus(find_corpus_dir(cfg))
        shared["texts"], shared["pages"] = sentence_lookup(corpus), page_index(corpus)
        splits_dir = find_splits_dir(cfg)
        shared["claims"] = {"val": load_claims(splits_dir, "val"), "test": load_claims(splits_dir, test_split_name(cfg))}
    texts, pages, claims = shared["texts"], shared["pages"], shared["claims"]
    key = f"cache::{nc['model']}"
    if key not in shared:
        shared[key] = NLICache(cfg, NLIScorer(nc["model"], nc["fp16"], nc["max_length"], nc["batch_size"], log), log)
    cache = shared[key]

    evid, missing, agg, gold = {}, {}, {}, {}
    for sp in ("val", "test"):
        evid[sp], missing[sp], desc = load_evidence(cfg, pc["evidence"], sp, claims[sp], texts, pages)
        agg[sp] = aggregate(sentence_table(evid[sp], claims[sp], texts, cache), claims[sp])
        gold[sp] = gold_indices(claims[sp])
    log.info(f"Pipeline '{pc['name']}': evidence {desc}, NLI {nc['model']}, calibration {pc['calibration']}")

    grid = pc["threshold_grid"]
    if pc["calibration"]:
        taus = np.round(np.arange(pc["tau_min"], pc["tau_max"] + 1e-9, pc["tau_step"]), 3)
        params = calibrate(agg["val"], gold["val"], grid, taus)
        raw = rule_labels(agg["test"]["max_pe"], agg["test"]["max_pc"], params["theta_e"], params["theta_c"])
        conf = confidence(softmax(claim_logits(agg["test"]["P"]) / params["T"]), raw)
        labels = abstain(raw, conf, params["tau"])
        conf = np.where(labels != raw, confidence(softmax(claim_logits(agg["test"]["P"]) / params["T"]), labels), conf)
    else:
        te, tc, _ = tune_thresholds(agg["val"]["max_pe"], agg["val"]["max_pc"], gold["val"], grid)
        params = {"theta_e": te, "theta_c": tc, "T": 1.0, "tau": None}
        labels = rule_labels(agg["test"]["max_pe"], agg["test"]["max_pc"], te, tc)
        conf = confidence(agg["test"]["P"], labels)

    # Optional for chains: answer NEI when the chain found no confident hop 2. Kept only if it helps on validation.
    if missing["test"] and pc.get("missing_link_nei") == "tune":
        val_lab = rule_labels(agg["val"]["max_pe"], agg["val"]["max_pc"], params["theta_e"], params["theta_c"])
        ml_val = np.array([missing["val"].get(c["id"], False) for c in claims["val"]])
        f_off, f_on = macro_f1_idx(gold["val"], val_lab), macro_f1_idx(gold["val"], np.where(ml_val, 2, val_lab))
        params["missing_link_nei"] = bool(f_on > f_off)
        log.info(f"missing_link -> NEI: validation macro-F1 {f_off:.4f} without, {f_on:.4f} with -> "
                 f"{'used' if params['missing_link_nei'] else 'not used'}")
        if params["missing_link_nei"]:
            ml = np.array([missing["test"].get(c["id"], False) for c in claims["test"]])
            labels = np.where(ml, 2, labels)
            conf = np.where(ml, agg["test"]["P"][:, 2], conf)

    test = claims["test"]
    preds = [{"id": c["id"], "predicted_label": LABEL_OF[l], "predicted_evidence": [[p, int(s)] for p, s in evid["test"].get(c["id"], [])[:5]]}
             for c, l in zip(test, labels)]
    write_jsonl(preds, run_dir / "predictions_test.jsonl")
    scores = score_predictions(preds, test, log)
    sf = mean_supporting_fact_prf(([tuple(e) for e in p["predicted_evidence"]], {x for g in evidence_sets(c) for x in g})
                                  for p, c in zip(preds, test) if evidence_sets(c))
    correct = (labels == gold["test"]).tolist()
    row = {
        "name": pc["name"], "evidence": desc, "nli_model": nc["model"], "test_split": test_split_name(cfg), "n_claims": len(test), "n_val": len(claims["val"]),
        "label_accuracy": scores["label_accuracy"], "macro_f1": macro_f1_idx(gold["test"], labels),
        "fever_score": scores["fever_score"], "sf_f1": sf.get("sf_f1", float("nan")), "sf_precision": sf.get("sf_precision"),
        "sf_recall": sf.get("sf_recall"), "ece": expected_calibration_error(conf.tolist(), correct),
        "predicted_label_counts": {LABEL_OF[i]: int((labels == i).sum()) for i in range(3)}, "params": params,
        "scorer": scores["scorer"], **{k: v for k, v in scores.items() if k.startswith("official_")},
    }
    save_json(row, run_dir / "metrics.json")
    log.info(f"{pc['name']}: label accuracy {row['label_accuracy']:.4f} | macro-F1 {row['macro_f1']:.4f} | FEVER score "
             f"{row['fever_score']:.4f} | sf-F1 {row['sf_f1']:.4f} | ECE {row['ece']:.4f} | predicted {row['predicted_label_counts']}")
    return row


# --------------------------------------------------------------------------- table
def row_slug(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")


def run_table(table: str, base_config: str) -> int:
    base = load_config(base_config)
    entries = base["tables"][table]
    shared: dict = {}
    rows = []
    for entry in entries:
        cfg = load_config(entry["config"])
        if entry.get("needs_output") and find_output(cfg, entry["needs_output"], "metrics.json") is None:
            print(f"Skipping {entry['config']}: outputs/{entry['needs_output']} not found")
            continue
        cfg["run_name"] = f"pipeline/{row_slug(cfg['pipeline']['name'])}"
        rows.append(run_pipeline(cfg, shared))

    out_dir = resolve(base["paths"]["outputs_dir"]) / "tables"
    out_dir.mkdir(parents=True, exist_ok=True)
    cols = ["name", "evidence", "label_accuracy", "macro_f1", "fever_score", "sf_f1", "ece", "n_claims", "test_split"]
    df = pd.DataFrame(rows)[cols]
    df.to_csv(out_dir / f"{table}.csv", index=False)
    header = "| Pipeline | Evidence | Label accuracy | Macro-F1 | FEVER score | Supporting-fact F1 | ECE |"
    lines = [header, "|---|---|---|---|---|---|---|"]
    for r in rows:
        lines.append(f"| {r['name']} | {r['evidence']} | {r['label_accuracy']:.4f} | {r['macro_f1']:.4f} | {r['fever_score']:.4f} | "
                     f"{r['sf_f1']:.4f} | {r['ece']:.4f} |")
    note = (f"\nTest split: {rows[0]['test_split']} ({rows[0]['n_claims']:,} claims). Thresholds, temperature and tau are tuned on the "
            f"{rows[0]['n_val']:,}-claim validation split. Supporting-fact F1 = mean over SUPPORTS/REFUTES claims; for the gold-evidence row it "
            "is close to 1 by construction.\n") if rows else ""
    (out_dir / f"{table}.md").write_text(f"# {table.capitalize()} results\n\n" + "\n".join(lines) + "\n" + note, encoding="utf-8")
    print("\n".join(lines) + "\n" + note)
    print(f"Wrote {out_dir / f'{table}.csv'} and {out_dir / f'{table}.md'}")
    return 0


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--config", help="one pipeline config, e.g. configs/pipeline_hybrid.yaml")
    parser.add_argument("--table", help="table name listed in configs/pipeline_base.yaml, e.g. verification")
    parser.add_argument("--base", default="configs/pipeline_base.yaml")
    args = parser.parse_args(argv)
    if args.table:
        return run_table(args.table, args.base)
    if not args.config:
        parser.error("give --table or --config")
    run_pipeline(load_config(args.config))
    return 0


if __name__ == "__main__":
    sys.exit(main())
