"""Prompt 13 (M4): calibration and "NOT ENOUGH INFO when unsure".

Run (after Prompt 12; uses the cached NLI logits, so a GPU is only needed for pairs not yet scored):
    python -m src.verify.calibrate --config configs/calibrate.yaml

What is calibrated
- For each claim the verifier gives a distribution P over (SUPPORTS, REFUTES, NEI): the normalised
  (max p(E), max p(C), min p(N)) over its evidence sentences (src/verify/evidence.aggregate). Its log is
  treated as the claim's logits z.
- Temperature scaling: P_T = softmax(z / T), one number T fitted on validation by minimising the negative
  log-likelihood of the gold labels. T > 1 softens over-confident probabilities; the predicted label does
  not change, only how sure the system says it is.
- Confidence of a claim = P_T[label chosen by the aggregation rule]. ECE (10 bins) and a reliability
  diagram (confidence vs accuracy per bin) are reported before (T = 1) and after.
- Abstention: if the confidence < tau, answer NOT ENOUGH INFO. tau in [0.3, 0.9] is tuned on validation for
  macro-F1; the accuracy vs coverage curve (coverage = share of claims answered without abstaining) is drawn.
Outputs in outputs/calibration/: metrics.json (T, tau, ECE before/after), reliability_val.png,
reliability_test.png, coverage.png, coverage.csv.
"""
from __future__ import annotations

import argparse
import sys

import numpy as np
import pandas as pd

from src.eval.metrics import expected_calibration_error, reliability_bins
from src.retrieval.common import find_corpus_dir, find_splits_dir, load_claims, load_corpus, sentence_lookup, test_split_name
from src.utils import load_config, save_json, set_seed, start_run
from src.verify.evidence import (
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


# --------------------------------------------------------------------------- temperature scaling
def claim_logits(P: np.ndarray) -> np.ndarray:
    return np.log(np.clip(P, 1e-8, 1.0))


def nll(z: np.ndarray, y: np.ndarray, T: float) -> float:
    p = softmax(z / T)
    return float(-np.mean(np.log(np.clip(p[np.arange(len(y)), y], 1e-12, None))))


def fit_temperature(z: np.ndarray, y: np.ndarray) -> float:
    """T minimising validation NLL, searched over log T in [-3, 3] (T between 0.05 and 20)."""
    from scipy.optimize import minimize_scalar

    res = minimize_scalar(lambda lt: nll(z, y, float(np.exp(lt))), bounds=(-3.0, 3.0), method="bounded")
    return float(np.exp(res.x))


def confidence(P: np.ndarray, labels: np.ndarray) -> np.ndarray:
    return P[np.arange(len(labels)), labels]


# --------------------------------------------------------------------------- abstention
def abstain(labels: np.ndarray, conf: np.ndarray, tau: float) -> np.ndarray:
    """Replace the label by NEI (index 2) when the confidence is below tau."""
    return np.where(conf < tau, 2, labels)


def coverage_curve(labels: np.ndarray, conf: np.ndarray, gold: np.ndarray, taus) -> pd.DataFrame:
    rows = []
    for tau in taus:
        answered = conf >= tau
        final = abstain(labels, conf, tau)
        rows.append({"tau": round(float(tau), 3), "coverage": float(answered.mean()),
                     "accuracy_answered": float((labels[answered] == gold[answered]).mean()) if answered.any() else float("nan"),
                     "accuracy_final": float((final == gold).mean()), "macro_f1_final": macro_f1_idx(gold, final)})
    return pd.DataFrame(rows)


def tune_tau(labels: np.ndarray, conf: np.ndarray, gold: np.ndarray, taus) -> tuple[float, pd.DataFrame]:
    """tau with the best validation macro-F1 (ties -> the smaller tau, i.e. more claims answered).
    Taus are compared after rounding to 3 decimals, so float steps cannot decide a tie."""
    curve = coverage_curve(labels, conf, gold, taus)
    best = curve.assign(neg_tau=-curve["tau"], f=curve["macro_f1_final"].round(6)).sort_values(["f", "neg_tau"], ascending=False).iloc[0]
    return float(best["tau"]), curve


def calibrate(val_agg: dict, val_gold: np.ndarray, grid: list[float], taus) -> dict:
    """Everything fitted on validation: thresholds, T, tau."""
    te, tc, _ = tune_thresholds(val_agg["max_pe"], val_agg["max_pc"], val_gold, grid)
    labels = rule_labels(val_agg["max_pe"], val_agg["max_pc"], te, tc)
    z = claim_logits(val_agg["P"])
    T = fit_temperature(z, val_gold)
    tau, _ = tune_tau(labels, confidence(softmax(z / T), labels), val_gold, taus)
    return {"theta_e": te, "theta_c": tc, "T": T, "tau": tau}


def apply(agg: dict, params: dict, use_tau: bool = True) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(final labels, calibrated confidence, labels before abstention) for one split."""
    labels = rule_labels(agg["max_pe"], agg["max_pc"], params["theta_e"], params["theta_c"])
    conf = confidence(softmax(claim_logits(agg["P"]) / params["T"]), labels)
    return (abstain(labels, conf, params["tau"]) if use_tau else labels), conf, labels


# --------------------------------------------------------------------------- plots
def plot_reliability(before: list[dict], after: list[dict], ece_b: float, ece_a: float, title: str, path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot([0, 1], [0, 1], ls="--", color="grey", lw=1, label="perfect calibration")
    for rows, name, ece, mk in ((before, "before (T = 1)", ece_b, "o"), (after, "after temperature scaling", ece_a, "s")):
        pts = [(r["confidence"], r["accuracy"]) for r in rows if r["count"]]
        ax.plot([p[0] for p in pts], [p[1] for p in pts], marker=mk, label=f"{name}, ECE {ece:.3f}")
    ax.set_xlabel("confidence (mean per bin)")
    ax.set_ylabel("accuracy (per bin)")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_title(title)
    ax.legend(loc="upper left", fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_coverage(curves: dict, tau: float, path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6, 4))
    for name, c in curves.items():
        ax.plot(c["coverage"], c["accuracy_answered"], marker="o", ms=3, label=f"{name}: accuracy on answered claims")
        sel = c[np.isclose(c["tau"], tau)]
        if len(sel):
            ax.annotate(f"tau = {tau}", (sel["coverage"].iloc[0], sel["accuracy_answered"].iloc[0]), textcoords="offset points", xytext=(5, -12), fontsize=8)
    ax.set_xlabel("coverage (share of claims answered, tau from 0.3 to 0.9)")
    ax.set_ylabel("accuracy")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


# --------------------------------------------------------------------------- main
def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--config", default="configs/calibrate.yaml")
    args = parser.parse_args(argv)

    cfg = load_config(args.config)
    set_seed(cfg["seed"])
    run_dir, log = start_run(cfg)
    cc = cfg["calibration"]
    taus = np.round(np.arange(cc["tau_min"], cc["tau_max"] + 1e-9, cc["tau_step"]), 3)

    corpus = load_corpus(find_corpus_dir(cfg))
    texts, pages = sentence_lookup(corpus), page_index(corpus)
    splits_dir = find_splits_dir(cfg)
    split = test_split_name(cfg)
    claims = {"val": load_claims(splits_dir, "val"), "test": load_claims(splits_dir, split)}
    cache = NLICache(cfg, NLIScorer(cc["model"], cc["fp16"], cc["max_length"], cc["batch_size"], log), log)
    agg, gold = {}, {}
    for sp in ("val", "test"):
        evid, _, desc = load_evidence(cfg, cc["evidence"], sp, claims[sp], texts, pages)
        agg[sp] = aggregate(sentence_table(evid, claims[sp], texts, cache), claims[sp])
        gold[sp] = gold_indices(claims[sp])
    log.info(f"Calibrating {cc['model']} with evidence {desc}")

    params = calibrate(agg["val"], gold["val"], cc["threshold_grid"], taus)
    log.info(f"Fitted on validation: theta_e {params['theta_e']}, theta_c {params['theta_c']}, T {params['T']:.3f}, tau {params['tau']}")

    metrics = {"evidence": desc, "model": cc["model"], "test_split": split, **params, "splits": {}}
    curves = {}
    for sp in ("val", "test"):
        labels = rule_labels(agg[sp]["max_pe"], agg[sp]["max_pc"], params["theta_e"], params["theta_c"])
        z = claim_logits(agg[sp]["P"])
        correct = (labels == gold[sp]).tolist()
        conf_b, conf_a = confidence(softmax(z), labels), confidence(softmax(z / params["T"]), labels)
        ece_b = expected_calibration_error(conf_b.tolist(), correct)
        ece_a = expected_calibration_error(conf_a.tolist(), correct)
        rb, ra = reliability_bins(conf_b.tolist(), correct), reliability_bins(conf_a.tolist(), correct)
        plot_reliability(rb, ra, ece_b, ece_a, f"Reliability diagram ({sp})", run_dir / f"reliability_{sp}.png")
        curves[sp] = coverage_curve(labels, conf_a, gold[sp], taus)
        final = abstain(labels, conf_a, params["tau"])
        metrics["splits"][sp] = {
            "ece_before": ece_b, "ece_after": ece_a, "nll_before": nll(z, gold[sp], 1.0), "nll_after": nll(z, gold[sp], params["T"]),
            "accuracy_no_abstention": float((labels == gold[sp]).mean()), "macro_f1_no_abstention": macro_f1_idx(gold[sp], labels),
            "accuracy_with_abstention": float((final == gold[sp]).mean()), "macro_f1_with_abstention": macro_f1_idx(gold[sp], final),
            "coverage_at_tau": float((conf_a >= params["tau"]).mean()), "reliability_before": rb, "reliability_after": ra,
        }
        m = metrics["splits"][sp]
        log.info(f"{sp.upper()}: ECE {ece_b:.4f} -> {ece_a:.4f} | NLL {m['nll_before']:.4f} -> {m['nll_after']:.4f} | "
                 f"macro-F1 {m['macro_f1_no_abstention']:.4f} -> {m['macro_f1_with_abstention']:.4f} with tau {params['tau']} "
                 f"(coverage {m['coverage_at_tau']:.1%})")
    pd.concat({sp: c for sp, c in curves.items()}, names=["split"]).to_csv(run_dir / "coverage.csv")
    plot_coverage(curves, params["tau"], run_dir / "coverage.png")
    log.info("Accuracy vs coverage on test:\n" + curves["test"].round(4).to_string(index=False))
    save_json(metrics, run_dir / "metrics.json")
    log.info(f"Outputs written to {run_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
