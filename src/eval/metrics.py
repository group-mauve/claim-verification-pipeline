"""Prompt 4 (M1): every evaluation metric used in the project.

Conventions
- A ranked list is a Python list, best first. Items can be anything hashable: a page id
  ("Telemundo") for page-level metrics, or a (page, sent_id) tuple for sentence-level metrics.
- Relevance is binary: an item is either in the relevant set or not.
- Per-query functions return one number; `retrieval_report` averages them over many claims.
"""
from __future__ import annotations

import math
from collections.abc import Hashable, Iterable, Sequence

from src.utils import LABELS

Item = Hashable


# ============================================================================ ranking metrics
def dedupe(items: Iterable[Item]) -> list[Item]:
    """Remove repeats, keeping the first (best-ranked) occurrence."""
    seen, out = set(), []
    for x in items:
        if x not in seen:
            seen.add(x)
            out.append(x)
    return out


def to_page_ranking(ranked_sentences: Sequence[tuple[str, int]]) -> list[str]:
    """Turn a ranked list of (page, sent_id) into a ranked list of pages.

    A page takes the rank of its best sentence, which is the same as ranking pages by their best
    sentence score.
    """
    return dedupe(page for page, _ in ranked_sentences)


def recall_at_k(ranked: Sequence[Item], relevant: set, k: int) -> float:
    """Share of the relevant items that appear in the top k results.

    Recall@5 = 0.67 means two of the three relevant items were in the first five results.
    """
    if not relevant:
        return 0.0
    return len(set(ranked[:k]) & relevant) / len(relevant)


def reciprocal_rank(ranked: Sequence[Item], relevant: set) -> float:
    """1 / rank of the first relevant result (0 if none is retrieved).

    Averaged over queries this is MRR; it rewards putting a correct item at the very top.
    """
    for i, x in enumerate(ranked, start=1):
        if x in relevant:
            return 1.0 / i
    return 0.0


def ndcg_at_k(ranked: Sequence[Item], relevant: set, k: int) -> float:
    """Discounted cumulative gain of the top k, divided by the best possible DCG.

    Each relevant item at rank r adds 1 / log2(r + 1), so hits near the top count more;
    dividing by the ideal DCG puts the score between 0 and 1.
    """
    if not relevant:
        return 0.0
    dcg = sum(1.0 / math.log2(i + 1) for i, x in enumerate(ranked[:k], start=1) if x in relevant)
    ideal = sum(1.0 / math.log2(i + 1) for i in range(1, min(len(relevant), k) + 1))
    return dcg / ideal


def evidence_set_hit_at_k(ranked_sentences: Sequence[tuple[str, int]], sets: list[list[tuple[str, int]]], k: int) -> float:
    """1 if at least one complete annotated evidence set is inside the top k sentences, else 0.

    This is the FEVER-style retrieval measure: a claim is only provable if every sentence of
    some evidence set was found.
    """
    top = set(ranked_sentences[:k])
    return float(any(all(tuple(s) in top for s in group) for group in sets))


def retrieval_report(
    runs: dict, gold: dict, ks: Sequence[int] = (5, 10, 20), ndcg_k: int = 10
) -> dict[str, dict[str, float]]:
    """Mean retrieval metrics over claims, at sentence and page level.

    runs: claim_id -> ranked list of (page, sent_id)
    gold: claim_id -> list of evidence sets, each a list of (page, sent_id)
    Claims with no evidence (NOT ENOUGH INFO) are skipped, since nothing is relevant for them.
    """
    sums: dict[str, dict[str, float]] = {"sentence": {}, "page": {}}
    n = 0
    for cid, sets in gold.items():
        if not sets:
            continue
        n += 1
        ranked_s = [tuple(x) for x in runs.get(cid, [])]
        rel_s = {tuple(x) for group in sets for x in group}
        ranked_p = to_page_ranking(ranked_s)
        rel_p = {p for p, _ in rel_s}
        for level, ranked, rel in (("sentence", ranked_s, rel_s), ("page", ranked_p, rel_p)):
            row = sums[level]
            for k in ks:
                row[f"recall@{k}"] = row.get(f"recall@{k}", 0.0) + recall_at_k(ranked, rel, k)
            row["mrr"] = row.get("mrr", 0.0) + reciprocal_rank(ranked, rel)
            row[f"ndcg@{ndcg_k}"] = row.get(f"ndcg@{ndcg_k}", 0.0) + ndcg_at_k(ranked, rel, ndcg_k)
        for k in ks:
            key = f"evidence_set_hit@{k}"
            sums["sentence"][key] = sums["sentence"].get(key, 0.0) + evidence_set_hit_at_k(ranked_s, sets, k)
    report = {level: {m: v / n for m, v in row.items()} for level, row in sums.items()} if n else sums
    report["n_claims"] = n  # type: ignore[assignment]
    return report


# ============================================================================ evidence metrics
def supporting_fact_prf(predicted: Iterable[Item], gold: Iterable[Item]) -> tuple[float, float, float]:
    """Precision, recall and F1 between a predicted and a gold set of (page, sent_id) facts.

    Precision is the share of predicted facts that are correct; recall is the share of gold facts
    that were predicted; F1 is their harmonic mean (HotpotQA's supporting-fact metric).
    """
    pred, true = set(predicted), set(gold)
    tp = len(pred & true)
    p = tp / len(pred) if pred else 0.0
    r = tp / len(true) if true else 0.0
    f1 = 2 * p * r / (p + r) if p + r else 0.0
    return p, r, f1


def mean_supporting_fact_prf(pairs: Iterable[tuple[Iterable[Item], Iterable[Item]]]) -> dict[str, float]:
    """Average supporting_fact_prf over many (predicted, gold) pairs."""
    tot, n = [0.0, 0.0, 0.0], 0
    for pred, gold in pairs:
        for i, v in enumerate(supporting_fact_prf(pred, gold)):
            tot[i] += v
        n += 1
    return {"sf_precision": tot[0] / n, "sf_recall": tot[1] / n, "sf_f1": tot[2] / n} if n else {}


# ============================================================================ classification
def accuracy(y_true: Sequence[str], y_pred: Sequence[str]) -> float:
    """Share of claims whose predicted label equals the gold label.

    Easy to read, but it can look good on an imbalanced set by favouring the majority class.
    """
    assert len(y_true) == len(y_pred) and y_true
    return sum(t == p for t, p in zip(y_true, y_pred)) / len(y_true)


def per_class_f1(y_true: Sequence[str], y_pred: Sequence[str], labels: Sequence[str] = LABELS) -> dict[str, float]:
    out = {}
    for label in labels:
        tp = sum(t == label and p == label for t, p in zip(y_true, y_pred))
        fp = sum(t != label and p == label for t, p in zip(y_true, y_pred))
        fn = sum(t == label and p != label for t, p in zip(y_true, y_pred))
        prec = tp / (tp + fp) if tp + fp else 0.0
        rec = tp / (tp + fn) if tp + fn else 0.0
        out[label] = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    return out


def macro_f1(y_true: Sequence[str], y_pred: Sequence[str], labels: Sequence[str] = LABELS) -> float:
    """Unweighted mean of the F1 score of each of the three labels.

    Every class counts equally, so a model that ignores NOT ENOUGH INFO is penalised even if
    its accuracy looks fine.
    """
    scores = per_class_f1(y_true, y_pred, labels)
    return sum(scores.values()) / len(labels)


# ============================================================================ calibration
def _bin_index(conf: float, n_bins: int) -> int:
    """Bin b holds confidences in (b/M, (b+1)/M]. The 1e-9 stops 0.3*10 = 3.0000000000000004
    from pushing 0.3 into the next bin."""
    return min(max(math.ceil(conf * n_bins - 1e-9) - 1, 0), n_bins - 1)


def expected_calibration_error(confidences: Sequence[float], correct: Sequence[bool], n_bins: int = 10) -> float:
    """Average gap between confidence and accuracy, over M equal-width confidence bins.

    ECE = sum over bins of (bin size / N) * |accuracy in bin - mean confidence in bin|; a model
    that says 80% and is right 80% of the time scores 0.
    Bins are (0, 0.1], (0.1, 0.2], ..., (0.9, 1.0]; a confidence of exactly 0 goes in the first bin.
    """
    assert len(confidences) == len(correct) and confidences
    n = len(confidences)
    bins = [[0, 0.0, 0.0] for _ in range(n_bins)]  # count, sum confidence, sum correct
    for conf, ok in zip(confidences, correct):
        b = _bin_index(conf, n_bins)
        bins[b][0] += 1
        bins[b][1] += conf
        bins[b][2] += float(ok)
    return sum(cnt / n * abs(s_ok / cnt - s_conf / cnt) for cnt, s_conf, s_ok in bins if cnt)


def reliability_bins(confidences: Sequence[float], correct: Sequence[bool], n_bins: int = 10) -> list[dict]:
    """Per-bin (count, mean confidence, accuracy) for drawing a reliability diagram."""
    rows = []
    which = [_bin_index(c, n_bins) for c in confidences]
    for b in range(n_bins):
        lo, hi = b / n_bins, (b + 1) / n_bins
        idx = [i for i, w in enumerate(which) if w == b]
        rows.append({
            "bin": b, "lo": lo, "hi": hi, "count": len(idx),
            "confidence": sum(confidences[i] for i in idx) / len(idx) if idx else None,
            "accuracy": sum(correct[i] for i in idx) / len(idx) if idx else None,
        })
    return rows


# ============================================================================ FEVER score
def is_strictly_correct(pred_label: str, pred_evidence: Sequence, gold_label: str, gold_evidence: list, max_evidence: int = 5) -> bool:
    """FEVER rule for one claim: label right AND (gold is NEI OR one full evidence set predicted)."""
    if pred_label.upper() != gold_label.upper():
        return False
    if gold_label.upper() == "NOT ENOUGH INFO":
        return True
    predicted = {(str(p), int(s)) for p, s in list(pred_evidence)[:max_evidence]}
    for group in gold_evidence:
        needed = [(e[2], int(e[3])) for e in group if e[2] is not None]
        if needed and all(x in predicted for x in needed):
            return True
    return False


def fever_score(predictions: Sequence[dict], gold: Sequence[dict], max_evidence: int = 5) -> dict[str, float]:
    """FEVER score and label accuracy, matching the official sheffieldnlp scorer.

    FEVER score counts a claim only if its label is right and, for S/R claims, the first five
    predicted sentences contain every sentence of at least one annotated evidence set. Label
    accuracy ignores evidence.

    predictions[i] = {"predicted_label": str, "predicted_evidence": [[page, sent_id], ...]}
    gold[i]        = a FEVER claim dict with "label" and "evidence" (same order as predictions).
    """
    assert len(predictions) == len(gold) and gold
    strict = correct = 0
    for p, g in zip(predictions, gold):
        if p["predicted_label"].upper() == g["label"].upper():
            correct += 1
            strict += is_strictly_correct(p["predicted_label"], p["predicted_evidence"], g["label"], g["evidence"], max_evidence)
    return {"fever_score": strict / len(gold), "label_accuracy": correct / len(gold)}
