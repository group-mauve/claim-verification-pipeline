"""Unit tests for src/eval/metrics.py.  Run:  python -m pytest tests -q"""
import random

import pytest

from src.eval import metrics as M
from src.utils import LABELS, locate_file, read_jsonl


# ----------------------------------------------------------------- hand-worked slide examples
def test_slide_ranking_example():
    # Top-5 results, relevant items at ranks 2 and 4, three relevant items in total.
    ranked = ["a", "R1", "b", "R2", "c"]
    relevant = {"R1", "R2", "R3"}
    assert round(M.recall_at_k(ranked, relevant, 5), 2) == 0.67
    assert M.reciprocal_rank(ranked, relevant) == 0.5
    assert round(M.ndcg_at_k(ranked, relevant, 5), 2) == 0.50


def test_slide_supporting_fact_example():
    p, r, f1 = M.supporting_fact_prf({1, 2, 4, 5}, {1, 2, 4, 6, 7})
    assert (round(p, 2), round(r, 2), round(f1, 2)) == (0.75, 0.60, 0.67)


# ----------------------------------------------------------------- ranking edge cases
def test_ranking_edge_cases():
    assert M.recall_at_k(["x"], set(), 5) == 0.0
    assert M.reciprocal_rank(["x", "y"], {"z"}) == 0.0
    assert M.ndcg_at_k(["R"], {"R"}, 10) == 1.0           # perfect ranking
    assert M.ndcg_at_k(["a", "b"], {"R"}, 10) == 0.0
    assert M.recall_at_k(["a", "R"], {"R"}, 1) == 0.0       # cut-off respected


def test_page_ranking_and_report():
    ranked = [("P1", 3), ("P1", 0), ("P2", 1), ("P3", 0)]
    assert M.to_page_ranking(ranked) == ["P1", "P2", "P3"]
    gold = {1: [[("P2", 1)]], 2: [[("P1", 0), ("P3", 0)]], 3: []}  # claim 3 is NEI -> skipped
    runs = {1: ranked, 2: ranked}
    rep = M.retrieval_report(runs, gold, ks=(1, 5), ndcg_k=5)
    assert rep["n_claims"] == 2
    assert rep["page"]["mrr"] == pytest.approx((1 / 2 + 1) / 2)
    assert rep["sentence"]["evidence_set_hit@5"] == 1.0
    assert rep["sentence"]["evidence_set_hit@1"] == 0.0


# ----------------------------------------------------------------- classification
def test_accuracy_and_macro_f1():
    y_true = ["SUPPORTS", "SUPPORTS", "REFUTES", "NOT ENOUGH INFO"]
    y_pred = ["SUPPORTS", "REFUTES", "REFUTES", "SUPPORTS"]
    assert M.accuracy(y_true, y_pred) == 0.5
    # F1: S = 2*(.5*.5)/(1) = .5 ; R = 2*(.5*1)/1.5 = .667 ; NEI = 0
    assert M.macro_f1(y_true, y_pred) == pytest.approx((0.5 + 2 / 3 + 0) / 3)


def test_macro_f1_matches_sklearn():
    sklearn = pytest.importorskip("sklearn.metrics")
    rng = random.Random(0)
    y_true = [rng.choice(LABELS) for _ in range(500)]
    y_pred = [rng.choice(LABELS) for _ in range(500)]
    assert M.macro_f1(y_true, y_pred) == pytest.approx(sklearn.f1_score(y_true, y_pred, labels=LABELS, average="macro"))


# ----------------------------------------------------------------- calibration
def test_ece():
    assert M.expected_calibration_error([0.8] * 10, [True] * 8 + [False] * 2) == pytest.approx(0.0)
    assert M.expected_calibration_error([1.0, 1.0], [False, False]) == pytest.approx(1.0)
    # Two bins: (0.9,1] acc 1 conf .95 ; (0.5,0.6] acc 0 conf .55  -> (0.05 + 0.55) / 2
    assert M.expected_calibration_error([0.95, 0.55], [True, False]) == pytest.approx(0.30)
    # 0.3 belongs to bin (0.2, 0.3], not (0.3, 0.4]
    assert [b["count"] for b in M.reliability_bins([0.3], [True])][2] == 1


# ----------------------------------------------------------------- FEVER score
GOLD = [
    {"label": "SUPPORTS", "evidence": [[[1, 1, "A", 0]], [[2, 2, "B", 3], [2, 2, "C", 1]]]},
    {"label": "REFUTES", "evidence": [[[3, 3, "D", 2], [3, 3, "E", 0]]]},
    {"label": "NOT ENOUGH INFO", "evidence": [[[4, None, None, None]]]},
    {"label": "SUPPORTS", "evidence": [[[5, 5, "F", 0]]]},
]


def test_fever_score_hand_example():
    preds = [
        {"predicted_label": "SUPPORTS", "predicted_evidence": [["B", 3], ["C", 1]]},  # 2nd set complete -> strict
        {"predicted_label": "REFUTES", "predicted_evidence": [["D", 2]]},             # set incomplete -> label only
        {"predicted_label": "NOT ENOUGH INFO", "predicted_evidence": []},             # NEI needs no evidence
        {"predicted_label": "REFUTES", "predicted_evidence": [["F", 0]]},             # wrong label
    ]
    assert M.fever_score(preds, GOLD) == {"fever_score": 0.5, "label_accuracy": 0.75}


def test_fever_score_only_first_five_sentences_count():
    preds = [{"predicted_label": "SUPPORTS", "predicted_evidence": [["x", i] for i in range(5)] + [["A", 0]]}]
    assert M.fever_score(preds, GOLD[:1])["fever_score"] == 0.0


def _random_predictions(gold, seed=42):
    """Random labels; evidence mixes gold sentences, partial sets and noise to hit every branch."""
    rng = random.Random(seed)
    preds = []
    for g in gold:
        label = g["label"] if rng.random() < 0.6 else rng.choice(LABELS)
        ev = []
        groups = [grp for grp in g["evidence"] if grp[0][2] is not None]
        if groups and rng.random() < 0.7:
            grp = rng.choice(groups)
            ev = [[e[2], e[3]] for e in grp]
            if len(ev) > 1 and rng.random() < 0.3:
                ev = ev[:-1]  # incomplete set
        ev += [[f"Noise_{rng.randint(0, 99)}", rng.randint(0, 9)] for _ in range(rng.randint(0, 5))]
        rng.shuffle(ev)
        preds.append({"predicted_label": label, "predicted_evidence": ev})
    return preds


def test_matches_official_fever_scorer_on_200_dev_claims():
    scorer = pytest.importorskip("fever.scorer", reason="clone sheffieldnlp/fever-scorer and add its src/ folder to PYTHONPATH")
    dev_path = locate_file("shared_task_dev.jsonl", ["data/raw", ".", "/kaggle/input"])
    if dev_path is None:
        pytest.skip("shared_task_dev.jsonl not found")
    dev = read_jsonl(dev_path)
    gold = random.Random(42).sample(dev, 200)
    preds = _random_predictions(gold)
    ours = M.fever_score(preds, gold)
    official = [{"label": g["label"], "evidence": g["evidence"], **p} for g, p in zip(gold, preds)]
    strict, acc, *_ = scorer.fever_score(official, max_evidence=5)
    assert ours["fever_score"] == pytest.approx(strict)
    assert ours["label_accuracy"] == pytest.approx(acc)
