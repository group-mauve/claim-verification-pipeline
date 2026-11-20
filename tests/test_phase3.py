"""Tests for Phase 3 (Prompts 8-14): fusion, fine-tuning rows, chain, NLI aggregation, calibration, pipeline.

Unit tests check the hand-worked numbers; the end-to-end test runs every Phase 3 script in order on the
tiny corpus with fake models (no GPU, no downloads)."""
import json
import logging

import numpy as np
import pytest

from src.chain import two_hop as H
from src.retrieval import common as C
from src.retrieval import hybrid as Y
from src.retrieval.finetune import build_training_rows
from src.utils import load_config
from src.verify import calibrate as K
from src.verify import evidence as E
from tests.fakes import FakeNLIScorer, FakeST

log = logging.getLogger("test")


# ----------------------------------------------------------------- Prompt 9: fusion
def test_rrf_slide_example():
    fused = Y.rrf_scores([["A", "B", "C", "D"], ["C", "A", "E", "F"]], k=60)
    assert [d for d, _ in fused[:2]] == ["A", "C"]
    assert round(fused[0][1], 4) == 0.0325 and round(fused[1][1], 4) == 0.0323
    assert dict(fused)["B"] == pytest.approx(1 / 62)


def test_minmax_and_weighted_fusion():
    assert Y.minmax({"a": 2.0, "b": 4.0, "c": 3.0}) == {"a": 0.0, "b": 1.0, "c": 0.5}
    assert Y.minmax({"a": 5.0}) == {"a": 1.0}
    fused = dict(Y.weighted_scores({"x": 0.9, "y": 0.1}, {"y": 30.0, "z": 10.0}, alpha=0.5))
    assert fused == {"x": 0.5, "y": 0.5, "z": 0.0}
    run = Y.fuse_runs({1: [("P", 0, 30.0), ("Q", 1, 10.0)]}, {1: [("Q", 1, 0.9), ("R", 2, 0.5)]}, "rrf", top_k=2)
    assert [(p, s) for p, s, _ in run[1]] == [("Q", 1), ("P", 0)]


# ----------------------------------------------------------------- Prompt 8: training rows
def test_hard_negative_skips_evidence_and_evidence_pages():
    texts = {("A", 0): "a0", ("A", 1): "a1", ("B", 0): "b0", ("C", 0): "c0"}
    claims = [{"id": 1, "label": "SUPPORTS", "claim": "x", "evidence": [[[0, 0, "A", 0]]]},
              {"id": 2, "label": "NOT ENOUGH INFO", "claim": "y", "evidence": [[[0, None, None, None]]]}]
    run = {1: [("A", 0, 9.0), ("A", 1, 8.0), ("B", 0, 7.0), ("C", 0, 6.0)]}
    rows, stats = build_training_rows(claims, run, texts, seed=42)
    assert len(rows) == 1 and stats["skipped_nei"] == 1
    assert rows[0]["pos_id"] == ["A", 0]
    assert rows[0]["neg_id"] == ["B", 0] and rows[0]["neg_bm25_rank"] == 3   # A:1 is skipped: same page as the evidence
    assert rows[0]["positive"] == "A: a0"


# ----------------------------------------------------------------- Prompt 11: chain helpers
def test_bridge_entities_and_chain_budget():
    assert H.base_name("Soul_Food_-LRB-film-RRB-") == "Soul Food"
    assert H.mentioned_in("Soul Food (film)", "Soul Food is a 1997 film")
    links = {("Damon_Albarn", 0): ["Blur (band)", "Damon Albarn", "Gorillaz"]}
    hop1 = [(("Damon_Albarn", 0), 0.9), (("Other", 1), 0.5)]
    br = H.pick_bridges("Damon Albarn sang for Blur.", hop1, links, {("Other", 1): ["Parlophone"]}, lambda k: k[0], 3)
    assert [b["entity"] for b in br] == ["Gorillaz", "Parlophone"]       # Blur is in the claim, Damon Albarn is the page itself
    assert [b["source"] for b in br] == ["link", "ner"]
    hop1 = [((f"P{i}", 0), 1.0 - i / 10) for i in range(5)]
    chain, missing = H.assemble_chain(hop1, [(("Q", 0), 0.8, "e"), (("Q", 1), 0.3, "e")], threshold=0.5, budget=5, hop2_k=2)
    assert [h for h, *_ in chain] == [1, 1, 1, 1, 2] and not missing
    chain, missing = H.assemble_chain(hop1, [(("Q", 0), 0.4, "e")], threshold=0.5, budget=5, hop2_k=2)
    assert len(chain) == 5 and missing


def test_spans_pages():
    assert H.spans_pages({"evidence": [[[0, 0, "A", 0], [0, 0, "B", 3]]]})
    assert not H.spans_pages({"evidence": [[[0, 0, "A", 0]], [[0, 0, "B", 1]]]})


# ----------------------------------------------------------------- Prompt 12: NLI helpers
def test_label_columns_follow_id2label():
    assert E.label_columns({0: "CONTRADICTION", 1: "NEUTRAL", 2: "ENTAILMENT"}) == [2, 0, 1]      # roberta-large-mnli
    assert E.label_columns({0: "contradiction", 1: "entailment", 2: "neutral"}) == [1, 0, 2]      # nli-deberta-v3-base
    with pytest.raises(AssertionError):
        E.label_columns({0: "LABEL_0", 1: "LABEL_1", 2: "LABEL_2"})


def test_aggregation_rule_and_threshold_tuning():
    import pandas as pd

    claims = [{"id": i} for i in range(4)]
    table = pd.DataFrame({"id": [0, 0, 1, 2], "p_e": [0.9, 0.1, 0.2, 0.3], "p_c": [0.05, 0.1, 0.7, 0.2], "p_n": [0.05, 0.8, 0.1, 0.5]})
    agg = E.aggregate(table, claims)
    assert agg["max_pe"].tolist() == [0.9, 0.2, 0.3, 0.0] and agg["min_pn"].tolist() == [0.05, 0.1, 0.5, 1.0]
    assert np.allclose(agg["P"][3], [0, 0, 1]) and np.allclose(agg["P"].sum(1), 1)
    assert E.rule_labels(agg["max_pe"], agg["max_pc"], 0.5, 0.5).tolist() == [0, 1, 2, 2]
    te, tc, f = E.tune_thresholds(agg["max_pe"], agg["max_pc"], np.array([0, 1, 2, 2]), [0.25, 0.5, 0.75])
    assert f == 1.0 and te >= 0.3 and tc >= 0.2


def test_oracle_evidence_for_nei_is_sampled_from_nearest_page():
    texts = {("A", 0): "", ("A", 2): "", ("B", 0): "", ("B", 1): "", ("B", 4): ""}
    pages = {"A": [0, 2], "B": [0, 1, 4]}
    claims = [{"id": 1, "label": "SUPPORTS", "evidence": [[[0, 0, "A", 2]], [[0, 0, "A", 2], [0, 0, "A", 0]]]},
              {"id": 2, "label": "NOT ENOUGH INFO", "evidence": [[[0, None, None, None]]]}]
    ev = E.oracle_evidence(claims, texts, pages, {2: [("B", 4, 3.0)]}, k=5, nei_sentences=2, seed=42)
    assert ev[1] == [("A", 2), ("A", 0)]
    assert len(ev[2]) == 2 and all(p == "B" for p, _ in ev[2])
    assert ev == E.oracle_evidence(claims, texts, pages, {2: [("B", 4, 3.0)]}, k=5, nei_sentences=2, seed=42)


# ----------------------------------------------------------------- Prompt 13: calibration
def test_temperature_scaling_recovers_known_temperature():
    rng = np.random.default_rng(0)
    z_true = rng.normal(size=(5000, 3)) * 2
    p = E.softmax(z_true)
    y = np.array([rng.choice(3, p=row) for row in p])
    T = K.fit_temperature(z_true * 3.0, y)            # logits 3x too sharp -> T should be about 3
    assert 2.6 < T < 3.4
    assert K.nll(z_true * 3.0, y, T) < K.nll(z_true * 3.0, y, 1.0)


def test_abstention_and_tau_tuning():
    labels, conf, gold = np.array([0, 1, 0, 1]), np.array([0.95, 0.4, 0.35, 0.9]), np.array([0, 2, 2, 1])
    assert K.abstain(labels, conf, 0.5).tolist() == [0, 2, 2, 1]
    tau, curve = K.tune_tau(labels, conf, gold, [0.3, 0.5, 0.7])
    assert tau == 0.5 and curve.loc[curve["tau"] == 0.5, "coverage"].item() == 0.5


def test_config_base_inheritance(tmp_path):
    (tmp_path / "base.yaml").write_text("seed: 42\npipeline: {calibration: false, grid: [1, 2]}\n")
    (tmp_path / "row.yaml").write_text("base: base.yaml\npipeline: {name: X, calibration: true}\n")
    assert load_config(tmp_path / "row.yaml") == {"seed": 42, "pipeline": {"calibration": True, "grid": [1, 2], "name": "X"}}


def test_resolve_run_auto(tmp_path):
    cfg = {"paths": {"outputs_dir": str(tmp_path / "outputs"), "search_dirs": []}}
    assert C.resolve_run(cfg, "auto") == "hybrid" and C.resolve_run(cfg, "bm25") == "bm25"
    (tmp_path / "outputs" / "rerank").mkdir(parents=True)
    (tmp_path / "outputs" / "rerank" / "metrics.json").write_text(json.dumps({"keep": True}))
    assert C.resolve_run(cfg, "auto") == "rerank" and C.resolve_run(cfg, "auto_alt") == "hybrid"


# ----------------------------------------------------------------- end to end: Prompts 8 -> 14
def test_phase3_end_to_end(tiny, monkeypatch):
    from src import pipeline as PL
    from src.retrieval import dense as D
    from src.retrieval import finetune as F
    from src.retrieval import rerank as R
    from src.verify import nli as N
    from tests.fakes import overlap

    root, write_cfg = tiny
    out = root / "outputs"
    splits = root / "data" / "splits"
    fake_loader = lambda name, fp16: (FakeST(), "cpu")  # noqa: E731
    monkeypatch.setattr(D, "load_bi_encoder", fake_loader)
    monkeypatch.setattr(F, "load_bi_encoder", fake_loader)
    model_dir = root / "models" / "minilm_ft"

    def fake_train(rows, fc, seed, out_dir, run_dir, log):
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "config.json").write_text("{}")
        return [{"loss": 1.0}]

    monkeypatch.setattr(F, "train_model", fake_train)
    for cls_mod in (N, K, PL):
        monkeypatch.setattr(cls_mod, "NLIScorer", FakeNLIScorer)
    monkeypatch.setattr(R, "cross_encoder_scorer", lambda cfg: (lambda a, b: np.array([overlap(x, y) for x, y in zip(a, b)]), "cpu"))

    # BM25 runs (Prompt 5 output): wrong page first, the right sentence second.
    for sp, name in (("train", "train"), ("val", "val"), ("test", "test_full")):
        claims = [json.loads(l) for l in open(splits / f"{name}.jsonl")]
        C.write_run({c["id"]: [(f"Page_{(c['id'] + 1) % 60}", 0, 3.0), (f"Page_{c['id']}", 1, 2.0), (f"Page_{c['id']}", 0, 1.0)] for c in claims},
                    out / "bm25" / f"run_{sp}.jsonl")
    # A small HotpotQA file (4 questions, 2 gold + 1 decoy paragraph each).
    hot = [{"_id": str(i), "question": f"Which band did person{i} join after school{i}?", "answer": "x",
            "supporting_facts": [[f"Person{i}", 0], [f"Band{i}", 0]],
            "context": [[f"Person{i}", [f"Person{i} studied at school{i} and joined Band{i} ."]],
                        [f"Band{i}", [f"Band{i} is a rock band formed in 1990 ."]],
                        [f"Decoy{i}", ["An unrelated sentence about weather ."]]]} for i in range(4)]
    (root / "data" / "raw").mkdir()
    (root / "data" / "raw" / "hotpot_dev_distractor_v1.json").write_text(json.dumps(hot))

    dense_eval = {"model_name": "x", "text_format": "title: sentence", "batch_size": 8, "fp16": False, "shard_size": 100,
                  "top_k": 20, "latency_queries": 3, "save_faiss_index": False, "compare_with": "bm25"}
    ft_cfg = write_cfg("ft", {"run_name": "finetune", "seed": 42, "test_subset": "full", "dense_eval": dense_eval,
                              "finetune": {"base_model": "fake", "bm25_run": "bm25", "baseline_run": "dense_offshelf",
                                           "output_model_dir": str(model_dir), "dense_run": "dense_ft", "batch_size": 8,
                                           "learning_rate": 2e-5, "epochs": 1, "warmup_ratio": 0.1, "fp16": False,
                                           "max_seq_length": 64, "logging_steps": 5, "false_negative_examples": 3}})
    assert F.main(["--config", ft_cfg]) == 0
    fm = json.loads((out / "finetune" / "metrics.json").read_text())
    assert fm["n_rows"] == 20 and (out / "dense_ft" / "run_test.jsonl").exists()

    hy_cfg = write_cfg("hy", {"run_name": "hybrid", "seed": 42, "test_subset": "full",
                              "hybrid": {"sparse_run": "bm25", "dense_run": "dense_ft", "noise_dense_run": "dense_ft_seed43", "top_k": 20,
                                         "rrf_k": 60, "alphas": [0.2, 0.5, 0.8], "tune_metric": ["sentence", "recall@10"], "ks": [5, 10, 20]}})
    assert Y.main(["--config", hy_cfg]) == 0
    hm = json.loads((out / "hybrid" / "metrics.json").read_text())
    assert hm["results"]["Hybrid RRF"]["test"]["sentence"]["recall@5"] == 1.0

    rr_cfg = write_cfg("rr", {"run_name": "rerank", "seed": 42, "test_subset": "full",
                              "rerank": {"model": "fake", "input_run": "hybrid", "top_n": 10, "batch_size": 8, "max_length": 64,
                                         "fp16": False, "ks": [5, 10], "latency_queries": 3, "default_noise": 0.005}})
    assert R.main(["--config", rr_cfg]) == 0
    assert "keep" in json.loads((out / "rerank" / "metrics.json").read_text())

    thresholds = [0.2, 0.5, 0.8, 1.01]
    ch_cfg = write_cfg("ch", {"run_name": "chain", "seed": 42, "test_subset": "full",
                              "chain": {"hop1_run": "auto", "dense_model": str(model_dir), "dense_run": "dense_ft", "budget": 3,
                                        "bridge_from": 2, "max_bridges": 2, "hop2_k": 1, "search_k": 5, "thresholds": thresholds,
                                        "spacy_model": "en_core_web_sm",
                                        "hotpot": {"budget": 2, "hop2_k": 1, "bridge_from": 1, "max_bridges": 1, "tune_questions": 2,
                                                   "thresholds": thresholds, "max_questions": None}}})
    assert H.main(["--config", ch_cfg]) == 0
    cm = json.loads((out / "chain" / "metrics.json").read_text())
    rows = [json.loads(l) for l in open(out / "chain" / "chains_test.jsonl")]
    assert len(rows) == 15 and all(len(r["chain"]) == 3 for r in rows)
    assert cm["hotpot"]["n_eval"] == 2 and 0 <= cm["hotpot"]["two_hop"]["sf_f1"] <= 1

    grid = [0.2, 0.4, 0.6, 0.8]
    nli_cfg = write_cfg("nli", {"run_name": "verify", "seed": 42, "test_subset": "full",
                                "nli": {"main_model": "fake/nli-main", "compare_model": "fake/nli-other", "compare_split": "test_5k",
                                        "fp16": False, "max_length": 64, "batch_size": 8, "threshold_grid": grid,
                                        "sources": {"retrieved": {"type": "run", "run": "auto", "k": 3},
                                                    "chain": {"type": "chain", "run": "chain"},
                                                    "oracle": {"type": "gold", "nearest_run": "bm25", "nei_sentences": 2, "k": 3}},
                                        "compare_sources": ["retrieved", "oracle"], "concat_sources": ["retrieved"]}})
    assert N.main(["--config", nli_cfg]) == 0
    nm = json.loads((out / "verify" / "metrics.json").read_text())
    assert nm["results"]["main|oracle|per_sentence"]["test"]["accuracy"] > 0.3
    assert list((out / "verify").glob("probs_nli_main_oracle_test_full.parquet"))
    assert (out / "nli_cache" / "nli_main.parquet").exists()

    cal_cfg = write_cfg("cal", {"run_name": "calibration", "seed": 42, "test_subset": "full",
                                "calibration": {"model": "fake/nli-main", "fp16": False, "max_length": 64, "batch_size": 8,
                                                "evidence": {"type": "chain", "run": "chain"}, "threshold_grid": grid,
                                                "tau_min": 0.3, "tau_max": 0.9, "tau_step": 0.1}})
    assert K.main(["--config", cal_cfg]) == 0
    km = json.loads((out / "calibration" / "metrics.json").read_text())
    assert km["T"] > 0 and (out / "calibration" / "reliability_test.png").exists()

    base = {"seed": 42, "test_subset": "full", "nli": {"model": "fake/nli-main", "fp16": False, "max_length": 64, "batch_size": 8},
            "pipeline": {"calibration": False, "missing_link_nei": "tune", "threshold_grid": grid, "tau_min": 0.3, "tau_max": 0.9, "tau_step": 0.1}}
    rows_cfg = []
    for name, ev, calib in (("NLI on gold evidence", {"type": "gold", "nearest_run": "bm25", "nei_sentences": 2, "k": 3}, False),
                            ("BM25 + NLI", {"type": "run", "run": "bm25", "k": 3}, False),
                            ("Hybrid + 2-hop + NLI", {"type": "chain", "run": "chain"}, False),
                            ("+ calibration", {"type": "chain", "run": "chain"}, True)):
        rows_cfg.append({"config": write_cfg(f"row{len(rows_cfg)}", {**base, "run_name": "x", "pipeline": {**base["pipeline"], "name": name,
                                                                                                            "evidence": ev, "calibration": calib}})})
    rows_cfg.append({"config": rows_cfg[1]["config"], "needs_output": "does_not_exist"})
    base_cfg = write_cfg("pbase", {**base, "tables": {"verification": rows_cfg}})
    assert PL.main(["--table", "verification", "--base", base_cfg]) == 0
    import pandas as pd

    table = pd.read_csv(out / "tables" / "verification.csv")
    assert list(table["name"]) == ["NLI on gold evidence", "BM25 + NLI", "Hybrid + 2-hop + NLI", "+ calibration"]
    assert table[["label_accuracy", "macro_f1", "fever_score", "sf_f1", "ece"]].apply(lambda c: c.between(0, 1)).all().all()
    gold_row = table.iloc[0]
    assert gold_row["sf_f1"] > 0.99 and gold_row["fever_score"] <= gold_row["label_accuracy"]
    preds = [json.loads(l) for l in open(out / "pipeline" / "bm25_nli" / "predictions_test.jsonl")]
    assert len(preds) == 15 and set(preds[0]) == {"id", "predicted_label", "predicted_evidence"}
