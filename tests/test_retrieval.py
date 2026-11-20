"""Tests for src/retrieval (Prompts 5-7) on a tiny synthetic corpus. No GPU, Java or model downloads:
neural encoders are replaced by a deterministic hashing encoder, so the full scripts still run."""
import json

import numpy as np
import pytest

from src.retrieval import bm25 as B
from src.retrieval import common as C
from src.retrieval import dense as D
from tests.fakes import FakeST

# ----------------------------------------------------------------- small helpers
def test_docid_roundtrip_and_doc_text():
    assert B.parse_docid(B.make_docid("Telemundo", 4)) == ("Telemundo", 4)
    assert B.parse_docid("Star_Wars-COLON-_Episode:12") == ("Star_Wars-COLON-_Episode", 12)
    assert C.doc_text("Soul_Food_-LRB-film-RRB-", "It is a 1997 -LRB- American -RRB- film .") == \
        "Soul Food (film): It is a 1997 ( American ) film ."


def test_run_file_roundtrip(tmp_path):
    run = {1: [("A", 0, 2.5), ("B", 3, 1.0)], 2: []}
    C.write_run(run, tmp_path / "r.jsonl")
    assert C.read_run(tmp_path / "r.jsonl") == run


def test_exact_search_matches_brute_force():
    rng = np.random.default_rng(0)
    docs, qs = rng.normal(size=(500, 16)).astype(np.float32), rng.normal(size=(7, 16)).astype(np.float32)
    scores, idx = C.exact_search(docs, qs, 5)
    brute = np.argsort(-(qs @ docs.T), axis=1)[:, :5]
    assert (idx == brute).all()
    assert np.allclose(scores, np.take_along_axis(qs @ docs.T, brute, axis=1), atol=1e-5)


def test_biggest_wins():
    a, b = {1: 1.0, 2: 0.5, 3: 0.2, 4: 0.0}, {1: 0.1, 2: 0.5, 3: 1.0, 4: 0.0}
    assert C.biggest_wins(a, b) == [1]
    assert C.biggest_wins(b, a) == [3]


def test_find_output_with_subfolder(tmp_path):
    f = tmp_path / "input" / "nb-output" / "project" / "outputs" / "dense_offshelf" / "embeddings" / "meta.json"
    f.parent.mkdir(parents=True)
    f.write_text("{}")
    cfg = {"paths": {"outputs_dir": str(tmp_path / "none"), "search_dirs": [str(tmp_path / "input")]}}
    assert C.find_output(cfg, "dense_offshelf", "embeddings/meta.json") == f
    assert C.find_output(cfg, "bm25", "embeddings/meta.json") is None


def test_locate_file_follows_symlinked_inputs(tmp_path):
    import os

    from src.utils import describe_inputs, locate_file

    real = tmp_path / "datasets" / "owner" / "fever-corpus-v1" / "corpus"
    real.mkdir(parents=True)
    (real / "sentences.parquet").write_text("x")
    inputs = tmp_path / "input"
    inputs.mkdir()
    try:
        os.symlink(tmp_path / "datasets" / "owner" / "fever-corpus-v1", inputs / "fever-corpus-v1", target_is_directory=True)
    except (OSError, NotImplementedError):
        pytest.skip("this OS/user cannot create symlinks")
    assert locate_file("sentences.parquet", [str(inputs)]) == inputs / "fever-corpus-v1" / "corpus" / "sentences.parquet"
    assert "corpus" in describe_inputs(str(inputs))


def test_encode_in_shards_resumes(tmp_path):
    calls = []

    def enc(texts):
        calls.append(len(texts))
        return np.ones((len(texts), 4), np.float32)

    import logging

    log = logging.getLogger("t")
    texts = [f"t{i}" for i in range(25)]
    v1 = D.encode_in_shards(texts, enc, tmp_path, 10, {"model": "m"}, log)
    assert v1.shape == (25, 4) and calls == [10, 10, 5]
    (tmp_path / "emb_002.npy").unlink()                       # simulate a timeout during the last shard
    D.encode_in_shards(texts, enc, tmp_path, 10, {"model": "m"}, log)
    assert calls == [10, 10, 5, 5]                            # only the missing shard is re-encoded
    D.encode_in_shards(texts, enc, tmp_path, 10, {"model": "other"}, log)
    assert calls == [10, 10, 5, 5, 10, 10, 5]                 # other model -> everything re-encoded


def test_rank_bm25_backend():
    pytest.importorskip("rank_bm25")
    s = B.RankBM25(["A:0", "B:0", "C:0"], ["Telemundo Spanish television network", "Paris capital France", "Damon Albarn English musician"])
    s.set_bm25(0.9, 0.4)
    assert s.search("capital of France", 2)[0].docid == "B:0"


# ----------------------------------------------------------------- end-to-end on a tiny corpus
# The `tiny` fixture is in conftest.py; FakeST in tests/fakes.py.


def test_bm25_quick_end_to_end(tiny):
    pytest.importorskip("rank_bm25")
    root, write_cfg = tiny
    cfg = write_cfg("bm25", {"run_name": "bm25", "seed": 42, "test_subset": "full",
                             "bm25": {"backend": "pyserini", "threads": 1, "top_k": 10, "train_top_k": 5,
                                      "grid": {"k1": [0.9, 1.2], "b": [0.4]}, "tune_metric": ["sentence", "recall@10"], "latency_queries": 5},
                             "quick_test": {"n_docs": 100, "n_claims": 15}})
    assert B.main(["--config", cfg, "--quick"]) == 0
    m = json.loads((root / "outputs" / "bm25_quick" / "metrics.json").read_text())
    assert m["val"]["page"]["recall@5"] == 1.0          # claims share rare words with their evidence page


def test_dense_and_dpr_end_to_end(tiny, monkeypatch):
    root, write_cfg = tiny
    monkeypatch.setattr(D, "load_bi_encoder", lambda name, fp16: (FakeST(), "cpu"))
    # A fake BM25 run, so the dense script prints its side-by-side comparison.
    out = root / "outputs" / "bm25"
    test = [json.loads(l) for l in open(root / "data" / "splits" / "test_full.jsonl")]
    C.write_run({c["id"]: [(f"Page_{(c['id'] + 1) % 60}", 0, 1.0), (f"Page_{c['id']}", 1, 0.5)] for c in test}, out / "run_test.jsonl")

    dense_cfg = write_cfg("dense", {"run_name": "dense_offshelf", "seed": 42, "test_subset": "full",
                                    "dense": {"model_name": "fake", "text_format": "title: sentence", "batch_size": 8, "fp16": False,
                                              "shard_size": 50, "top_k": 10, "latency_queries": 3, "save_faiss_index": False,
                                              "compare_with": "bm25"}})
    assert D.main(["--config", dense_cfg]) == 0
    dm = json.loads((root / "outputs" / "dense_offshelf" / "metrics.json").read_text())
    assert dm["test"]["n_claims"] == 10 and dm["test"]["page"]["recall@10"] > 0.5
    assert len(list((root / "outputs" / "dense_offshelf" / "embeddings").glob("emb_*.npy"))) == 4   # 180 sentences / 50
    assert (root / "outputs" / "dense_offshelf" / "examples.json").exists()

    from src.retrieval import dpr_compare as P

    fake = FakeST(32)
    monkeypatch.setattr(P, "load_bi_encoder", lambda name, fp16: (FakeST(), "cpu"))
    monkeypatch.setattr(P, "dpr_encoders", lambda cfg: (lambda pairs: fake.encode([p.replace("\t", " ") for p in pairs], False),
                                                        lambda qs: fake.encode(qs, False), "cpu"))
    dpr_cfg = write_cfg("dpr", {"run_name": "dpr_compare", "seed": 42,
                                "dpr": {"ctx_model": "x", "question_model": "y", "max_length": 32, "batch_size": 8, "fp16": False, "shard_size": 50},
                                "minilm": {"run_name": "dense_offshelf", "model_name": "fake"},
                                "pool_from": "bm25", "top_k": 10, "latency_queries": 3})
    assert P.main(["--config", dpr_cfg]) == 0
    pm = json.loads((root / "outputs" / "dpr_compare" / "metrics.json").read_text())
    assert set(pm["results"]) == {"minilm", "dpr"} and pm["pool_sentences"] > 0
    assert "DPR" in (root / "outputs" / "dpr_compare" / "comparison.md").read_text()
