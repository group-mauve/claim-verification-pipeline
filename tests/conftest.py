"""Shared test setup.

1. Make the official FEVER scorer importable without relying on PYTHONPATH. The notebook clones
   sheffieldnlp/fever-scorer to /kaggle/working/fever-scorer; a kernel restart clears PYTHONPATH, which
   used to make the official-scorer test skip. Set FEVER_SCORER_DIR to use another clone location.
2. The `tiny` fixture: a 60-page synthetic corpus with claims, splits and a config writer, used by the
   end-to-end tests of the retrieval, chain, verification and pipeline scripts.
"""
import glob
import json
import os
import sys

import pandas as pd
import pytest
import yaml

for root in filter(None, [os.environ.get("FEVER_SCORER_DIR"), "/kaggle/working/fever-scorer"]):
    hits = glob.glob(os.path.join(root, "**", "fever", "scorer.py"), recursive=True)
    if hits:
        sys.path.insert(0, os.path.dirname(os.path.dirname(hits[0])))
        break

LABELS = ["SUPPORTS", "REFUTES", "NOT ENOUGH INFO"]


@pytest.fixture()
def tiny(tmp_path):
    """60 pages x 3 sentences; claim i shares rare words with sentence 1 of Page_i.
    Sentence 0 of each page links to the next page (for the two-hop chain)."""
    rows, claims = [], []
    for i in range(60):
        for s in range(3):
            rows.append({"page_id": f"Page_{i}", "sent_id": s, "text": f"Page {i} fact {s} about topic{i} and word{i * 3 + s} .",
                         "links": [f"Page_{(i + 1) % 60}"] if s == 0 else []})
    corpus_dir, splits_dir = tmp_path / "data" / "corpus", tmp_path / "data" / "splits"
    corpus_dir.mkdir(parents=True)
    splits_dir.mkdir(parents=True)
    pd.DataFrame(rows).to_parquet(corpus_dir / "sentences.parquet", index=False)
    for i in range(60):
        lab = LABELS[i % 3]
        ev = [[[i, None, None, None]]] if lab == LABELS[2] else [[[i, i, f"Page_{i}", 1]]]
        claims.append({"id": i, "label": lab, "claim": f"topic{i} word{i * 3 + 1}", "evidence": ev})
    for name, part in (("train", claims[:30]), ("val", claims[30:45]), ("test_full", claims[45:]), ("test_5k", claims[45:])):
        with open(splits_dir / f"{name}.jsonl", "w") as f:
            f.writelines(json.dumps(c) + "\n" for c in part)

    def write_cfg(name, body):
        body = {**body, "paths": {"outputs_dir": str(tmp_path / "outputs"), "search_dirs": [str(tmp_path / "data"), str(tmp_path / "outputs")]}}
        p = tmp_path / f"{name}.yaml"
        p.write_text(yaml.safe_dump(body))
        return str(p)

    return tmp_path, write_cfg
