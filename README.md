# Dense Retrieval, Chain-of-Evidence & Entailment Verification

Given a claim, the system retrieves evidence sentences from Wikipedia, chains evidence over at most
two hops, and labels the claim SUPPORTS / REFUTES / NOT ENOUGH INFO with the (page, sentence id) of
each evidence sentence it used. Data: FEVER 1.0 and HotpotQA (distractor dev). Runs on Kaggle.

## Status

| Phase | Prompts | Status |
|---|---|---|
| 1 · Data and evaluation harness (M1) | 1–4 | done on Kaggle: all counts verified, corpus 50,000 pages / 308,328 sentences, 100% S/R evidence coverage |
| 2 · Baselines (M1, M2) | 5–7 | code written; tests pass locally (encoders mocked); **not yet run on Kaggle** |
| 3 · Pipeline v1 | 8–14 | not started |
| 4 · Improvements, ablations, errors | 15–18 | not started |
| 5 · Demo, report, viva | 19–21 | not started |

## Repository layout

```
configs/data.yaml          paths, URLs, expected counts, split sizes, corpus settings (seed 42)
src/utils.py               config loading, seeds, run logging, JSONL I/O, FEVER evidence helpers
src/data/download.py       Prompt 1: download + parse wiki dump to Parquet + verify counts
src/data/splits.py         Prompt 2: train 30k / val 2k / test_full / test_5k
src/data/page_bm25.py      sparse-matrix BM25 used to find decoy pages
src/data/build_corpus.py   Prompt 3: restricted ~50k-page corpus
src/retrieval/common.py    shared run-file format, evaluation, latency, exact vector search
src/retrieval/bm25.py      Prompt 5: Pyserini BM25 (+ rank_bm25 quick test), k1/b tuning
src/retrieval/dense.py     Prompt 6: MiniLM bi-encoder, sharded encoding, FAISS IndexFlatIP
src/retrieval/dpr_compare.py Prompt 7: DPR checkpoint vs MiniLM on test_5k
src/eval/metrics.py        Prompt 4: Recall@K, MRR, nDCG@K, supporting-fact P/R/F1,
                           accuracy, macro-F1, ECE, FEVER score
tests/                     pytest unit tests (slide examples, official-scorer comparison, data helpers)
notebooks/01_data.ipynb    runs Phase 1 on Kaggle and packages the outputs as a Dataset
notebooks/02_bm25.ipynb    Prompt 5 on a CPU session
notebooks/03_dense.ipynb   Prompts 6-7 on a GPU session
data/README.md             data sources, file layout, Kaggle Dataset upload steps
outputs/<run_name>/<step>/ run.log, config_used.yaml, metrics.json for every script run
```

## Running Phase 1

On Kaggle, follow `notebooks/01_data.ipynb`. Locally or in any shell, from the repository root:

```
pip install -r requirements.txt
git clone https://github.com/sheffieldnlp/fever-scorer   # then add its src/ folder to PYTHONPATH (pip install fails)
python -m pytest -q                                   # metric tests
python -m src.data.download     --config configs/data.yaml
python -m src.data.splits       --config configs/data.yaml
python -m src.data.build_corpus --config configs/data.yaml
```

Every script reads the YAML config, fixes seed 42, logs to `outputs/data_v1/<step>/run.log` and
writes `metrics.json` there. Each one stops with an error if a sanity check fails.

## Running Phase 2

| Notebook | Session | Inputs | Produces |
|---|---|---|---|
| `02_bm25.ipynb` | CPU, Internet on | `fever-corpus-v1` | `outputs/bm25/`: run_test / run_val / run_train.jsonl, metrics.json, lucene_index/ |
| `03_dense.ipynb` | GPU T4, Internet on | `fever-corpus-v1` + output of 02 | `outputs/dense_offshelf/` (embeddings, index.faiss, runs, examples.json), `outputs/dpr_compare/` (comparison.md) |

Run file format (all retrievers): one line per claim, `{"id": ..., "hits": [[page_id, sent_id, score], ...]}`,
best first. Retrieval metrics count only SUPPORTS/REFUTES claims (NEI claims have no evidence).
The BM25 val and train runs are saved now because Prompt 9 (fusion tuning) and Prompt 8 (hard negatives) need them.

## Reproducibility facts (fill in the corpus numbers after the Kaggle run)

- Seed: 42 everywhere.
- Splits: train 30,000 (16,508 S / 6,141 R / 7,351 NEI), val 2,000 (1,101 / 409 / 490),
  test = FEVER shared-task dev 19,998 (6,666 each) or the balanced 5k subset (1,667 / 1,667 / 1,666).
- Corpus: 50,000 pages (9,197 evidence + 40,803 BM25 decoy + 0 random), 308,328 sentences; `corpus/corpus_stats.json`.
- Model checkpoints: added from Phase 2.

## Design choice to explain in the viva

The decoy search in Prompt 3 uses `src/data/page_bm25.py`, a short BM25 written with scipy sparse
matrices, instead of Pyserini. Every page becomes a row of BM25 term weights, a claim becomes a 0/1
term vector, and one sparse matrix product scores all 5.4M pages at once. This avoids installing
Java just to build a temporary index. Pyserini remains the main sparse retriever in Prompt 5.

## Unicode page titles (important for everyone)

FEVER stores accented page titles in two Unicode forms (e.g. `García` as one character or as
`i` + combining accent), even within the claim files. All code converts titles with
`src.utils.canonical_page` (NFC). Always build page ids through `evidence_sets` / `evidence_pages`
or `canonical_page`, never from raw `e[2]`. When calling the official FEVER scorer (Prompt 14),
pass gold evidence with titles converted by `canonical_page`, because the scorer compares strings exactly.

## References (Phase 1)

- Thorne et al., 2018. FEVER: a Large-scale Dataset for Fact Extraction and VERification. NAACL.
- Yang et al., 2018. HotpotQA: A Dataset for Diverse, Explainable Multi-hop Question Answering. EMNLP.
- Robertson & Zaragoza, 2009. The Probabilistic Relevance Framework: BM25 and Beyond.
- Official FEVER scorer: https://github.com/sheffieldnlp/fever-scorer
- Libraries: NumPy, pandas, PyArrow, SciPy, scikit-learn, PyYAML, pytest.
