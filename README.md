# Dense Retrieval, Chain-of-Evidence & Entailment Verification

Given a claim, the system retrieves evidence sentences from Wikipedia, chains evidence over at most
two hops, and labels the claim SUPPORTS / REFUTES / NOT ENOUGH INFO with the (page, sentence id) of
each evidence sentence it used. Data: FEVER 1.0 and HotpotQA (distractor dev). Runs on Kaggle.

## Status

| Phase | Prompts | Status |
|---|---|---|
| 1 · Data and evaluation harness (M1) | 1–4 | done on Kaggle: all counts verified, corpus 50,000 pages / 308,328 sentences, 100% S/R evidence coverage |
| 2 · Baselines (M1, M2) | 5–7 | done on Kaggle (results below) |
| 3 · Pipeline v1 (M2, M3, M4; M1 integrates) | 8–14 | code written; tests pass locally (models replaced by fakes); **not yet run on Kaggle** |
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
src/retrieval/finetune.py  Prompt 8: MiniLM fine-tuned with in-batch + BM25 hard negatives
src/retrieval/hybrid.py    Prompt 9: Reciprocal Rank Fusion and weighted score fusion
src/retrieval/rerank.py    Prompt 10: optional cross-encoder rerank, kept only if it beats seed noise
src/chain/two_hop.py       Prompt 11: two-hop evidence chain (FEVER multi-page claims + HotpotQA)
src/cross_encoder.py       scores (text, text) pairs with a Hugging Face classifier (rerank, NLI)
src/verify/evidence.py     evidence sources, NLI logit cache, aggregation rule, threshold tuning
src/verify/nli.py          Prompt 12: NLI verifier, model comparison, concatenated premise, oracle
src/verify/calibrate.py    Prompt 13: temperature scaling, ECE, reliability diagram, NEI when unsure
src/pipeline.py            Prompt 14: end-to-end pipelines, official FEVER scorer, results table
src/eval/metrics.py        Prompt 4: Recall@K, MRR, nDCG@K, supporting-fact P/R/F1,
                           accuracy, macro-F1, ECE, FEVER score
tests/                     pytest unit tests (slide examples, official-scorer comparison, data helpers)
notebooks/01_data.ipynb    runs Phase 1 on Kaggle and packages the outputs as a Dataset
notebooks/02_bm25.ipynb    Prompt 5 on a CPU session
notebooks/03_dense.ipynb   Prompts 6-7 on a GPU session
notebooks/04_finetune_hybrid.ipynb  Prompts 8-10 (GPU)
notebooks/05_chain.ipynb   Prompt 11 (GPU)
notebooks/06_verify_pipeline.ipynb  Prompts 12-14 (GPU)
configs/pipeline_*.yaml    one file per pipeline row; pipeline_base.yaml lists the table rows
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

### Phase 2 results (Kaggle)

Test = all 19,998 shared-task dev claims; retrieval metrics over the 13,332 SUPPORTS/REFUTES claims.

| Retriever | Sentence R@10 | Sentence MRR | Page R@10 | Page MRR | ms/query | Index |
|---|---|---|---|---|---|---|
| BM25 (Pyserini, k1 = 0.6, b = 0.4 tuned on val) | 0.8267 | 0.7220 | 0.9292 | 0.7973 | 3.85 (CPU) | 24 MB Lucene |
| MiniLM off the shelf (FAISS Flat) | 0.8608 | 0.7166 | 0.9475 | 0.8619 | 54.3 (T4 encode + CPU search) | 0.441 GB |

DPR vs MiniLM on the 5k subset (pool of 296,051 sentences: evidence pages + BM25 top-100 pages): MiniLM sentence
R@10 0.855 / MRR 0.720 against DPR 0.800 / 0.625, at half the index size (384 vs 768 dimensions) and about half
the query time. DPR has a slightly higher page MRR (0.871 vs 0.858): it finds the right page but ranks the right
sentence lower. We keep MiniLM. The BM25 grid favoured the smallest b (0.4): sentences are short and similar in
length, so length normalisation matters little.

## Running Phase 3

Each notebook clones this repository, runs the unit tests, then the scripts. Attach the inputs listed in its first
cell, run it, then **Save Version → Save & Run All (Commit)** so the next notebook can attach its output.

| Notebook | Session | Inputs | Produces |
|---|---|---|---|
| `04_finetune_hybrid.ipynb` | GPU T4 | `fever-corpus-v1`, output of 02 (optional: output of 03) | `models/minilm_ft/`, `outputs/dense_ft/`, `outputs/finetune/`, `outputs/hybrid/`, `outputs/rerank/` |
| `05_chain.ipynb` | GPU T4 | `fever-corpus-v1`, outputs of 02 and 04 | `outputs/chain/` (chains_val/test.jsonl, metrics.json, examples.json) |
| `06_verify_pipeline.ipynb` | GPU T4 | `fever-corpus-v1`, outputs of 02, 04 and 05 | `outputs/verify/`, `outputs/calibration/`, `outputs/nli_cache/`, `outputs/pipeline/`, `outputs/tables/verification.{csv,md}` |

The same steps from a shell, in order:
```
python -m src.retrieval.finetune --config configs/finetune.yaml            # Prompt 8 (seed 42)
python -m src.retrieval.finetune --config configs/finetune.yaml --seed 43  # second seed: noise estimate
python -m src.retrieval.hybrid   --config configs/hybrid.yaml              # Prompt 9
python -m src.retrieval.rerank   --config configs/rerank.yaml              # Prompt 10
python -m src.chain.two_hop      --config configs/chain.yaml               # Prompt 11
python -m src.verify.nli         --config configs/nli.yaml                 # Prompt 12
python -m src.verify.calibrate   --config configs/calibrate.yaml           # Prompt 13
python -m src.pipeline --table verification                                # Prompt 14: outputs/tables/verification.md
```

Design points to explain in the viva:
- **Every tuned number comes from validation.** BM25 k1/b, the fusion alpha, the rerank decision, the hop-2
  threshold, theta_e/theta_c, T and tau are all fitted on the 2,000 validation claims and applied once to test.
- **Rerank decision rule.** "Noise between two seeds" = the MRR difference between two hybrid runs whose dense
  models were fine-tuned with seeds 42 and 43. The reranker is kept only if its validation MRR gain is larger.
  Later configs say `run: auto`, which means "reranked hybrid if kept, else hybrid"; an extra table row uses the
  other choice, so the decision's effect on FEVER score is visible too.
- **Chain budget.** A chain has 5 sentences like a plain top 5: up to 2 hop-2 sentences replace the last hop-1
  sentences. The FEVER scorer reads only the first 5 predicted sentences, and the equal budget makes
  "1 hop vs 2 hops" a fair comparison. Hop 2 uses the dense retriever because its cosine score has a fixed
  scale, so one threshold means the same for every claim (RRF scores only encode ranks).
- **Claim-level probabilities.** The verifier scores each evidence sentence separately. A claim's distribution
  is normalise(max p(E), max p(C), min p(N)): the most entailing, the most contradicting and the least neutral
  sentence. The aggregation rule decides the label; temperature scaling only changes the confidence.
- **NLI cache.** Logits are stored per (claim, sentence) in `outputs/nli_cache/<model>.parquet`; every pipeline
  that shows the same sentence to the same claim reuses them.

## NLI checkpoints and FEVER leakage (Prompt 12)

| Checkpoint | Fine-tuned on (model card) | Trained on FEVER? |
|---|---|---|
| `cross-encoder/nli-deberta-v3-base` (main) | SNLI + MultiNLI | No |
| `roberta-large-mnli` (comparison) | MultiNLI | No |
| `cross-encoder/ms-marco-MiniLM-L-6-v2` (reranker) | MS MARCO passage ranking | No |

Neither NLI model saw FEVER claims or labels, so there is no label leakage. Their pre-training text includes
English Wikipedia, so the models may have seen the evidence sentences themselves (not the claims). Checkpoints
such as `ynie/roberta-large-snli_mnli_fever_anli` were trained on FEVER-NLI and are avoided for that reason.

## Reproducibility facts

- Seed: 42 everywhere.
- Splits: train 30,000 (16,508 S / 6,141 R / 7,351 NEI), val 2,000 (1,101 / 409 / 490),
  test = FEVER shared-task dev 19,998 (6,666 each) or the balanced 5k subset (1,667 / 1,667 / 1,666).
- Corpus: 50,000 pages (9,197 evidence + 40,803 BM25 decoy + 0 random), 308,328 sentences; `corpus/corpus_stats.json`.
- Model checkpoints: `sentence-transformers/all-MiniLM-L6-v2`, `facebook/dpr-ctx_encoder-single-nq-base`,
  `facebook/dpr-question_encoder-single-nq-base` (Phase 2); `cross-encoder/ms-marco-MiniLM-L-6-v2`,
  `cross-encoder/nli-deberta-v3-base`, `roberta-large-mnli`, spaCy `en_core_web_sm` (Phase 3).
- Fine-tuning: 1 epoch, batch 128, learning rate 2e-5, 10% warmup, fp16, max 128 tokens, seed 42 (and 43 for noise).

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

## References

- Thorne et al., 2018. FEVER: a Large-scale Dataset for Fact Extraction and VERification. NAACL.
- Yang et al., 2018. HotpotQA: A Dataset for Diverse, Explainable Multi-hop Question Answering. EMNLP.
- Robertson & Zaragoza, 2009. The Probabilistic Relevance Framework: BM25 and Beyond.
- Official FEVER scorer: https://github.com/sheffieldnlp/fever-scorer
- Karpukhin et al., 2020. Dense Passage Retrieval for Open-Domain Question Answering. EMNLP.
- Reimers & Gurevych, 2019. Sentence-BERT. EMNLP. (sentence-transformers, MultipleNegativesRankingLoss)
- Cormack, Clarke & Buettcher, 2009. Reciprocal Rank Fusion Outperforms Condorcet and Individual Rank Learning Methods. SIGIR.
- Nogueira & Cho, 2019. Passage Re-ranking with BERT. arXiv. (cross-encoder reranking)
- He, Gao & Chen, 2021. DeBERTaV3. arXiv. Liu et al., 2019. RoBERTa. arXiv. Williams et al., 2018. MultiNLI. NAACL.
- Guo et al., 2017. On Calibration of Modern Neural Networks. ICML. (temperature scaling, ECE)
- Libraries: NumPy, pandas, PyArrow, SciPy, scikit-learn, PyYAML, pytest, Pyserini, FAISS, PyTorch,
  Hugging Face Transformers and Datasets, sentence-transformers, spaCy, matplotlib.
