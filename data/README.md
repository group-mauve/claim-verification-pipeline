# Data

Produced by `src/data/download.py`, `src/data/splits.py` and `src/data/build_corpus.py` (config: `configs/data.yaml`, seed 42).

## Sources

| File | Source | Size | Verified count |
|---|---|---|---|
| `raw/train.jsonl` | FEVER 1.0 (Thorne et al., 2018), fever.ai | 145,449 claims | 80,035 S / 29,775 R / 35,639 NEI |
| `raw/shared_task_dev.jsonl` | FEVER 1.0 shared-task dev | 19,998 claims | 6,666 per label |
| `raw/wiki-pages.zip` | FEVER June 2017 Wikipedia dump | 109 JSONL files | 5,416,537 pages |
| `raw/hotpot_dev_distractor_v1.json` | HotpotQA (Yang et al., 2018) | dev, distractor setting | 7,405 questions |

The official FEVER test labels are hidden, so the labelled shared-task dev set is our test set.

## Layout

```
data/
  raw/                         downloaded files (above)
  wiki_sentences/              all 5.4M pages, one shard per wiki-NNN.jsonl
    sentences-NNN.parquet      page_id, sent_id, text, links
    pages-NNN.parquet          page_id, n_sentences, first_sentence
    _stats.json
  splits/
    train.jsonl                30,000 FEVER train claims, label mix kept (16,508 / 6,141 / 7,351)
    val.jsonl                  2,000 other FEVER train claims, same mix (1,101 / 409 / 490); tuning only
    test_full.jsonl            all 19,998 shared-task dev claims
    test_5k.jsonl              class-balanced 5,000-claim dev subset (1,667 / 1,667 / 1,666)
    split_stats.json
  corpus/
    sentences.parquet          page_id, sent_id, text, links  (<= 1M sentences)
    pages.parquet              page_id, source (evidence | decoy | random), n_sentences, first_sentence
    corpus_stats.json          pages per source, sentence total, evidence coverage, index-memory estimate
```

## Field notes

- **page_id** is the FEVER page title exactly as in the dump, e.g. `Soul_Food_-LRB-film-RRB-`
  (`-LRB-`/`-RRB-` are brackets). `src.utils.readable_title` turns it into `Soul Food (film)`.
- **sent_id** is the sentence number from the dump's `lines` field; annotated evidence uses the same
  numbers. Empty lines are dropped, so sent ids can have gaps.
- **links** are the hyperlinked page titles listed after each sentence in the dump. The two-hop
  chain (Prompt 11) uses them as bridge entities.
- **Evidence format** in the claim files: `evidence` is a list of alternative evidence *sets*; each
  item is `[annotation_id, evidence_id, page, sent_id]`. A claim is proven by any one complete set.
  NEI claims have `page = null`.

## Restricted corpus

1. **evidence**: every page cited by evidence of a train, val or test_full claim (always kept).
2. **decoy**: per claim, the top 3 non-evidence pages from a BM25 search over title + first
   sentence of all 5.4M pages (`src/data/page_bm25.py`). Rank-1 decoys of all claims are added before
   rank-2 ones, until the page target or sentence cap is reached.
3. **random**: random non-empty pages until about 50,000 pages.

Real counts are in `corpus/corpus_stats.json` after the run. Every SUPPORTS / REFUTES test claim
must have all its evidence pages in the corpus; the script stops with an error otherwise.

## Uploading as a Kaggle Dataset

From the notebook `notebooks/01_data.ipynb` (recommended):
1. Run the last cell, which copies `corpus/`, `splits/` and the small raw files to
   `/kaggle/working/fever_corpus_v1/` and deletes the large wiki files from the output.
2. **Save Version → Save & Run All (Commit)**.
3. When the run finishes, open the notebook's **Output** tab → **New Dataset** → name it
   `fever-corpus-v1` → keep it private and share it with your teammates (Settings → Collaborators).
4. In later notebooks: **Add Input → Your Datasets → fever-corpus-v1**. It appears at
   `/kaggle/input/fever-corpus-v1/`.

From your own machine with the Kaggle CLI (`pip install kaggle`, API token in `~/.kaggle/kaggle.json`):
```
kaggle datasets init -p fever_corpus_v1          # creates dataset-metadata.json
# edit dataset-metadata.json: "title": "fever-corpus-v1", "id": "<your-username>/fever-corpus-v1"
kaggle datasets create -p fever_corpus_v1 --dir-mode zip
kaggle datasets version -p fever_corpus_v1 -m "rebuild" --dir-mode zip   # for later updates
```
