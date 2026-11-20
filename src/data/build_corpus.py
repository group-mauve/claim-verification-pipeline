"""Prompt 3 (M1): build the restricted Wikipedia corpus (~50k pages, <= 1M sentences).

Run (after download.py and splits.py):
    python -m src.data.build_corpus --config configs/data.yaml

Pages come from three sources, added in this priority order:
1. evidence: every page cited by annotated evidence of a train, val or test_full claim. Always kept,
   so every SUPPORTS / REFUTES test claim can be answered from the corpus.
2. decoy: for each claim, the top 3 pages from a BM25 search over (title + first sentence) of all
   5.4M pages that are NOT that claim's evidence pages. These look similar to the right answer and
   make retrieval realistically hard. If there are more decoys than room, lower BM25 ranks go first
   (every claim's rank-1 decoy before any rank-2 decoy).
3. random: random pages until the total reaches target_pages.
A page is added only if the sentence total stays <= max_sentences (evidence pages are exempt).
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import time
from collections import Counter

import pandas as pd
import pyarrow as pa
import pyarrow.dataset as ds
import pyarrow.parquet as pq

from src.data.page_bm25 import SparseBM25
from src.utils import (
    LABELS,
    canonical_page,
    evidence_pages,
    evidence_sets,
    load_config,
    read_jsonl,
    readable_title,
    resolve,
    save_json,
    set_seed,
    start_run,
)


def load_all_pages(wiki_dir) -> pd.DataFrame:
    files = sorted(str(p) for p in wiki_dir.glob("pages-*.parquet"))
    if not files:
        raise FileNotFoundError(f"No pages-*.parquet in {wiki_dir}; run src.data.download first.")
    pages = ds.dataset(files, format="parquet").to_table().to_pandas()
    # Keep the title exactly as stored in the Parquet (needed to filter the sentence shards) and
    # use the NFC form everywhere else. Shards parsed before the Unicode fix are NFD for accents.
    pages["stored_id"] = pages["page_id"]
    pages["page_id"] = pages["page_id"].map(canonical_page)
    return pages.drop_duplicates("page_id", keep="first").reset_index(drop=True)


def find_decoys(claims: list[dict], pages: pd.DataFrame, cfg: dict, cache_path, log) -> dict[str, list[str]]:
    """claim id -> up to decoys_per_claim non-evidence page ids, best BM25 match first. Cached."""
    if cache_path.exists():
        log.info(f"Loading cached decoys from {cache_path}")
        cached = json.loads(cache_path.read_text())
        # A cache written before the Unicode fix may hold an evidence page in its other form;
        # normalise and drop any decoy that is actually one of the claim's evidence pages.
        by_id = {str(x["id"]): evidence_pages(x) for x in claims}
        return {k: [p for p in map(canonical_page, v) if p not in by_id.get(k, set())] for k, v in cached.items()}

    c = cfg["corpus"]
    log.info(f"Building temporary BM25 over {len(pages):,} pages (title + first sentence) ...")
    t0 = time.time()
    docs = [readable_title(p) + " " + s for p, s in zip(pages["page_id"], pages["first_sentence"])]
    bm25 = SparseBM25(k1=c["bm25_k1"], b=c["bm25_b"]).fit(docs)
    del docs
    log.info(f"  index built in {time.time() - t0:.0f}s")

    page_ids = pages["page_id"].tolist()
    decoys = {}
    t0 = time.time()
    step = 2000
    for i in range(0, len(claims), step):
        batch = claims[i : i + step]
        hits = bm25.search([x["claim"] for x in batch], k=c["decoy_candidates"], batch_size=c["bm25_batch_size"])
        for claim, ranked in zip(batch, hits):
            gold = evidence_pages(claim)
            picks = [page_ids[j] for j, _ in ranked if page_ids[j] not in gold]
            decoys[str(claim["id"])] = picks[: c["decoys_per_claim"]]
        done = i + len(batch)
        log.info(f"  decoys for {done:,}/{len(claims):,} claims ({done / (time.time() - t0):.0f} claims/s)")
    cache_path.write_text(json.dumps(decoys))
    return decoys


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--config", default="configs/data.yaml")
    args = parser.parse_args(argv)

    cfg = load_config(args.config)
    set_seed(cfg["seed"])
    run_dir, log = start_run(cfg, "build_corpus")
    c = cfg["corpus"]
    splits_dir, wiki_dir = resolve(cfg["paths"]["splits_dir"]), resolve(cfg["paths"]["wiki_parquet_dir"])
    out_dir = resolve(cfg["paths"]["corpus_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    splits = {name: read_jsonl(splits_dir / f"{name}.jsonl") for name in ("train", "val", "test_full")}
    claims = [x for rows in splits.values() for x in rows]
    log.info(f"Claims: " + ", ".join(f"{k} {len(v):,}" for k, v in splits.items()))

    pages = load_all_pages(wiki_dir)
    n_sent = dict(zip(pages["page_id"], pages["n_sentences"]))
    log.info(f"Wikipedia pages available: {len(pages):,}")

    # 1. Evidence pages ------------------------------------------------------------------
    ev_pages = set().union(*(evidence_pages(x) for x in claims))
    missing = sorted(p for p in ev_pages if p not in n_sent)
    if missing:
        log.warning(f"{len(missing)} evidence pages are not in the dump, e.g. {missing[:5]}")
    ev_pages -= set(missing)
    selected = {p: "evidence" for p in ev_pages}
    total_sent = sum(n_sent[p] for p in ev_pages)
    log.info(f"Evidence pages: {len(ev_pages):,} ({total_sent:,} sentences)")

    def try_add(page: str, source: str) -> bool:
        nonlocal total_sent
        if page in selected or page not in n_sent or n_sent[page] == 0:
            return False
        if len(selected) >= c["target_pages"] or total_sent + n_sent[page] > c["max_sentences"]:
            return False
        selected[page] = source
        total_sent += n_sent[page]
        return True

    # 2. Decoy pages, rank 1 for every claim first, then rank 2, then rank 3 ---------------
    decoys = find_decoys(claims, pages, cfg, out_dir / "_decoy_cache.json", log)
    candidates = 0
    for rank in range(c["decoys_per_claim"]):
        for claim in claims:
            picks = decoys.get(str(claim["id"]), [])
            if rank < len(picks):
                candidates += 1
                try_add(picks[rank], "decoy")
    n_decoy = sum(1 for s in selected.values() if s == "decoy")
    log.info(f"Decoy pages: {n_decoy:,} added out of {candidates:,} claim-decoy candidates")

    # 3. Random pages until target --------------------------------------------------------
    rng = random.Random(cfg["seed"])
    pool = [p for p in pages["page_id"] if p not in selected and n_sent[p] > 0]
    rng.shuffle(pool)
    for p in pool:
        if len(selected) >= c["target_pages"] or total_sent >= c["max_sentences"]:
            break
        try_add(p, "random")
    del pool

    # 4. Collect the sentences of the selected pages ---------------------------------------
    log.info("Reading sentences of the selected pages ...")
    files = sorted(str(p) for p in wiki_dir.glob("sentences-*.parquet"))
    stored = pages.loc[pages["page_id"].isin(selected), "stored_id"].tolist()
    table = ds.dataset(files, format="parquet").to_table(filter=ds.field("page_id").isin(sorted(stored)))
    sentences = table.to_pandas()
    sentences["page_id"] = sentences["page_id"].map(canonical_page)
    sentences["links"] = sentences["links"].map(lambda ls: [canonical_page(t) for t in ls])
    sentences = sentences.sort_values(["page_id", "sent_id"]).reset_index(drop=True)
    if len(sentences) > c["max_sentences"]:
        log.warning(f"{len(sentences):,} sentences exceed the cap because evidence pages alone are that large")

    page_df = pages[pages["page_id"].isin(selected)].copy()
    page_df["source"] = page_df["page_id"].map(selected)
    page_df = page_df[["page_id", "source", "n_sentences", "first_sentence"]].sort_values("page_id")
    pq.write_table(pa.Table.from_pandas(sentences, preserve_index=False), out_dir / "sentences.parquet", compression="zstd")
    pq.write_table(pa.Table.from_pandas(page_df, preserve_index=False), out_dir / "pages.parquet", compression="zstd")

    # 5. Statistics and coverage checks ---------------------------------------------------
    source_counts = Counter(selected.values())
    sent_keys = set(zip(sentences["page_id"], sentences["sent_id"]))
    coverage = {}
    for name in ("train", "val", "test_full"):
        per_label = {}
        for label in LABELS[:2]:
            rows = [x for x in splits[name] if x["label"] == label]
            pages_ok = sum(all(p in selected for p in evidence_pages(x)) for x in rows)
            sets_ok = sum(any(all(k in sent_keys for k in s) for s in evidence_sets(x)) for x in rows)
            per_label[label] = {
                "claims": len(rows),
                "all_evidence_pages_in_corpus": pages_ok / max(len(rows), 1),
                "some_full_evidence_set_in_corpus": sets_ok / max(len(rows), 1),
            }
        coverage[name] = per_label

    bytes_needed = len(sentences) * c["embedding_dim"] * 4
    stats = {
        "pages_total": len(selected),
        "pages_by_source": dict(source_counts),
        "sentences_total": int(len(sentences)),
        "missing_evidence_pages_in_dump": len(missing),
        "coverage": coverage,
        "index_memory_bytes_float32": bytes_needed,
        "index_memory_gb_float32": round(bytes_needed / 1024**3, 3),
    }
    save_json(stats, out_dir / "corpus_stats.json")
    save_json(stats, run_dir / "metrics.json")

    log.info("=" * 70)
    for src in ("evidence", "decoy", "random"):
        log.info(f"pages from {src:9s}: {source_counts[src]:>7,}")
    log.info(f"total pages        : {len(selected):>7,}")
    log.info(f"total sentences    : {len(sentences):>7,}")
    for name, per_label in coverage.items():
        for label, v in per_label.items():
            log.info(
                f"{name:9s} {label:8s}: evidence pages in corpus {100 * v['all_evidence_pages_in_corpus']:.2f}% | "
                f"a full evidence set in corpus {100 * v['some_full_evidence_set_in_corpus']:.2f}%"
            )
    log.info(f"FAISS Flat index estimate: {len(sentences):,} x {c['embedding_dim']} x 4 bytes = {stats['index_memory_gb_float32']} GB")
    log.info(f"Example rows:\n{sentences.head(3).to_string()}")

    for label in LABELS[:2]:
        share = coverage["test_full"][label]["all_evidence_pages_in_corpus"]
        if share < 1.0:
            raise AssertionError(f"Only {100 * share:.2f}% of test {label} claims have all evidence pages in the corpus (must be 100%)")
    log.info("check ok: 100% of S/R test claims have all their evidence pages in the corpus")
    log.info(f"Corpus written to {out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
