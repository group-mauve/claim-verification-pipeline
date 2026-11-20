"""Prompt 8 (M2): fine-tune MiniLM the DPR way (in-batch negatives + one BM25 hard negative per claim).

Run (GPU session, after Prompt 5 so that outputs/bm25/run_train.jsonl exists):
    python -m src.retrieval.finetune --config configs/finetune.yaml              # seed 42 -> dense_ft
    python -m src.retrieval.finetune --config configs/finetune.yaml --seed 43    # second seed, for the noise estimate in Prompt 10

Steps
1. Training rows, one per SUPPORTS / REFUTES claim of our 30k train split:
   anchor   = the claim,
   positive = one annotated evidence sentence (picked at random among the claim's evidence sentences),
   negative = the highest-ranked BM25 sentence that is NOT annotated evidence and NOT from an evidence page.
   All sentences use the same "title: sentence" text as the index.
2. False-negative check: a hard negative can be a correct sentence that the annotators did not list. We flag
   rows where the off-the-shelf model already scores the negative at least as high as the positive, and
   print 10 hard negatives to read by hand.
3. Loss: MultipleNegativesRankingLoss. For each claim in a batch of B rows, the model must pick its own
   positive out of 2B candidates (all B positives + all B hard negatives of the batch), with softmax
   cross-entropy over cosine similarities x 20. A no-duplicates batch sampler keeps a sentence from
   appearing twice in one batch (it would otherwise be a wrong "negative" for its other claim).
4. 1 epoch, batch 128, learning rate 2e-5, 10% linear warmup, fp16, seed 42.
5. Re-encode the corpus with the fine-tuned model, rebuild the FAISS index, evaluate on val and test
   (src/retrieval/dense.run_dense), and compare validation Recall@10 before and after.
Outputs: models/minilm_ft/ (the model), outputs/dense_ft/ (run_val, run_test, metrics, embeddings, index),
outputs/finetune/ (training rows, loss curve, false-negative check, before/after summary).
"""
from __future__ import annotations

import argparse
import copy
import math
import random
import sys
import time
from collections import Counter

import numpy as np

from src.retrieval.common import (
    doc_text,
    find_corpus_dir,
    find_output,
    find_splits_dir,
    format_report,
    load_claims,
    load_corpus,
    read_run,
    require_output,
    sentence_lookup,
)
from src.retrieval.dense import load_bi_encoder, run_dense, st_encoder
from src.utils import clean_text, evidence_pages, evidence_sets, load_config, resolve, save_json, set_seed, start_run, write_jsonl


# --------------------------------------------------------------------------- training rows
def build_training_rows(train: list[dict], bm25_run: dict, texts: dict, seed: int) -> tuple[list[dict], Counter]:
    """One (anchor, positive, hard negative) row per S/R claim. See the module docstring for the rules."""
    rng = random.Random(seed)
    keys = sorted(texts)  # for the rare claim whose BM25 list has no usable negative
    rows, stats = [], Counter()
    for c in train:
        sets = evidence_sets(c)
        if not sets:
            stats["skipped_nei"] += 1
            continue
        gold = {x for g in sets for x in g}
        candidates = sorted(x for x in gold if x in texts)
        if not candidates:
            stats["skipped_no_evidence_text"] += 1
            continue
        pos = rng.choice(candidates)
        ev_pages = evidence_pages(c)
        neg, neg_rank = None, None
        for rank, (p, s, _) in enumerate(bm25_run.get(c["id"], []), start=1):
            if (p, s) not in gold and p not in ev_pages and (p, s) in texts:
                neg, neg_rank = (p, s), rank
                break
        if neg is None:
            stats["random_negative"] += 1
            while neg is None or neg[0] in ev_pages:
                neg = keys[rng.randrange(len(keys))]
        else:
            stats["bm25_negative"] += 1
        rows.append({
            "id": c["id"], "label": c["label"], "anchor": clean_text(c["claim"]),
            "positive": doc_text(pos[0], texts[pos]), "negative": doc_text(neg[0], texts[neg]),
            "pos_id": list(pos), "neg_id": list(neg), "neg_bm25_rank": neg_rank,
        })
    return rows, stats


def false_negative_check(rows: list[dict], encode, n_show: int, seed: int, log) -> dict:
    """Flag rows whose hard negative the base model scores >= the gold positive; print n_show negatives."""
    a = encode([r["anchor"] for r in rows])
    p = encode([r["positive"] for r in rows])
    n = encode([r["negative"] for r in rows])
    sim_pos, sim_neg = (a * p).sum(1), (a * n).sum(1)
    flagged = sim_neg >= sim_pos
    log.info(f"False-negative check: base model scores the hard negative >= the positive for "
             f"{flagged.sum():,} of {len(rows):,} rows ({100 * flagged.mean():.1f}%). These are the rows most likely "
             "to hold unlabelled correct evidence (or a model mistake).")
    show = random.Random(seed).sample(range(len(rows)), min(n_show, len(rows)))
    examples = []
    for i in show:
        r = rows[i]
        mark = "FLAG: read carefully" if flagged[i] else "looks like a true negative"
        log.info(f"[{r['label']}] {r['anchor']}\n    positive ({sim_pos[i]:.3f}): {r['positive'][:160]}\n"
                 f"    negative ({sim_neg[i]:.3f}, BM25 rank {r['neg_bm25_rank']}): {r['negative'][:160]}\n    -> {mark}")
        examples.append({**{k: r[k] for k in ("id", "label", "anchor", "positive", "negative", "neg_bm25_rank")},
                         "sim_positive": float(sim_pos[i]), "sim_negative": float(sim_neg[i]), "flagged": bool(flagged[i])})
    return {"n_rows": len(rows), "n_flagged": int(flagged.sum()), "share_flagged": float(flagged.mean()), "examples": examples}


# --------------------------------------------------------------------------- training
def train_model(rows: list[dict], fc: dict, seed: int, out_dir, run_dir, log) -> list[dict]:
    import torch
    from datasets import Dataset
    from sentence_transformers import (
        SentenceTransformer,
        SentenceTransformerTrainer,
        SentenceTransformerTrainingArguments,
        losses,
    )
    from sentence_transformers.training_args import BatchSamplers

    cuda = torch.cuda.is_available()
    model = SentenceTransformer(fc["base_model"], device="cuda" if cuda else "cpu")
    model.max_seq_length = fc["max_seq_length"]
    data = Dataset.from_dict({k: [r[k] for r in rows] for k in ("anchor", "positive", "negative")})
    loss = losses.MultipleNegativesRankingLoss(model)
    steps = math.ceil(len(rows) / fc["batch_size"]) * fc["epochs"]
    args = SentenceTransformerTrainingArguments(
        output_dir=str(run_dir / "checkpoints"),
        num_train_epochs=fc["epochs"],
        per_device_train_batch_size=fc["batch_size"],
        learning_rate=fc["learning_rate"],
        warmup_steps=int(fc["warmup_ratio"] * steps),
        lr_scheduler_type="linear",
        fp16=fc["fp16"] and cuda,
        seed=seed,
        data_seed=seed,
        batch_sampler=BatchSamplers.NO_DUPLICATES,
        save_strategy="no",
        logging_steps=fc["logging_steps"],
        report_to="none",
    )
    log.info(f"Training {fc['base_model']}: {len(rows):,} rows, batch {fc['batch_size']}, {steps} steps, "
             f"lr {fc['learning_rate']}, warmup {args.warmup_steps} steps, fp16 {args.fp16}, seed {seed}")
    t0 = time.time()
    trainer = SentenceTransformerTrainer(model=model, args=args, train_dataset=data, loss=loss)
    trainer.train()
    log.info(f"Training took {time.time() - t0:.0f}s")
    history = trainer.state.log_history
    losses_logged = [h["loss"] for h in history if "loss" in h]
    if losses_logged:
        log.info(f"Training loss: first logged {losses_logged[0]:.4f}, last logged {losses_logged[-1]:.4f}")
    out_dir.mkdir(parents=True, exist_ok=True)
    model.save(str(out_dir))
    log.info(f"Saved fine-tuned model to {out_dir}")
    return history


# --------------------------------------------------------------------------- main
def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--config", default="configs/finetune.yaml")
    parser.add_argument("--seed", type=int, default=None, help="override the seed; run names get a _seed<N> suffix")
    parser.add_argument("--skip-train", action="store_true", help="reuse an existing models/minilm_ft")
    args = parser.parse_args(argv)

    cfg = load_config(args.config)
    fc = cfg["finetune"]
    suffix = ""
    if args.seed is not None and args.seed != cfg["seed"]:
        suffix = f"_seed{args.seed}"
        cfg["seed"] = args.seed
    cfg["run_name"] += suffix
    seed = cfg["seed"]
    set_seed(seed)
    run_dir, log = start_run(cfg)
    model_dir = resolve(fc["output_model_dir"] + suffix)

    corpus = load_corpus(find_corpus_dir(cfg))
    texts = sentence_lookup(corpus)
    splits_dir = find_splits_dir(cfg)
    train = load_claims(splits_dir, "train")
    bm25_path = require_output(cfg, fc["bm25_run"], "run_train.jsonl")
    bm25_run = read_run(bm25_path)
    log.info(f"Train claims {len(train):,}; BM25 train run {bm25_path} ({len(bm25_run):,} claims)")

    rows, stats = build_training_rows(train, bm25_run, texts, seed)
    log.info(f"Training rows: {len(rows):,} | {dict(stats)}")
    ranks = [r["neg_bm25_rank"] for r in rows if r["neg_bm25_rank"]]
    log.info(f"Hard negative BM25 rank: median {np.median(ranks):.0f}, rank 1 for {np.mean(np.array(ranks) == 1):.1%} of rows")
    log.info(f"Example row:\n  anchor:   {rows[0]['anchor']}\n  positive: {rows[0]['positive'][:150]}\n  negative: {rows[0]['negative'][:150]}")
    write_jsonl(rows, run_dir / "train_rows.jsonl")

    # ---- validation Recall@10 of the base model (reused from Prompt 6 if that output is attached)
    dense_base = copy.deepcopy(cfg["dense_eval"])
    base_metrics_path = find_output(cfg, fc["baseline_run"], "metrics.json")
    if base_metrics_path is not None:
        import json

        before = json.loads(base_metrics_path.read_text(encoding="utf-8"))
        log.info(f"Off-the-shelf metrics reused from {base_metrics_path}")
    else:
        log.info("No off-the-shelf dense output found; evaluating the base model now (about 2 minutes).")
        before = run_dense({**cfg, "run_name": fc["baseline_run"], "dense": {**dense_base, "model_name": fc["base_model"]}})

    base_model, _ = load_bi_encoder(fc["base_model"], fp16=True)
    fn = false_negative_check(rows, st_encoder(base_model, 256), fc["false_negative_examples"], seed, log)
    del base_model
    save_json(fn, run_dir / "false_negative_check.json")

    # ---- train, then re-encode the corpus and evaluate
    if args.skip_train and model_dir.exists():
        log.info(f"--skip-train: using {model_dir}")
        history = []
    else:
        history = train_model(rows, fc, seed, model_dir, run_dir, log)
    save_json(history, run_dir / "loss_history.json")

    dense_cfg = {**cfg, "run_name": fc["dense_run"] + suffix,
                 "dense": {**dense_base, "model_name": str(model_dir), "save_faiss_index": not suffix and dense_base["save_faiss_index"],
                           "compare_with": dense_base.get("compare_with") if not suffix else None}}
    after = run_dense(dense_cfg)

    summary = {
        "seed": seed, "n_rows": len(rows), "row_stats": dict(stats), "model_dir": str(model_dir),
        "val_before": before["val"], "val_after": after["val"],
        "test_before": before.get("test"), "test_after": after["test"],
        "val_sentence_recall@10": {"before": before["val"]["sentence"]["recall@10"], "after": after["val"]["sentence"]["recall@10"]},
        "false_negative_share": fn["share_flagged"],
    }
    save_json(summary, run_dir / "metrics.json")
    log.info(f"Validation sentence Recall@10: {summary['val_sentence_recall@10']['before']:.4f} (off the shelf) -> "
             f"{summary['val_sentence_recall@10']['after']:.4f} (fine-tuned)")
    log.info("TEST, fine-tuned:\n" + format_report(after["test"], {"ms/query": after["ms_per_query"], "index GB": after["index_size_gb"]}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
