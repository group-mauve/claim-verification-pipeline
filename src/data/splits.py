"""Prompt 2 (M1): build train / validation / test splits.

Run:
    python -m src.data.splits --config configs/data.yaml

- train: 30,000 claims sampled from FEVER train, keeping its label mix (55.0 / 20.5 / 24.5 %).
- val:   2,000 *other* FEVER train claims, same mix. Used only for tuning.
- test_full: all 19,998 shared-task dev claims (official test labels are hidden).
- test_5k:   a fixed class-balanced 5,000-claim subset of dev (1,667 / 1,667 / 1,666), for when
             full-dev runs are too slow. Later scripts choose with the config flag test_subset.
"""
from __future__ import annotations

import argparse
import random
import sys
from collections import Counter, defaultdict

from src.utils import LABELS, load_config, locate_file, read_jsonl, resolve, save_json, set_seed, start_run, write_jsonl


def allocate(total: int, weights: dict[str, float]) -> dict[str, int]:
    """Split `total` into integer counts proportional to `weights` (largest-remainder rounding).

    Example: allocate(30000, train label counts) -> {SUPPORTS: 16508, REFUTES: 6141, NEI: 7351}.
    The counts always sum exactly to `total`.
    """
    s = sum(weights.values())
    exact = {k: total * w / s for k, w in weights.items()}
    counts = {k: int(v) for k, v in exact.items()}
    leftover = total - sum(counts.values())
    for k in sorted(exact, key=lambda k: exact[k] - counts[k], reverse=True)[:leftover]:
        counts[k] += 1
    return counts


def stratified_sample(claims: list[dict], counts: dict[str, int], rng: random.Random) -> list[dict]:
    """Sample exactly counts[label] claims of each label without replacement."""
    by_label = defaultdict(list)
    for c in claims:
        by_label[c["label"]].append(c)
    picked = []
    for label, n in counts.items():
        pool = by_label[label]
        if n > len(pool):
            raise ValueError(f"asked for {n} {label} claims but only {len(pool)} available")
        picked.extend(rng.sample(pool, n))
    rng.shuffle(picked)
    return picked


def describe(name: str, rows: list[dict], log) -> dict:
    counts = Counter(r["label"] for r in rows)
    shares = " / ".join(f"{l[:3]} {counts[l]:,} ({100 * counts[l] / max(len(rows), 1):.1f}%)" for l in LABELS)
    log.info(f"{name:10s} {len(rows):>7,} claims | {shares}")
    return {"total": len(rows), **{l: counts[l] for l in LABELS}}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--config", default="configs/data.yaml")
    args = parser.parse_args(argv)

    cfg = load_config(args.config)
    set_seed(cfg["seed"])
    run_dir, log = start_run(cfg, "splits")
    sp = cfg["splits"]
    rng = random.Random(cfg["seed"])

    train_path = locate_file("train.jsonl", cfg["paths"]["search_dirs"])
    dev_path = locate_file("shared_task_dev.jsonl", cfg["paths"]["search_dirs"])
    if not train_path or not dev_path:
        raise FileNotFoundError("Run `python -m src.data.download` first (train.jsonl / shared_task_dev.jsonl not found).")
    fever_train, fever_dev = read_jsonl(train_path), read_jsonl(dev_path)

    # Duplicate ids would break the "no id in two splits" guarantee, so check first.
    ids = Counter(c["id"] for c in fever_train)
    dupes = [i for i, n in ids.items() if n > 1]
    if dupes:
        log.warning(f"{len(dupes)} duplicate ids in FEVER train; keeping the first occurrence of each")
        seen, unique = set(), []
        for c in fever_train:
            if c["id"] not in seen:
                seen.add(c["id"])
                unique.append(c)
        fever_train = unique

    mix = Counter(c["label"] for c in fever_train)
    train = stratified_sample(fever_train, allocate(sp["train_size"], mix), rng)
    train_ids = {c["id"] for c in train}
    remaining = [c for c in fever_train if c["id"] not in train_ids]
    val = stratified_sample(remaining, allocate(sp["val_size"], mix), rng)

    test_full = fever_dev
    balanced = allocate(sp["test_subset_size"], {l: 1 for l in LABELS})
    test_5k = stratified_sample(fever_dev, balanced, random.Random(cfg["seed"]))

    # No claim may appear in two splits (test_5k is a subset of test_full by design).
    split_ids = {"train": train_ids, "val": {c["id"] for c in val}, "test_full": {c["id"] for c in test_full}}
    names = list(split_ids)
    for i, a in enumerate(names):
        for b in names[i + 1 :]:
            overlap = split_ids[a] & split_ids[b]
            assert not overlap, f"{len(overlap)} claim ids appear in both {a} and {b}, e.g. {sorted(overlap)[:5]}"
    assert {c["id"] for c in test_5k} <= split_ids["test_full"]
    log.info("check ok: no claim id appears in two splits; test_5k is a subset of test_full")

    out_dir = resolve(cfg["paths"]["splits_dir"])
    stats = {}
    for name, rows in [("train", train), ("val", val), ("test_full", test_full), ("test_5k", test_5k)]:
        write_jsonl(rows, out_dir / f"{name}.jsonl")
        stats[name] = describe(name, rows, log)
    log.info(f"Example train claim: {train[0]['label']} | {train[0]['claim']}")
    save_json(stats, run_dir / "metrics.json")
    save_json(stats, out_dir / "split_stats.json")
    log.info(f"Splits written to {out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
