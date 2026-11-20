"""Prompt 1 (M1): download FEVER + HotpotQA, parse the Wikipedia dump to Parquet, verify counts.

Run:
    python -m src.data.download --config configs/data.yaml

What it does, in order:
1. For each raw file, look for an existing copy (data/raw, repo root, /kaggle/input). If none is
   found, download it from the URLs in the config (fever.ai first, S3 mirror second).
2. Stream every wiki-pages/wiki-NNN.jsonl file out of the zip (no need to unzip 6+ GB to disk) and
   write one Parquet shard per file with columns page_id, sent_id, text, links.
   'links' are the hyperlinked page titles listed after each sentence in the dump's 'lines' field;
   Prompt 11 uses them as bridge entities.
3. Check the claim, label, page and question counts against the published numbers and stop with an
   error if any differ.
4. Print three training claims with their evidence resolved to sentence text.
"""
from __future__ import annotations

import argparse
import io
import json
import re
import shutil
import sys
import time
import urllib.request
import zipfile
from collections import Counter
from pathlib import Path

import pyarrow as pa
import pyarrow.dataset as ds
import pyarrow.parquet as pq

from src.utils import (
    canonical_page,
    evidence_sets,
    iter_jsonl,
    load_config,
    locate_file,
    readable_title,
    resolve,
    save_json,
    set_seed,
    start_run,
)

SENT_SCHEMA = pa.schema(
    [("page_id", pa.string()), ("sent_id", pa.int32()), ("text", pa.string()), ("links", pa.list_(pa.string()))]
)
PAGE_SCHEMA = pa.schema([("page_id", pa.string()), ("n_sentences", pa.int32()), ("first_sentence", pa.string())])


class DataCheckError(RuntimeError):
    """Raised when a verified count does not match the published number."""


# --------------------------------------------------------------------------- download
def download(urls: list[str], dest: Path, log) -> None:
    """Try each URL in turn; stream to dest.part and rename only when complete."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    last_error = None
    for url in urls:
        try:
            log.info(f"Downloading {url}")
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 (research-data-download)"})
            with urllib.request.urlopen(req, timeout=60) as resp, open(tmp, "wb") as out:
                total = int(resp.headers.get("Content-Length", 0))
                done, t0, last_print = 0, time.time(), 0.0
                while chunk := resp.read(1 << 20):
                    out.write(chunk)
                    done += len(chunk)
                    if time.time() - last_print > 15:
                        pct = f"{100 * done / total:.1f}%" if total else "?"
                        log.info(f"  {done / 1e6:,.0f} MB ({pct}) at {done / 1e6 / (time.time() - t0 + 1e-9):.1f} MB/s")
                        last_print = time.time()
            tmp.replace(dest)
            log.info(f"Saved {dest} ({dest.stat().st_size / 1e6:,.1f} MB)")
            return
        except Exception as exc:  # noqa: BLE001 - try the next mirror on any failure
            last_error = exc
            log.warning(f"  failed: {exc}")
            tmp.unlink(missing_ok=True)
    raise RuntimeError(
        f"Could not download {dest.name} from any URL ({last_error}).\n"
        "On Kaggle: turn on 'Internet' in the notebook settings (needs a phone-verified account), "
        "or click 'Add Input', search Kaggle Datasets for 'FEVER' / 'fever wiki-pages' / "
        "'hotpotqa', add one that contains this exact file name, and re-run. Any file under "
        "/kaggle/input is found automatically."
    )


def get_raw_file(name: str, cfg: dict, log) -> Path:
    found = locate_file(name, cfg["paths"]["search_dirs"])
    if found:
        log.info(f"Found existing {name}: {found}")
        return found
    dest = resolve(cfg["paths"]["raw_dir"]) / name
    try:
        download(cfg["download"]["files"][name], dest, log)
    except RuntimeError:
        if name != "hotpot_dev_distractor_v1.json":
            raise
        # The original HotpotQA host (curtis.ml.cmu.edu) is often unreachable; fall back to the
        # Hugging Face Hub copy of the same dev distractor set.
        hotpot_from_hf(dest, cfg["download"].get("hotpot_hf_repos", ["hotpotqa/hotpot_qa"]), log)
    return dest


def hotpot_from_hf(dest: Path, repos: list[str], log) -> None:
    """Load HotpotQA dev (distractor) from the HF Hub and save it in the original JSON format.

    HF stores supporting_facts as {"title": [...], "sent_id": [...]} and context as
    {"title": [...], "sentences": [[...], ...]}; the original file uses lists of [title, sent_id]
    and [title, [sentences]]. We convert back so every later step reads one format.
    """
    from datasets import load_dataset  # preinstalled on Kaggle

    last_error = None
    for repo in repos:
        try:
            log.info(f"Loading HotpotQA distractor validation split from Hugging Face: {repo}")
            data = load_dataset(repo, "distractor", split="validation")
            break
        except Exception as exc:  # noqa: BLE001 - try the next repo name
            last_error = exc
            log.warning(f"  failed: {exc}")
    else:
        raise RuntimeError(
            f"HotpotQA could not be loaded from the original host or Hugging Face ({last_error}). "
            "Add a Kaggle Dataset containing hotpot_dev_distractor_v1.json as an input and re-run."
        )
    rows = []
    for ex in data:
        rows.append({
            "_id": ex["id"],
            "question": ex["question"],
            "answer": ex["answer"],
            "type": ex["type"],
            "level": ex["level"],
            "supporting_facts": [[t, int(s)] for t, s in zip(ex["supporting_facts"]["title"], ex["supporting_facts"]["sent_id"])],
            "context": [[t, list(s)] for t, s in zip(ex["context"]["title"], ex["context"]["sentences"])],
        })
    dest.parent.mkdir(parents=True, exist_ok=True)
    with open(dest, "w", encoding="utf-8") as f:
        json.dump(rows, f, ensure_ascii=False)
    log.info(f"Saved {len(rows):,} HotpotQA questions to {dest} (converted from {repo})")


def find_wiki_source(cfg: dict, log) -> Path:
    """Return the wiki-pages.zip, or a folder of already-unzipped wiki-NNN.jsonl files."""
    unzipped = locate_file("wiki-001.jsonl", cfg["paths"]["search_dirs"])
    if unzipped:
        log.info(f"Found unzipped wiki pages in {unzipped.parent}")
        return unzipped.parent
    return get_raw_file("wiki-pages.zip", cfg, log)


# --------------------------------------------------------------------------- wiki parsing
def parse_lines_field(lines: str) -> list[tuple[int, str, list[str]]]:
    """Parse the 'lines' field of one wiki page.

    Each line looks like:  "<sent_id>\\t<sentence>\\t<anchor1>\\t<title1>\\t<anchor2>\\t<title2>..."
    The extra fields come in (anchor text, linked page title) pairs; we keep the linked titles.
    Empty sentences are dropped because they can never be evidence.
    """
    out = []
    for raw in lines.split("\n"):
        parts = raw.split("\t")
        if len(parts) < 2 or not parts[0].isdigit():
            continue
        text = parts[1].strip()
        if not text:
            continue
        extras = parts[2:]
        links = [canonical_page(t.strip()) for t in extras[1::2] if t.strip()]
        out.append((int(parts[0]), text, links))
    return out


WIKI_FILE = re.compile(r"^wiki-\d+\.jsonl$")


def is_wiki_file(path: str) -> bool:
    """True only for real dump files like 'wiki-pages/wiki-001.jsonl'.

    Rejects macOS metadata entries such as '__MACOSX/wiki-pages/._wiki-001.jsonl', which are
    binary and caused a UnicodeDecodeError.
    """
    return "__MACOSX" not in path and bool(WIKI_FILE.match(Path(path).name))


def iter_wiki_files(source: Path):
    """Yield (file_name, text stream) for each wiki-NNN.jsonl, from a zip or a folder."""
    if source.is_dir():
        for path in sorted(p for p in source.glob("wiki-*.jsonl") if is_wiki_file(str(p))):
            with open(path, "r", encoding="utf-8") as f:
                yield path.name, f
    else:
        with zipfile.ZipFile(source) as zf:
            members = sorted(m for m in zf.namelist() if is_wiki_file(m))
            if not members:
                raise RuntimeError(f"No wiki-NNN.jsonl files inside {source}")
            for m in members:
                with zf.open(m) as raw:
                    yield Path(m).name, io.TextIOWrapper(raw, encoding="utf-8")


def parse_wiki(source: Path, out_dir: Path, compression: str, log) -> dict:
    """Write data/wiki_sentences/sentences-NNN.parquet and pages-NNN.parquet. Resumable."""
    out_dir.mkdir(parents=True, exist_ok=True)
    stats_path = out_dir / "_stats.json"
    if stats_path.exists():
        stats = json.loads(stats_path.read_text())
        log.info(f"Wiki already parsed ({stats['files']} shards); delete {stats_path} to redo.")
        return stats

    stats = Counter()
    for name, stream in iter_wiki_files(source):
        shard = name.replace("wiki-", "").replace(".jsonl", "")
        sent_path, page_path = out_dir / f"sentences-{shard}.parquet", out_dir / f"pages-{shard}.parquet"
        cols = {"page_id": [], "sent_id": [], "text": [], "links": []}
        pages = {"page_id": [], "n_sentences": [], "first_sentence": []}
        n_lines = n_empty = 0
        for line in stream:
            if not line.strip():
                continue
            n_lines += 1
            page = json.loads(line)
            pid = canonical_page(page.get("id", ""))
            if not pid:
                n_empty += 1
                continue
            sents = parse_lines_field(page.get("lines", ""))
            for sid, text, links in sents:
                cols["page_id"].append(pid)
                cols["sent_id"].append(sid)
                cols["text"].append(text)
                cols["links"].append(links)
            pages["page_id"].append(pid)
            pages["n_sentences"].append(len(sents))
            pages["first_sentence"].append(sents[0][1] if sents else "")
        if not sent_path.exists():  # resume: shards written in an earlier session are kept
            pq.write_table(pa.table(cols, schema=SENT_SCHEMA), sent_path, compression=compression)
            pq.write_table(pa.table(pages, schema=PAGE_SCHEMA), page_path, compression=compression)
        stats["files"] += 1
        stats["json_lines"] += n_lines
        stats["empty_id_lines"] += n_empty
        stats["pages"] += len(pages["page_id"])
        stats["sentences"] += len(cols["page_id"])
        log.info(f"{name}: {len(pages['page_id']):,} pages, {len(cols['page_id']):,} sentences")
    stats = dict(stats)
    save_json(stats, stats_path)
    return stats


# --------------------------------------------------------------------------- verification
def check(condition: bool, message: str, log) -> None:
    if not condition:
        log.error(f"CHECK FAILED: {message}")
        raise DataCheckError(message)
    log.info(f"check ok: {message}")


def verify_claims(path: Path, total: int, labels: dict, name: str, log) -> dict:
    counts = Counter(c["label"] for c in iter_jsonl(path))
    n = sum(counts.values())
    log.info(f"{name}: {n:,} claims, labels {dict(counts)}")
    check(n == total, f"{name} has {n:,} claims (expected {total:,})", log)
    for label, expected in labels.items():
        check(counts[label] == expected, f"{name} {label} = {counts[label]:,} (expected {expected:,})", log)
    return {"total": n, "labels": dict(counts)}


def show_examples(train_path: Path, wiki_dir: Path, log, n: int = 3) -> list[dict]:
    """Print n training claims (S, R, and one multi-sentence) with evidence text."""
    wanted, picked = ["SUPPORTS", "REFUTES", "MULTI"], []
    for claim in iter_jsonl(train_path):
        sets = evidence_sets(claim)
        if not sets:
            continue
        kind = "MULTI" if len(sets[0]) > 1 else claim["label"]
        if kind in wanted:
            wanted.remove(kind)
            picked.append(claim)
        if len(picked) == n:
            break

    keys = {item for c in picked for item in evidence_sets(c)[0]}
    dataset = ds.dataset(sorted(str(p) for p in wiki_dir.glob("sentences-*.parquet")), format="parquet")
    table = dataset.to_table(filter=ds.field("page_id").isin(sorted({p for p, _ in keys})), columns=["page_id", "sent_id", "text"])
    lookup = {(p, s): t for p, s, t in zip(*(table.column(c).to_pylist() for c in ["page_id", "sent_id", "text"]))}

    examples = []
    for c in picked:
        log.info("-" * 80)
        log.info(f"[{c['label']}] {c['claim']}")
        resolved = []
        for page, sid in evidence_sets(c)[0]:
            text = lookup.get((page, sid), "<missing from dump>")
            log.info(f"   ({readable_title(page)}, {sid}): {text}")
            resolved.append({"page": page, "sent_id": sid, "text": text})
        examples.append({"id": c["id"], "label": c["label"], "claim": c["claim"], "evidence": resolved})
    return examples


# --------------------------------------------------------------------------- main
def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--config", default="configs/data.yaml")
    parser.add_argument("--skip-wiki", action="store_true", help="only download/verify claims (quick test)")
    args = parser.parse_args(argv)

    cfg = load_config(args.config)
    set_seed(cfg["seed"])
    run_dir, log = start_run(cfg, "download")
    exp = cfg["download"]["expected"]
    results = {}

    train = get_raw_file("train.jsonl", cfg, log)
    dev = get_raw_file("shared_task_dev.jsonl", cfg, log)
    hotpot = get_raw_file("hotpot_dev_distractor_v1.json", cfg, log)

    results["train"] = verify_claims(train, exp["train_total"], exp["train_labels"], "FEVER train", log)
    results["dev"] = verify_claims(dev, exp["dev_total"], exp["dev_labels"], "FEVER shared-task dev", log)

    with open(hotpot, "r", encoding="utf-8") as f:
        n_hotpot = len(json.load(f))
    check(n_hotpot == exp["hotpot_dev_distractor"], f"HotpotQA dev distractor has {n_hotpot:,} questions (expected {exp['hotpot_dev_distractor']:,})", log)
    results["hotpot_dev_distractor"] = n_hotpot

    # Copy small raw files into data/raw so later steps (and the Kaggle Dataset) have one location.
    raw_dir = resolve(cfg["paths"]["raw_dir"])
    raw_dir.mkdir(parents=True, exist_ok=True)
    for p in (train, dev, hotpot):
        if p.parent.resolve() != raw_dir.resolve():
            shutil.copy2(p, raw_dir / p.name)

    if not args.skip_wiki:
        wiki_dir = resolve(cfg["paths"]["wiki_parquet_dir"])
        stats = parse_wiki(find_wiki_source(cfg, log), wiki_dir, cfg["download"]["parquet_compression"], log)
        results["wiki"] = stats
        log.info(f"Wiki: {stats['json_lines']:,} JSON lines, {stats['pages']:,} pages with an id, {stats['sentences']:,} non-empty sentences")
        # The published 5,416,537 counts pages in the dump; accept it whether or not the
        # one blank-id record at the top of wiki-001 is counted.
        check(exp["wiki_pages"] in (stats["json_lines"], stats["pages"]), f"wiki pages = {stats['pages']:,} (+{stats['empty_id_lines']} blank ids; expected {exp['wiki_pages']:,})", log)
        results["examples"] = show_examples(train, wiki_dir, log)

    save_json(results, run_dir / "metrics.json")
    log.info(f"All checks passed. Summary written to {run_dir / 'metrics.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
