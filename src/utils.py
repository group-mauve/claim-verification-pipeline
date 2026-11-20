"""Shared helpers used by every script: config loading, seeds, logging, JSON/JSONL I/O.

Every script in this repo follows the same pattern:
    cfg = load_config(args.config)
    set_seed(cfg["seed"])
    run_dir, log = start_run(cfg, "<step name>")
    ...
    save_json(metrics, run_dir / "metrics.json")
"""
from __future__ import annotations

import json
import logging
import os
import random
import sys
import unicodedata
from pathlib import Path
from typing import Any, Iterable, Iterator

import yaml

# Repository root = the folder that contains src/. All relative paths in configs
# are resolved against it, so scripts work no matter where they are launched from.
REPO_ROOT = Path(__file__).resolve().parents[1]

LABELS = ["SUPPORTS", "REFUTES", "NOT ENOUGH INFO"]


# --------------------------------------------------------------------------- config
def load_config(path: str | os.PathLike) -> dict:
    """Read a YAML config file into a plain dict."""
    with open(resolve(path), "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def resolve(path: str | os.PathLike) -> Path:
    """Turn a config path into an absolute Path (relative paths start at the repo root)."""
    p = Path(path)
    return p if p.is_absolute() else REPO_ROOT / p


# --------------------------------------------------------------------------- seeds
def set_seed(seed: int) -> None:
    """Fix every random number generator we use so runs are repeatable."""
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import numpy as np

        np.random.seed(seed)
    except ImportError:
        pass
    try:  # torch is only needed from Phase 2 on
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


# --------------------------------------------------------------------------- logging
def start_run(cfg: dict, step: str | None = None) -> tuple[Path, logging.Logger]:
    """Create outputs/<run_name>/[<step>/] and a logger that writes to console and run.log."""
    run_dir = resolve(cfg["paths"]["outputs_dir"]) / cfg["run_name"]
    if step:
        run_dir = run_dir / step
    run_dir.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger(f"{cfg['run_name']}.{step or 'main'}")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s", "%H:%M:%S")
    for handler in (logging.StreamHandler(sys.stdout), logging.FileHandler(run_dir / "run.log", encoding="utf-8")):
        handler.setFormatter(fmt)
        logger.addHandler(handler)
    # Keep a copy of the exact config next to the results.
    with open(run_dir / "config_used.yaml", "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)
    return run_dir, logger


# --------------------------------------------------------------------------- I/O
def read_jsonl(path: str | os.PathLike) -> list[dict]:
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def iter_jsonl(path: str | os.PathLike) -> Iterator[dict]:
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def write_jsonl(rows: Iterable[dict], path: str | os.PathLike) -> int:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            n += 1
    return n


def save_json(obj: Any, path: str | os.PathLike) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


# --------------------------------------------------------------------------- FEVER helpers
_TITLE_TOKENS = {"-LRB-": "(", "-RRB-": ")", "-LSB-": "[", "-RSB-": "]", "-LCB-": "{", "-RCB-": "}", "-COLON-": ":"}


def canonical_page(page_id: str) -> str:
    """One Unicode form (NFC) for every page title.

    FEVER stores accented titles inconsistently: 'García' may be one character 'í' (NFC) or
    'i' + a combining accent (NFD). Both look identical but compare unequal, which made 88
    evidence pages look missing from the dump. Every title we read goes through this function.
    """
    return page_id if page_id.isascii() else unicodedata.normalize("NFC", page_id)


def readable_title(page_id: str) -> str:
    """FEVER page ids look like 'Soul_Food_-LRB-film-RRB-'; return 'Soul Food (film)'."""
    text = page_id
    for token, char in _TITLE_TOKENS.items():
        text = text.replace(token, char)
    return text.replace("_", " ")


def clean_text(text: str) -> str:
    """Undo the dump's tokenisation for model input: '-LRB- 2008 -RRB-' -> '( 2008 )', `` '' -> "."""
    for token, char in _TITLE_TOKENS.items():
        text = text.replace(token, char)
    return text.replace("``", '"').replace("''", '"')


def evidence_sets(claim: dict) -> list[list[tuple[str, int]]]:
    """Annotated evidence of a FEVER claim as a list of sets; each set is [(page, sent_id), ...].

    A claim is proven by ANY one complete set. NEI claims have page = None and return [].
    """
    sets = []
    for group in claim.get("evidence", []):
        items = [(canonical_page(e[2]), int(e[3])) for e in group if e[2] is not None]
        if items:
            sets.append(items)
    return sets


def evidence_pages(claim: dict) -> set[str]:
    return {page for group in evidence_sets(claim) for page, _ in group}


def locate_file(name: str, search_dirs: list[str]) -> Path | None:
    """Find an existing copy of `name` in any search dir (recursively). Returns None if absent."""
    for d in search_dirs:
        root = resolve(d)
        if not root.exists():
            continue
        direct = root / name
        if direct.is_file():
            return direct
        # Recursive search, but skip our own outputs and virtual environments.
        for hit in root.rglob(name):
            if hit.is_file() and not any(part in {"outputs", ".venv", "venv", ".git"} for part in hit.parts):
                return hit
    return None
