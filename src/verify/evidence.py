"""Evidence lists, NLI scoring with a cache, and the aggregation rule (shared by Prompts 12, 13 and 14).

Label order everywhere in this package: index 0 = E (entailment, SUPPORTS), 1 = C (contradiction,
REFUTES), 2 = N (neutral, NOT ENOUGH INFO).
"""
from __future__ import annotations

import hashlib
import random
import re
from pathlib import Path

import numpy as np
import pandas as pd

from src.retrieval.common import doc_text, find_output, read_run, require_output, resolve_run
from src.utils import clean_text, evidence_sets, resolve

LABEL_OF = ["SUPPORTS", "REFUTES", "NOT ENOUGH INFO"]          # index -> FEVER label
INDEX_OF = {lab: i for i, lab in enumerate(LABEL_OF)}           # FEVER label -> index


# --------------------------------------------------------------------------- evidence sources
def retrieved_evidence(run: dict, claims: list[dict], k: int) -> dict:
    """claim id -> top-k (page, sent_id) of a retriever run."""
    return {c["id"]: [(p, s) for p, s, _ in run.get(c["id"], [])[:k]] for c in claims}


def chain_evidence(rows: list[dict]) -> tuple[dict, dict]:
    """claim id -> chain sentences (in chain order), and claim id -> missing_link flag."""
    evid = {r["id"]: [(it["page_id"], int(it["sent_id"])) for it in r["chain"]] for r in rows}
    return evid, {r["id"]: bool(r["missing_link"]) for r in rows}


def page_index(corpus: pd.DataFrame) -> dict:
    """page_id -> sorted list of its sentence ids."""
    return corpus.groupby("page_id")["sent_id"].apply(lambda s: sorted(int(x) for x in s)).to_dict()


def oracle_evidence(claims: list[dict], texts: dict, pages: dict, nearest_run: dict, k: int, nei_sentences: int, seed: int) -> dict:
    """Annotated evidence instead of retrieved evidence (the verifier's upper bound).

    S/R claims: their annotated evidence sentences (all sets, annotation order, duplicates removed), up to k.
    NEI claims have no annotated evidence, so they get `nei_sentences` sentences sampled from the closest-
    matching page: the best page of the claim's BM25 run. Without that, an empty premise would give the
    label away.
    """
    out = {}
    for c in claims:
        sets = evidence_sets(c)
        if sets:
            out[c["id"]] = list(dict.fromkeys(x for g in sets for x in g if x in texts))[:k]
            continue
        hits = nearest_run.get(c["id"], [])
        page = hits[0][0] if hits else None
        sids = pages.get(page, [])
        rng = random.Random(f"{seed}-{c['id']}")
        out[c["id"]] = [(page, s) for s in sorted(rng.sample(sids, min(nei_sentences, len(sids))))]
    return out


def load_evidence(cfg: dict, spec: dict, split: str, claims: list[dict], texts: dict, pages: dict) -> tuple[dict, dict, str]:
    """Evidence for `claims` from a source spec. split is "val" or "test" (file suffix).

    spec examples: {type: run, run: auto, k: 5} | {type: chain, run: chain} | {type: gold, nearest_run: bm25, nei_sentences: 2, k: 5}
    Returns (evidence, missing_link flags (chain only, else {}), description).
    """
    kind = spec["type"]
    if kind == "run":
        name = resolve_run(cfg, spec["run"])
        return retrieved_evidence(read_run(require_output(cfg, name, f"run_{split}.jsonl")), claims, spec["k"]), {}, f"run:{name}"
    if kind == "chain":
        from src.utils import read_jsonl

        rows = read_jsonl(require_output(cfg, spec.get("run", "chain"), f"chains_{split}.jsonl"))
        evid, missing = chain_evidence(rows)
        ids = {c["id"] for c in claims}
        return {cid: e for cid, e in evid.items() if cid in ids}, missing, f"chain:{spec.get('run', 'chain')}"
    if kind == "gold":
        nearest = read_run(require_output(cfg, spec["nearest_run"], f"run_{split}.jsonl"))
        return oracle_evidence(claims, texts, pages, nearest, spec["k"], spec["nei_sentences"], cfg["seed"]), {}, "gold"
    raise ValueError(f"unknown evidence type {kind}")


# --------------------------------------------------------------------------- NLI model + cache
def label_columns(id2label: dict) -> list[int]:
    """Model output columns for [entailment, contradiction, neutral], read from config.id2label.

    Different checkpoints order their labels differently (roberta-large-mnli: 0 = contradiction,
    2 = entailment; cross-encoder/nli-deberta-v3-base: 0 = contradiction, 1 = entailment). Generic
    names such as LABEL_0 are refused, because then the mapping cannot be checked.
    """
    found = {}
    for idx, name in id2label.items():
        n = name.lower()
        key = "E" if n.startswith("entail") else "C" if n.startswith("contradict") else "N" if n.startswith("neutral") else None
        if key:
            found[key] = int(idx)
    assert set(found) == {"E", "C", "N"} and len(set(found.values())) == 3, f"cannot map labels {id2label} to E/C/N"
    return [found["E"], found["C"], found["N"]]


class NLIScorer:
    """Loads the NLI checkpoint on first use and returns logits in E, C, N order."""

    def __init__(self, name: str, fp16: bool, max_length: int, batch_size: int, log):
        self.name, self.fp16, self.max_length, self.batch_size, self.log = name, fp16, max_length, batch_size, log
        self._model, self.cols = None, None

    def _load(self):
        if self._model is None:
            from src.cross_encoder import PairModel

            self._model = PairModel(self.name, self.fp16, self.max_length, self.batch_size)
            self.cols = label_columns(self._model.id2label)
            self.log.info(f"NLI model {self.name} on {self._model.device}; id2label {self._model.id2label} -> columns E,C,N = {self.cols}")
        return self._model

    def logits(self, premises: list[str], hypotheses: list[str]) -> np.ndarray:
        model = self._load()
        return model.logits(premises, hypotheses, progress=len(premises) > 20_000)[:, self.cols]

    def sanity_check(self) -> None:
        """The mapping must give E for an obvious entailment and C for an obvious contradiction."""
        p = ["A man is playing a guitar on stage.", "A man is playing a guitar on stage."]
        h = ["A man is playing an instrument.", "Nobody is playing any instrument."]
        pred = self.logits(p, h).argmax(1).tolist()
        assert pred == [0, 1], f"{self.name}: expected [E, C] on the sanity pairs, got {pred} (0=E, 1=C, 2=N)"
        self.log.info(f"{self.name}: label mapping sanity check passed (entailment and contradiction pairs).")


def slug(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9]+", "_", name.split("/")[-1]).strip("_").lower()


class NLICache:
    """Logits per (claim id, page, sent_id), stored in outputs/nli_cache/<model>.parquet.

    The hypothesis is always the claim and the premise always "title: sentence", so one entry serves every
    pipeline that shows this sentence to this claim. A cache attached from an earlier notebook is reused.
    """

    def __init__(self, cfg: dict, scorer: NLIScorer, log):
        self.scorer, self.log = scorer, log
        self.path = resolve(cfg["paths"]["outputs_dir"]) / "nli_cache" / f"{slug(scorer.name)}.parquet"
        self.store: dict = {}
        src = self.path if self.path.exists() else find_output(cfg, "nli_cache", f"{slug(scorer.name)}.parquet")
        if src is not None:
            df = pd.read_parquet(src)
            self.store = {(int(i), p, int(s)): np.array([e, c, n], np.float32)
                          for i, p, s, e, c, n in zip(df["id"], df["page_id"], df["sent_id"], df["l_e"], df["l_c"], df["l_n"])}
            log.info(f"NLI cache: {len(self.store):,} entries from {src}")

    def get(self, items: list[tuple]) -> np.ndarray:
        """items: [(claim id, page, sent_id, premise, hypothesis)] -> logits (n, 3); computes what is missing."""
        missing = list(dict.fromkeys((i, p, s) for i, p, s, _, _ in items if (i, p, s) not in self.store))
        if missing:
            text = {(i, p, s): (pr, hy) for i, p, s, pr, hy in items}
            self.log.info(f"NLI: scoring {len(missing):,} new (claim, sentence) pairs with {self.scorer.name}")
            logits = self.scorer.logits([text[m][0] for m in missing], [text[m][1] for m in missing])
            for m, l in zip(missing, logits):
                self.store[m] = l
            self.save()
        return np.array([self.store[(i, p, s)] for i, p, s, _, _ in items], np.float32).reshape(-1, 3)

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        keys = list(self.store)
        arr = np.array([self.store[k] for k in keys], np.float32).reshape(-1, 3)
        pd.DataFrame({"id": [k[0] for k in keys], "page_id": [k[1] for k in keys], "sent_id": [k[2] for k in keys],
                      "l_e": arr[:, 0], "l_c": arr[:, 1], "l_n": arr[:, 2]}).to_parquet(self.path, index=False)


def softmax(z: np.ndarray, axis: int = -1) -> np.ndarray:
    z = z - z.max(axis=axis, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=axis, keepdims=True)


def sentence_table(evid: dict, claims: list[dict], texts: dict, cache: NLICache) -> pd.DataFrame:
    """One row per (claim, evidence sentence): logits and probabilities. premise = 'title: sentence'."""
    items, meta = [], []
    for c in claims:
        for rank, (p, s) in enumerate(evid.get(c["id"], []), start=1):
            if (p, s) in texts:
                items.append((c["id"], p, int(s), doc_text(p, texts[(p, s)]), clean_text(c["claim"])))
                meta.append((c["id"], rank, p, int(s)))
    logits = cache.get(items) if items else np.zeros((0, 3), np.float32)
    probs = softmax(logits) if len(logits) else logits
    df = pd.DataFrame(meta, columns=["id", "rank", "page_id", "sent_id"])
    for j, n in enumerate("ecn"):
        df[f"l_{n}"] = logits[:, j] if len(logits) else []
        df[f"p_{n}"] = probs[:, j] if len(probs) else []
    return df


def concat_table(evid: dict, claims: list[dict], texts: dict, cache: NLICache) -> pd.DataFrame:
    """Alternative: one premise per claim, all evidence sentences joined; one row per claim."""
    items, meta = [], []
    for c in claims:
        keys = [(p, s) for p, s in evid.get(c["id"], []) if (p, s) in texts]
        if not keys:
            continue
        premise = " ".join(doc_text(p, texts[(p, s)]) for p, s in keys)
        tag = "CONCAT:" + hashlib.sha1(repr(keys).encode()).hexdigest()[:16]
        items.append((c["id"], tag, -1, premise, clean_text(c["claim"])))
        meta.append((c["id"], 1, tag, -1))
    logits = cache.get(items) if items else np.zeros((0, 3), np.float32)
    df = pd.DataFrame(meta, columns=["id", "rank", "page_id", "sent_id"])
    probs = softmax(logits) if len(logits) else logits
    for j, n in enumerate("ecn"):
        df[f"l_{n}"] = logits[:, j] if len(logits) else []
        df[f"p_{n}"] = probs[:, j] if len(probs) else []
    return df


# --------------------------------------------------------------------------- aggregation
def aggregate(table: pd.DataFrame, claims: list[dict]) -> dict:
    """Per claim (in `claims` order): max p(E), max p(C), min p(N) over its evidence sentences, and a claim-level
    distribution P = normalise(max p(E), max p(C), min p(N)): the most entailing sentence, the most
    contradicting sentence, and the least neutral one. With one sentence P equals that sentence's probabilities.
    A claim with no evidence gets P = (0, 0, 1)."""
    g = table.groupby("id").agg(max_pe=("p_e", "max"), max_pc=("p_c", "max"), min_pn=("p_n", "min"), n=("p_e", "size"))
    g = g.reindex([c["id"] for c in claims])
    max_pe = g["max_pe"].fillna(0.0).to_numpy()
    max_pc = g["max_pc"].fillna(0.0).to_numpy()
    min_pn = g["min_pn"].fillna(1.0).to_numpy()
    P = np.stack([max_pe, max_pc, min_pn], axis=1)
    P = P / np.clip(P.sum(1, keepdims=True), 1e-12, None)
    return {"max_pe": max_pe, "max_pc": max_pc, "min_pn": min_pn, "P": P, "n_sentences": g["n"].fillna(0).astype(int).to_numpy()}


def rule_labels(max_pe: np.ndarray, max_pc: np.ndarray, theta_e: float, theta_c: float) -> np.ndarray:
    """Contradiction first, then entailment, otherwise NEI. Returns label indices (0=S, 1=R, 2=NEI)."""
    return np.where(max_pc > theta_c, 1, np.where(max_pe > theta_e, 0, 2))


def macro_f1_idx(y: np.ndarray, p: np.ndarray, n_classes: int = 3) -> float:
    f1s = []
    for k in range(n_classes):
        tp = np.sum((y == k) & (p == k))
        fp = np.sum((y != k) & (p == k))
        fn = np.sum((y == k) & (p != k))
        f1s.append(2 * tp / (2 * tp + fp + fn) if tp else 0.0)
    return float(np.mean(f1s))


def tune_thresholds(max_pe: np.ndarray, max_pc: np.ndarray, gold: np.ndarray, grid: list[float]) -> tuple[float, float, float]:
    """(theta_e, theta_c, macro-F1) maximising macro-F1 on validation; ties go to the higher thresholds."""
    best = (-1.0, 0.0, 0.0)
    for te in grid:
        for tc in grid:
            f = macro_f1_idx(gold, rule_labels(max_pe, max_pc, te, tc))
            if (f, te + tc) > (best[0], best[1] + best[2]):
                best = (f, te, tc)
    return best[1], best[2], best[0]


def gold_indices(claims: list[dict]) -> np.ndarray:
    return np.array([INDEX_OF[c["label"]] for c in claims])


def classification_report(gold: np.ndarray, pred: np.ndarray) -> dict:
    """Accuracy, macro-F1, per-class F1 and the confusion matrix (rows = gold, columns = predicted)."""
    from src.eval.metrics import accuracy, macro_f1, per_class_f1

    y, p = [LABEL_OF[i] for i in gold], [LABEL_OF[i] for i in pred]
    cm = [[int(np.sum((gold == a) & (pred == b))) for b in range(3)] for a in range(3)]
    return {"accuracy": accuracy(y, p), "macro_f1": macro_f1(y, p), "per_class_f1": per_class_f1(y, p), "confusion": cm, "n": len(y)}


def save_probs(table: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    table.to_parquet(path, index=False)
