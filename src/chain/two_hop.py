"""Prompt 11 (M3): two-hop evidence chain.

Run (GPU preferred, CPU works; needs Prompts 8-9 outputs):
    python -m src.chain.two_hop --config configs/chain.yaml

How one claim is processed
1. Hop 1: the top `budget` (5) sentences of the chosen retriever run (hybrid, or reranked if Prompt 10 kept it).
2. Bridge entities: from the top `bridge_from` hop-1 sentences, take the hyperlinked page titles stored with
   each sentence in the dump ('links'); for a sentence with no links, take named entities found by spaCy
   en_core_web_sm. Entities already named in the claim, and links back to the sentence's own page, are dropped.
   At most `max_bridges` entities are kept, in hop-1 rank order.
3. Hop 2: query = claim text + " " + bridge entity, searched with the fine-tuned dense retriever (its cosine
   score has a fixed scale, so one threshold means the same thing for every claim; RRF scores do not).
   Hop-2 sentences already in hop 1 are skipped; the rest are kept only if their cosine score >= threshold,
   at most `hop2_k` of them.
4. Chain = hop-1 sentences, then kept hop-2 sentences, cut to the same budget of 5 sentences
   (hop-1 top (5 - n_hop2) + n_hop2 hop-2 sentences). The FEVER scorer only reads the first 5 predicted
   sentences, so a chain longer than 5 would hide its hop-2 sentences; the equal budget also makes
   "1 hop vs 2 hops" a fair comparison.
5. missing_link = True when no bridge entity was found or no hop-2 sentence passed the threshold.
The threshold is tuned on validation (S/R claims) for the share of claims whose chain holds a complete
evidence set, with supporting-fact recall breaking ties; the grid includes "never use hop 2".

Evaluation
(a) FEVER test claims whose annotated evidence spans 2+ pages: supporting-fact recall (and P, F1, complete-set
    rate) of hop-1 top 5 vs the 2-hop chain of 5.
(b) HotpotQA dev distractor (each question comes with its 10 paragraphs: 2 gold + 8 decoys): the same procedure
    over those paragraphs' sentences only. Hyperlinks are not in the distractor file, so a "link" is the
    title of another of the 10 paragraphs that appears in a hop-1 sentence; spaCy entities otherwise. The
    threshold is tuned on the first `tune_questions` questions and results are reported on the rest.
Outputs in outputs/chain/: chains_val.jsonl, chains_test.jsonl, metrics.json, examples.json.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time

import numpy as np
import pandas as pd

from src.eval.metrics import evidence_set_hit_at_k, mean_supporting_fact_prf
from src.retrieval.common import (
    find_corpus_dir,
    find_model_dir,
    find_splits_dir,
    load_claims,
    load_corpus,
    read_run,
    require_output,
    resolve_run,
    sentence_lookup,
    test_split_name,
)
from src.utils import clean_text, evidence_sets, load_config, locate_file, readable_title, save_json, set_seed, start_run, write_jsonl

ENTITY_TYPES = {"PERSON", "ORG", "GPE", "LOC", "NORP", "FAC", "EVENT", "WORK_OF_ART", "PRODUCT", "LAW", "LANGUAGE"}
_PAREN = re.compile(r"\s*\([^)]*\)\s*$")
_CAPS = re.compile(r"\b(?:[A-Z][\w'-]+(?:\s+(?:of|the|de|and|von|van)?\s*[A-Z][\w'-]+)*)")


# --------------------------------------------------------------------------- entities
def base_name(entity: str) -> str:
    """'Soul_Food_-LRB-film-RRB-' or 'Soul Food (film)' -> 'Soul Food'."""
    return _PAREN.sub("", readable_title(entity)).strip()


def mentioned_in(entity: str, text: str) -> bool:
    name = base_name(entity).lower()
    return bool(name) and name in text.lower()


class EntityFinder:
    """Named entities with spaCy en_core_web_sm; a capitalised-words regex if spaCy is not installed."""

    def __init__(self, model: str, log):
        self.log, self.nlp = log, None
        try:
            import spacy

            self.nlp = spacy.load(model, disable=["lemmatizer", "tagger", "parser", "attribute_ruler"])
        except Exception as e:  # noqa: BLE001 - any failure means: use the fallback
            log.warning(f"spaCy model {model} not available ({e}); using a capitalised-words fallback for entities.")

    def __call__(self, texts: list[str]) -> list[list[str]]:
        if self.nlp is None:
            return [[m.group(0) for m in _CAPS.finditer(t) if len(m.group(0)) > 2] for t in texts]
        return [[e.text for e in doc.ents if e.label_ in ENTITY_TYPES] for doc in self.nlp.pipe(texts, batch_size=256)]


def pick_bridges(question: str, hop1: list[tuple], links: dict, ents: dict, own_title, max_bridges: int) -> list[dict]:
    """Bridge entities from hop-1 sentences: links first, named entities for sentences without links.

    hop1: [(key, ...)] in rank order, key = (page, sent_id); own_title(key) gives the sentence's page title.
    """
    out, seen = [], set()
    for item in hop1:
        key = item[0]
        cands = [(l, "link") for l in links.get(key, [])] or [(e, "ner") for e in ents.get(key, [])]
        for ent, src in cands:
            name = readable_title(ent).strip()
            low = base_name(name).lower()
            if not low or low in seen or low == base_name(own_title(key)).lower() or mentioned_in(name, question):
                continue
            seen.add(low)
            out.append({"entity": name, "source": src})
            if len(out) >= max_bridges:
                return out
    return out


def assemble_chain(hop1: list[tuple], candidates: list[tuple], threshold: float, budget: int, hop2_k: int):
    """hop1: [(key, score)]; candidates: [(key, score, bridge)] best first, none of them in hop1.
    Returns (chain items [(hop, key, score, bridge)], missing_link)."""
    kept = [c for c in candidates if c[1] >= threshold][:hop2_k]
    chain = [(1, k, s, None) for k, s in hop1[: budget - len(kept)]] + [(2, k, s, b) for k, s, b in kept]
    return chain, not kept


# --------------------------------------------------------------------------- FEVER
def spans_pages(claim: dict) -> bool:
    """True if some annotated evidence set uses sentences from two or more pages."""
    return any(len({p for p, _ in g}) >= 2 for g in evidence_sets(claim))


def evidence_quality(evid: dict, claims: list[dict]) -> dict:
    """Supporting-fact P/R/F1 and complete-set rate of claim id -> [(page, sent)] over S/R claims."""
    pairs, hits = [], []
    for c in claims:
        sets = evidence_sets(c)
        if not sets:
            continue
        pred = evid.get(c["id"], [])
        pairs.append((pred, {x for g in sets for x in g}))
        hits.append(evidence_set_hit_at_k(pred, sets, len(pred)))
    out = mean_supporting_fact_prf(pairs) or {"sf_precision": 0.0, "sf_recall": 0.0, "sf_f1": 0.0}
    out["complete_set_rate"] = float(np.mean(hits)) if hits else 0.0
    out["n_claims"] = len(pairs)
    return out


def hop2_search(claims, bridges: dict, hop1_keys: dict, encode, index, keys: list, search_k: int) -> dict:
    """claim id -> hop-2 candidates [(key, score, bridge)] best first (max score per sentence)."""
    queries, owners = [], []
    for c in claims:
        for b in bridges.get(c["id"], []):
            queries.append(f"{clean_text(c['claim'])} {b['entity']}")
            owners.append((c["id"], b["entity"]))
    cands: dict = {c["id"]: {} for c in claims}
    if queries:
        scores, idx = index.search(encode(queries), search_k)
        for (cid, ent), srow, irow in zip(owners, scores, idx):
            for sc, j in zip(srow, irow):
                if j < 0:
                    continue
                key = keys[j]
                if key in hop1_keys[cid]:
                    continue
                if key not in cands[cid] or sc > cands[cid][key][0]:
                    cands[cid][key] = (float(sc), ent)
    return {cid: sorted(((k, s, b) for k, (s, b) in d.items()), key=lambda x: -x[1]) for cid, d in cands.items()}


def tune_threshold(hop1: dict, cands: dict, claims: list[dict], grid: list[float], budget: int, hop2_k: int, log) -> tuple[float, list]:
    rows = []
    for t in grid:
        evid = {cid: [k for _, k, _, _ in assemble_chain(hop1[cid], cands.get(cid, []), t, budget, hop2_k)[0]] for cid in hop1}
        q = evidence_quality(evid, claims)
        used = np.mean([not assemble_chain(hop1[cid], cands.get(cid, []), t, budget, hop2_k)[1] for cid in hop1])
        rows.append({"threshold": t, **q, "share_with_hop2": float(used)})
        log.info(f"  threshold {t:.2f}: complete-set rate {q['complete_set_rate']:.4f}, sf recall {q['sf_recall']:.4f}, "
                 f"sf F1 {q['sf_f1']:.4f}, claims with a hop-2 sentence {used:.1%}")
    best = max(rows, key=lambda r: (r["complete_set_rate"], r["sf_recall"], r["threshold"]))
    return best["threshold"], rows


# --------------------------------------------------------------------------- HotpotQA
def hotpot_items(data: list[dict]):
    """Per question: list of (title, sent_idx, text) over its 10 paragraphs, and the gold supporting facts."""
    for q in data:
        sents = [(title, i, s) for title, lines in q["context"] for i, s in enumerate(lines) if s.strip()]
        gold = {(t, int(i)) for t, i in q["supporting_facts"]}
        yield q, sents, gold


def run_hotpot(data: list[dict], encode, ner: EntityFinder, hc: dict, log) -> dict:
    items = list(hotpot_items(data))
    flat = [f"{t}: {s.strip()}" for _, sents, _ in items for t, _, s in sents]
    log.info(f"HotpotQA: {len(items):,} questions, {len(flat):,} sentences to encode")
    vecs = encode(flat)
    qvecs = encode([q["question"] for q, _, _ in items])
    offsets = np.cumsum([0] + [len(s) for _, s, _ in items])
    budget, hop2_k, bridge_from = hc["budget"], hc["hop2_k"], hc["bridge_from"]

    # hop 1 and bridge entities
    hop1s, need_ner = [], {}
    for qi, (q, sents, _) in enumerate(items):
        v = vecs[offsets[qi] : offsets[qi + 1]]
        sc = v @ qvecs[qi]
        order = np.argsort(-sc, kind="stable")
        hop1s.append([((sents[j][0], sents[j][1]), float(sc[j])) for j in order[:budget]])
        titles = {t for t, _, _ in sents}
        for (key, _) in hop1s[-1][:bridge_from]:
            text = next(s for t, i, s in sents if (t, i) == key)
            linked = [t for t in titles if t != key[0] and mentioned_in(t, text)]
            if not linked:
                need_ner[(qi, key)] = text
    ner_out = dict(zip(need_ner, ner(list(need_ner.values())))) if need_ner else {}

    queries, owners, bridges_all = [], [], []
    for qi, (q, sents, _) in enumerate(items):
        titles = {t for t, _, _ in sents}
        text_of = {(t, i): s for t, i, s in sents}
        links = {key: [t for t in titles if t != key[0] and mentioned_in(t, text_of[key])] for key, _ in hop1s[qi][:bridge_from]}
        ents = {key: ner_out.get((qi, key), []) for key, _ in hop1s[qi][:bridge_from]}
        br = pick_bridges(q["question"], hop1s[qi][:bridge_from], links, ents, lambda k: k[0], hc["max_bridges"])
        bridges_all.append(br)
        for b in br:
            queries.append(f"{q['question']} {b['entity']}")
            owners.append(qi)
    hvecs = encode(queries) if queries else np.zeros((0, vecs.shape[1]), np.float32)

    cands = [dict() for _ in items]
    for qv, qi in zip(hvecs, owners):
        sents = items[qi][1]
        sc = vecs[offsets[qi] : offsets[qi + 1]] @ qv
        in_hop1 = {k for k, _ in hop1s[qi]}
        for j, s in enumerate(sc):
            key = (sents[j][0], sents[j][1])
            if key not in in_hop1 and (key not in cands[qi] or s > cands[qi][key]):
                cands[qi][key] = float(s)
    cand_lists = [sorted(((k, s, None) for k, s in d.items()), key=lambda x: -x[1]) for d in cands]

    def evaluate(idx: range, t: float) -> dict:
        one = mean_supporting_fact_prf(([k for k, _ in hop1s[i]], items[i][2]) for i in idx)
        two = mean_supporting_fact_prf(([k for _, k, _, _ in assemble_chain(hop1s[i], cand_lists[i], t, budget, hop2_k)[0]], items[i][2]) for i in idx)
        return {"one_hop": one, "two_hop": two}

    n_tune = min(hc["tune_questions"], len(items) // 2)
    tune_idx, eval_idx = range(0, n_tune), range(n_tune, len(items))
    grid = [{"threshold": t, **evaluate(tune_idx, t)["two_hop"]} for t in hc["thresholds"]]
    best_t = max(grid, key=lambda r: (r["sf_f1"], r["threshold"]))["threshold"]
    res = evaluate(eval_idx, best_t)
    res.update({"threshold": best_t, "tuning_grid": grid, "n_tune": n_tune, "n_eval": len(eval_idx),
                "share_with_bridge": float(np.mean([bool(b) for b in bridges_all])),
                "bridge_sources": {s: sum(b["source"] == s for br in bridges_all for b in br) for s in ("link", "ner")}})
    log.info(f"HotpotQA (threshold {best_t} tuned on {n_tune} questions, evaluated on {len(eval_idx):,}):\n"
             f"  1 hop  (top {budget}): P {res['one_hop']['sf_precision']:.4f} R {res['one_hop']['sf_recall']:.4f} F1 {res['one_hop']['sf_f1']:.4f}\n"
             f"  2 hops (chain of {budget}): P {res['two_hop']['sf_precision']:.4f} R {res['two_hop']['sf_recall']:.4f} F1 {res['two_hop']['sf_f1']:.4f}")
    return res


# --------------------------------------------------------------------------- main
def load_dense(cfg: dict, corpus: pd.DataFrame):
    """Fine-tuned query encoder + FAISS index over its corpus embeddings (row order = corpus order)."""
    from src.retrieval.dense import FlatIndex, load_bi_encoder, load_embeddings, st_encoder

    cc = cfg["chain"]
    model, device = load_bi_encoder(find_model_dir(cfg, cc["dense_model"]), True)
    index = FlatIndex(load_embeddings(cfg, cc["dense_run"], len(corpus)))
    return st_encoder(model, 256), index, device


def chain_rows(claims, hop1, cands, bridges, threshold, budget, hop2_k, texts) -> list[dict]:
    rows = []
    for c in claims:
        chain, missing = assemble_chain(hop1[c["id"]], cands.get(c["id"], []), threshold, budget, hop2_k)
        rows.append({
            "id": c["id"], "label": c["label"], "claim": c["claim"], "missing_link": missing,
            "bridges": bridges.get(c["id"], []),
            "chain": [{"hop": h, "page_id": k[0], "sent_id": int(k[1]), "text": texts.get(k, ""), "score": round(float(s), 6), "bridge": b}
                      for h, k, s, b in chain],
        })
    return rows


def print_chain(row: dict, gold: set, log) -> None:
    log.info(f"[{row['label']}] {row['claim']}  | missing_link={row['missing_link']} | bridges={[b['entity'] for b in row['bridges']]}")
    log.info(f"  gold: {sorted(gold)[:4]}")
    for it in row["chain"]:
        mark = "[GOLD]" if (it["page_id"], it["sent_id"]) in gold else "      "
        via = f" via '{it['bridge']}'" if it["bridge"] else ""
        log.info(f"  hop {it['hop']} {mark} ({readable_title(it['page_id'])}, {it['sent_id']}) {it['score']:.3f}{via} | {it['text'][:120]}")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--config", default="configs/chain.yaml")
    parser.add_argument("--skip-hotpot", action="store_true")
    args = parser.parse_args(argv)

    cfg = load_config(args.config)
    set_seed(cfg["seed"])
    run_dir, log = start_run(cfg)
    cc = cfg["chain"]
    budget, hop2_k = cc["budget"], cc["hop2_k"]

    corpus_dir = find_corpus_dir(cfg)
    corpus = load_corpus(corpus_dir)
    texts = sentence_lookup(corpus)
    keys = list(zip(corpus["page_id"], corpus["sent_id"].astype(int)))
    lk = pd.read_parquet(corpus_dir / "sentences.parquet", columns=["page_id", "sent_id", "links"])
    links = {(p, int(s)): list(l) for p, s, l in zip(lk["page_id"], lk["sent_id"], lk["links"]) if l is not None and len(l)}
    del lk
    log.info(f"Corpus {len(corpus):,} sentences, {len(links):,} with hyperlinks")

    hop1_name = resolve_run(cfg, cc["hop1_run"])
    splits_dir = find_splits_dir(cfg)
    split = test_split_name(cfg)
    claims = {"val": load_claims(splits_dir, "val"), "test": load_claims(splits_dir, split)}
    encode, index, device = load_dense(cfg, corpus)
    ner = EntityFinder(cc["spacy_model"], log)
    log.info(f"Hop 1 from run '{hop1_name}'; hop 2 with {cc['dense_model']} on {device}")

    hop1, cands, bridges = {}, {}, {}
    for sp in ("val", "test"):
        run = read_run(require_output(cfg, hop1_name, f"run_{sp}.jsonl"))
        h1 = {c["id"]: [((p, s), sc) for p, s, sc in run.get(c["id"], [])[:budget]] for c in claims[sp]}
        need = {k for cid in h1 for k, _ in h1[cid][: cc["bridge_from"]] if k not in links}
        t0 = time.time()
        ents = dict(zip(need, ner([texts.get(k, "") for k in need]))) if need else {}
        br = {c["id"]: pick_bridges(c["claim"], h1[c["id"]][: cc["bridge_from"]], links, ents, lambda k: k[0], cc["max_bridges"])
              for c in claims[sp]}
        log.info(f"{sp}: entity search on {len(need):,} hop-1 sentences without links took {time.time() - t0:.0f}s; "
                 f"{np.mean([bool(b) for b in br.values()]):.1%} of claims have a bridge entity")
        hop1[sp], bridges[sp] = h1, br
        cands[sp] = hop2_search(claims[sp], br, {cid: {k for k, _ in v} for cid, v in h1.items()}, encode, index, keys, cc["search_k"])

    log.info("Tuning the hop-2 threshold on validation S/R claims:")
    threshold, grid = tune_threshold(hop1["val"], cands["val"], claims["val"], cc["thresholds"], budget, hop2_k, log)
    log.info(f"Chosen threshold: {threshold}" + ("  (= never use hop 2: no threshold helped on validation)" if threshold > 1 else ""))

    rows = {}
    for sp in ("val", "test"):
        rows[sp] = chain_rows(claims[sp], hop1[sp], cands[sp], bridges[sp], threshold, budget, hop2_k, texts)
        write_jsonl(rows[sp], run_dir / f"chains_{sp}.jsonl")

    # ---- (a) FEVER: 1 hop vs 2 hops
    test = claims["test"]
    one = {c["id"]: [k for k, _ in hop1["test"][c["id"]]] for c in test}
    two = {r["id"]: [(it["page_id"], it["sent_id"]) for it in r["chain"]] for r in rows["test"]}
    multi = [c for c in test if spans_pages(c)]
    fever = {}
    for name, subset in (("multi_page", multi), ("all_sr", test)):
        fever[name] = {"one_hop": evidence_quality(one, subset), "two_hop": evidence_quality(two, subset)}
        o, t = fever[name]["one_hop"], fever[name]["two_hop"]
        log.info(f"FEVER test, {name} ({o['n_claims']:,} claims): sf recall {o['sf_recall']:.4f} -> {t['sf_recall']:.4f}, "
                 f"complete-set rate {o['complete_set_rate']:.4f} -> {t['complete_set_rate']:.4f}, sf F1 {o['sf_f1']:.4f} -> {t['sf_f1']:.4f}")
    fever["missing_link_share_test"] = float(np.mean([r["missing_link"] for r in rows["test"]]))

    # ---- examples: bridge worked / bridge failed / no bridge
    by_id = {c["id"]: c for c in test}
    examples, wanted = [], {"bridge worked": None, "bridge failed": None, "no bridge entity": None}
    for r in rows["test"]:
        c = by_id[r["id"]]
        if not spans_pages(c):
            continue
        gold = {x for g in evidence_sets(c) for x in g}
        hop2_gold = any(it["hop"] == 2 and (it["page_id"], it["sent_id"]) in gold for it in r["chain"])
        kind = ("bridge worked" if hop2_gold else "bridge failed") if r["bridges"] else "no bridge entity"
        if wanted[kind] is None:
            wanted[kind] = (r, gold)
        if all(wanted.values()):
            break
    for kind, val in wanted.items():
        log.info("=" * 90 + f"\nExample chain: {kind}")
        if val is None:
            log.info("  (no such claim among the multi-page test claims)")
            continue
        print_chain(val[0], val[1], log)
        examples.append({"kind": kind, **val[0]})
    save_json(examples, run_dir / "examples.json")

    # ---- (b) HotpotQA
    hotpot = None
    if not args.skip_hotpot:
        path = locate_file("hotpot_dev_distractor_v1.json", cfg["paths"]["search_dirs"])
        if path is None:
            log.warning("hotpot_dev_distractor_v1.json not found (it is in fever-corpus-v1/raw); skipping HotpotQA.")
        else:
            data = json.loads(path.read_text(encoding="utf-8"))
            if cc["hotpot"].get("max_questions"):
                data = data[: cc["hotpot"]["max_questions"]]
            hotpot = run_hotpot(data, encode, ner, cc["hotpot"], log)

    save_json({"hop1_run": hop1_name, "test_split": split, "threshold": threshold, "threshold_grid": grid,
               "budget": budget, "hop2_k": hop2_k, "fever": fever, "hotpot": hotpot}, run_dir / "metrics.json")
    log.info(f"Outputs written to {run_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
