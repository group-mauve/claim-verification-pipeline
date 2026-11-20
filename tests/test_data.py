"""Tests for the data-pipeline helpers on toy inputs (no downloads needed)."""
import random
from collections import Counter

from src.data.download import is_wiki_file, parse_lines_field


def test_is_wiki_file_skips_macos_metadata():
    assert is_wiki_file("wiki-pages/wiki-001.jsonl")
    assert is_wiki_file("wiki-109.jsonl")
    assert not is_wiki_file("__MACOSX/wiki-pages/._wiki-001.jsonl")
    assert not is_wiki_file("wiki-pages/._wiki-001.jsonl")
    assert not is_wiki_file("wiki-pages/README.txt")
from src.data.page_bm25 import SparseBM25
from src.data.splits import allocate, stratified_sample
from src.utils import evidence_pages, evidence_sets, readable_title


def test_parse_lines_field_keeps_links_and_drops_empty():
    lines = "0\tNikolaj is a Danish actor .\tDanish\tDenmark\tactor\tActor\n1\t\n2\tHe was born in 1970 ."
    out = parse_lines_field(lines)
    assert out == [(0, "Nikolaj is a Danish actor .", ["Denmark", "Actor"]), (2, "He was born in 1970 .", [])]


def test_readable_title_and_evidence_helpers():
    assert readable_title("Soul_Food_-LRB-film-RRB-") == "Soul Food (film)"
    claim = {"evidence": [[[1, 1, "A", 0]], [[2, 2, "B", 3], [2, 2, "C", 1]]]}
    assert evidence_sets(claim) == [[("A", 0)], [("B", 3), ("C", 1)]]
    assert evidence_pages(claim) == {"A", "B", "C"}
    assert evidence_sets({"evidence": [[[4, None, None, None]]]}) == []


def test_allocate_matches_fever_mix():
    mix = {"SUPPORTS": 80035, "REFUTES": 29775, "NOT ENOUGH INFO": 35639}
    counts = allocate(30000, mix)
    assert sum(counts.values()) == 30000
    assert counts == {"SUPPORTS": 16508, "REFUTES": 6141, "NOT ENOUGH INFO": 7351}
    assert sum(allocate(5000, {"a": 1, "b": 1, "c": 1}).values()) == 5000


def test_stratified_sample_exact_and_repeatable():
    claims = [{"id": i, "label": "S" if i % 3 else "R"} for i in range(300)]
    a = stratified_sample(claims, {"S": 20, "R": 10}, random.Random(42))
    b = stratified_sample(claims, {"S": 20, "R": 10}, random.Random(42))
    assert Counter(c["label"] for c in a) == {"S": 20, "R": 10}
    assert [c["id"] for c in a] == [c["id"] for c in b]
    assert len({c["id"] for c in a}) == 30


def test_sparse_bm25_ranks_matching_page_first():
    docs = [
        "Telemundo Telemundo is an American Spanish-language television network",
        "Soul Food (film) Soul Food is a 1997 American comedy-drama film",
        "Damon Albarn Damon Albarn is an English musician",
        "Paris Paris is the capital of France",
    ]
    bm25 = SparseBM25().fit(docs)
    hits = bm25.search(["Telemundo is an English-language television network", "capital of France"], k=2)
    assert hits[0][0][0] == 0
    assert hits[1][0][0] == 3
    assert all(len(h) <= 2 for h in hits)
