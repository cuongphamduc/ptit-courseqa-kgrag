"""Datasets and corpora in one schema.

question: {id, dataset, split, question, golds[list], type, unanswerable, gold_chunks[list], book_id?}
chunk:    {chunk_id, text, title, meta{...}}
"""
import functools
import json
import os

WORK = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ROOT = os.path.dirname(WORK)
BENCH = os.path.join(WORK, "benchmark")


def _jsonl(p):
    with open(p, encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


@functools.lru_cache(maxsize=None)
def corpus(name):
    """name: 'ptit' | 'viquad_test' | 'viquad_dev'"""
    if name == "ptit":
        books = json.load(open(os.path.join(WORK, "corpus", "books.json"), encoding="utf-8"))
        out = []
        for c in _jsonl(os.path.join(WORK, "corpus", "all_chunks.jsonl")):
            title = os.path.splitext(os.path.basename(books[c["book_id"]]))[0]
            out.append({"chunk_id": c["chunk_id"], "text": c["text"], "title": title,
                        "meta": {k: c.get(k) for k in ("book_id", "chapter", "chapter_no", "section", "sections_covered", "page_start", "page_end")}})
        return out
    return [{"chunk_id": c["chunk_id"], "text": c["text"], "title": c["title"], "meta": {"book_id": c["title"]}}
            for c in _jsonl(os.path.join(BENCH, f"{name.replace('viquad', 'viquad_corpus')}.jsonl"))]


def corpus_for(dataset, split):
    return "ptit" if dataset == "ptit" else f"viquad_{split}"


@functools.lru_cache(maxsize=None)
def questions(dataset, split):
    rows = _jsonl(os.path.join(BENCH, f"{dataset}_{split}.jsonl"))
    out = []
    for r in rows:
        if dataset == "ptit":
            una = r["type"] == "unanswerable"
            out.append({"id": r["id"], "dataset": dataset, "split": split, "question": r["question"],
                        "golds": [] if una else [r["answer"]], "type": r["type"], "cross_type": r.get("cross_type"),
                        "unanswerable": una, "gold_chunks": [] if una else sorted({e["chunk_id"] for e in r["evidence"]}),
                        "book_id": r["book_id"]})
        else:
            out.append({"id": r["id"], "dataset": dataset, "split": split, "question": r["question"], "golds": r["answers"],
                        "type": r["type"], "cross_type": None, "unanswerable": r["is_impossible"],
                        "gold_chunks": [] if r["is_impossible"] else [r["gold_context_id"]], "book_id": r["title"]})
    return tuple(out)
