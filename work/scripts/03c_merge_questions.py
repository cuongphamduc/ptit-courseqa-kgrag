"""Stage 3c: merge generator outputs, re-validate, resolve pages, de-duplicate, assign final IDs.

Input : work/qgen/raw/*.jsonl
Output: work/qgen/merged/ptit_draft.jsonl        all surviving draft questions with final IDs
        work/qgen/merged/dropped.jsonl           items dropped (validation error or near-duplicate) + reason
        work/qgen/merged/merge_report.json
"""
import collections
import glob
import importlib.util
import json
import os
import re
import unicodedata

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
spec = importlib.util.spec_from_file_location("v", os.path.join(ROOT, "work", "scripts", "03b_validate_questions.py"))
V = importlib.util.module_from_spec(spec)
spec.loader.exec_module(V)

DUP_THRESHOLD = 0.80  # char 3-5 gram TF-IDF cosine between questions
TYPE_ORDER = {"single": 0, "multi_intra": 1, "multi_cross": 2, "unanswerable": 3}


def load_paras():
    paras = collections.defaultdict(list)
    for f in glob.glob(os.path.join(ROOT, "work", "corpus", "clean", "*.paras.jsonl")):
        b = os.path.basename(f).split(".")[0]
        for line in open(f, encoding="utf-8"):
            p = json.loads(line)
            paras[b].append((p["page"], p["page_end"], V.norm(p["text"])))
    return paras


def resolve_page(paras, book, span, chunk):
    s = V.norm(span)
    key = s[:60]
    for page, page_end, text in paras[book]:
        pos = text.find(key)
        if pos >= 0 and chunk["page_start"] <= page <= chunk["page_end"] + 1:
            if page_end > page:  # paragraph spans pages: interpolate by position
                page = page + int((page_end - page + 1) * pos / max(1, len(text)))
            return min(max(page, chunk["page_start"]), chunk["page_end"])
    return chunk["page_start"]


def main():
    out_dir = os.path.join(ROOT, "work", "qgen", "merged")
    os.makedirs(out_dir, exist_ok=True)
    paras = load_paras()
    items, dropped = [], []
    for f in sorted(glob.glob(os.path.join(ROOT, "work", "qgen", "raw", "*.jsonl"))):
        src = os.path.basename(f)
        for n, line in enumerate(open(f, encoding="utf-8"), 1):
            if not line.strip():
                continue
            try:
                it = json.loads(line)
            except json.JSONDecodeError:
                dropped.append({"source": src, "line": n, "reason": "invalid JSON"})
                continue
            _, errs, warns = V.check(it, n)
            it["_source"] = src
            if errs:
                dropped.append(dict(it, reason="validation: " + "; ".join(errs)))
                continue
            it["auto_warnings"] = warns
            items.append(it)

    # resolve pages
    for it in items:
        for e in it["evidence"]:
            ch = V.CHUNKS[e["chunk_id"]]
            e["page"] = resolve_page(paras, ch["book_id"], e["span"], ch)
            e["chapter"] = ch["chapter"]
            e["section"] = ch["section"]

    # near-duplicate removal (per split, across all books)
    keep = [True] * len(items)
    for split in ("test", "dev"):
        idx = [i for i, it in enumerate(items) if it["split"] == split]
        if len(idx) < 2:
            continue
        texts = [V.norm(items[i]["question"]) for i in idx]
        tf = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=1).fit_transform(texts)
        sim = cosine_similarity(tf)
        for a in range(len(idx)):
            if not keep[idx[a]]:
                continue
            for b in range(a + 1, len(idx)):
                if keep[idx[b]] and sim[a, b] >= DUP_THRESHOLD:
                    keep[idx[b]] = False
                    dropped.append(dict(items[idx[b]], reason=f"near-duplicate of {items[idx[a]]['tmp_id']} (cos={sim[a, b]:.2f})"))
    # also forbid the same question across splits
    test_q = {V.norm(it["question"]) for it, k in zip(items, keep) if k and it["split"] == "test"}
    for i, it in enumerate(items):
        if keep[i] and it["split"] == "dev" and V.norm(it["question"]) in test_q:
            keep[i] = False
            dropped.append(dict(it, reason="same question in test"))
    items = [it for it, k in zip(items, keep) if k]

    # final IDs
    order = sorted(items, key=lambda x: (x["split"], x["book_id"], TYPE_ORDER[x["type"]], x["tmp_id"]))
    counters = collections.Counter()
    for it in order:
        counters[it["split"]] += 1
        it["id"] = f"PTIT-{it['split'].upper()}-{counters[it['split']]:04d}"
        it.setdefault("generator", "claude-opus-5-5")
        it["status"] = "draft"
    fields = ["id", "split", "book_id", "type", "cross_type", "question", "answer", "evidence", "reasoning_chain",
              "unanswerable_reason", "generator", "status", "tmp_id", "auto_warnings", "_source"]
    with open(os.path.join(out_dir, "ptit_draft.jsonl"), "w", encoding="utf-8") as f:
        for it in order:
            f.write(json.dumps({k: it.get(k) for k in fields}, ensure_ascii=False) + "\n")
    with open(os.path.join(out_dir, "dropped.jsonl"), "w", encoding="utf-8") as f:
        for it in dropped:
            f.write(json.dumps(it, ensure_ascii=False) + "\n")
    rep = {
        "kept": len(order), "dropped": len(dropped),
        "dropped_reasons": collections.Counter(re.sub(r" of .*|: .*", "", d["reason"]) for d in dropped),
        "by_split_type": collections.Counter(f"{i['split']}/{i['type']}" for i in order),
        "by_split_book": collections.Counter(f"{i['split']}/{i['book_id']}" for i in order),
        "by_split_cross_type": collections.Counter(f"{i['split']}/{i.get('cross_type')}" for i in order if i["type"] == "multi_cross"),
    }
    json.dump(rep, open(os.path.join(out_dir, "merge_report.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(json.dumps(rep, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
