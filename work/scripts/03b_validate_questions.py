"""Stage 3b: validate a generated question JSONL file (used by generator agents and by the merger).

Usage: work/.venv/bin/python work/scripts/03b_validate_questions.py work/qgen/raw/<file>.jsonl [--agent book:X|pair:A+B]
Prints one line per problem ("ERROR" must be fixed, "WARN" should be reviewed) and a quota summary.
Exit code 1 if any ERROR.
"""
import argparse
import collections
import json
import os
import re
import sys
import unicodedata

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
CHUNKS = {}
for line in open(os.path.join(ROOT, "work", "corpus", "all_chunks.jsonl"), encoding="utf-8"):
    c = json.loads(line)
    CHUNKS[c["chunk_id"]] = c
PLAN = json.load(open(os.path.join(ROOT, "work", "qgen", "plan.json"), encoding="utf-8"))
POOL = {cid: split for b in PLAN["pools"].values() for cid, split in b.items()}
TYPES = {"single", "multi_intra", "multi_cross", "unanswerable"}
REQUIRED = ["tmp_id", "split", "book_id", "type", "question", "answer", "evidence", "reasoning_chain"]


def norm(s):
    s = unicodedata.normalize("NFC", s or "")
    s = s.replace("“", '"').replace("”", '"').replace("‘", "'").replace("’", "'")
    return re.sub(r"\s+", " ", s).strip().lower()


def words(s):
    return re.findall(r"\w+", norm(s))


def longest_common_run(a, b):
    """Length (in words) of the longest common contiguous word sequence."""
    aw, bw = words(a), words(b)
    if not aw or not bw:
        return 0
    idx = collections.defaultdict(list)
    for j, w in enumerate(bw):
        idx[w].append(j)
    best, prev = 0, {}
    for i, w in enumerate(aw):
        cur = {}
        for j in idx.get(w, ()):
            cur[j] = prev.get(j - 1, 0) + 1
            best = max(best, cur[j])
        prev = cur
    return best


def check(item, n):
    errs, warns = [], []
    tag = item.get("tmp_id", f"line{n}")
    for k in REQUIRED:
        if k not in item:
            errs.append(f"missing field '{k}'")
    if errs:
        return tag, errs, warns
    t, split = item["type"], item["split"]
    if t not in TYPES:
        errs.append(f"bad type {t}")
    if split not in ("test", "dev"):
        errs.append(f"bad split {split}")
    q = item["question"] or ""
    if len(words(q)) < 6:
        errs.append("question too short")
    if not q.strip().endswith("?"):
        warns.append("question does not end with '?'")
    if re.search(r"(đoạn văn|văn bản trên|đoạn trên|tài liệu này|chương này|bài giảng này|giáo trình này|trong sách)", q, re.I):
        errs.append("question refers to 'the passage/this book' - must be self-contained")
    ev = item["evidence"]
    if not isinstance(ev, list) or not ev:
        errs.append("evidence must be a non-empty list")
        return tag, errs, warns
    cids, books, chaps = [], set(), set()
    for e in ev:
        cid = e.get("chunk_id")
        if cid not in CHUNKS:
            errs.append(f"unknown chunk_id {cid}")
            continue
        ch = CHUNKS[cid]
        cids.append(cid)
        books.add(ch["book_id"])
        chaps.add((ch["book_id"], str(ch["chapter_no"])))
        if e.get("book_id") != ch["book_id"]:
            errs.append(f"{cid}: book_id should be {ch['book_id']}")
        if POOL.get(cid) != split:
            errs.append(f"{cid}: chunk belongs to the {POOL.get(cid)} pool, not {split}")
        span = e.get("span") or ""
        if len(span) < 15:
            errs.append(f"{cid}: span too short")
        elif norm(span) not in norm(ch["text"]):
            errs.append(f"{cid}: span is not a verbatim substring of the chunk text")
        if len(span) > 600:
            warns.append(f"{cid}: span longer than 600 chars")
        run = longest_common_run(q, ch["text"])
        if run >= 10:
            errs.append(f"{cid}: question copies {run} consecutive words from the text")
        elif run >= 8:
            warns.append(f"{cid}: question shares {run} consecutive words with the text")
    if len(set(cids)) != len(cids):
        warns.append("duplicate chunk_id in evidence")
    ans = item["answer"]
    if t == "unanswerable":
        if ans not in (None, "null"):
            errs.append("unanswerable must have answer null")
        if not item.get("unanswerable_reason"):
            errs.append("missing unanswerable_reason")
    else:
        if not ans or not str(ans).strip():
            errs.append("answer missing")
        else:
            aw = words(ans)
            if len(aw) > 45:
                warns.append(f"answer long ({len(aw)} words)")
            evtxt = " ".join(norm(e.get("span", "")) for e in ev)
            evw = set(re.findall(r"\w+", evtxt))
            cov = sum(1 for w in aw if w in evw) / max(1, len(aw))
            if cov < 0.5:
                errs.append(f"answer not grounded in evidence spans (word coverage {cov:.2f})")
            elif cov < 0.7:
                warns.append(f"answer weakly grounded (word coverage {cov:.2f})")
    if t == "single" and len(set(cids)) != 1:
        errs.append("single must cite exactly one chunk")
    if t == "multi_intra":
        if len(set(cids)) < 2:
            errs.append("multi_intra needs >= 2 distinct chunks")
        if len(books) != 1:
            errs.append("multi_intra must stay within one book")
    if t == "multi_cross":
        ct = item.get("cross_type")
        if ct not in ("chapter", "book"):
            errs.append("multi_cross needs cross_type chapter|book")
        if len(set(cids)) < 2:
            errs.append("multi_cross needs >= 2 distinct chunks")
        if ct == "book" and len(books) < 2:
            errs.append("cross_type=book but evidence from one book")
        if ct == "chapter" and len(chaps) < 2:
            errs.append("cross_type=chapter but evidence from one chapter")
    if t.startswith("multi") and len(norm(item["reasoning_chain"])) < 30:
        errs.append("reasoning_chain too short for a multi-hop question")
    if item["book_id"] not in books and t != "unanswerable":
        errs.append("book_id (primary) must be one of the evidence books")
    return tag, errs, warns


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("path")
    ap.add_argument("--agent", default=None)
    a = ap.parse_args()
    items, n_err = [], 0
    for n, line in enumerate(open(a.path, encoding="utf-8"), 1):
        if not line.strip():
            continue
        try:
            items.append((n, json.loads(line)))
        except json.JSONDecodeError as ex:
            print(f"ERROR line{n}: invalid JSON ({ex})")
            n_err += 1
    ids = collections.Counter(it.get("tmp_id") for _, it in items)
    for k, v in ids.items():
        if v > 1:
            print(f"ERROR {k}: duplicate tmp_id")
            n_err += 1
    qs = collections.Counter(norm(it.get("question")) for _, it in items)
    for k, v in qs.items():
        if v > 1:
            print(f"ERROR duplicate question text: {k[:80]}")
            n_err += 1
    for n, it in items:
        tag, errs, warns = check(it, n)
        for e in errs:
            print(f"ERROR {tag}: {e}")
        for w in warns:
            print(f"WARN  {tag}: {w}")
        n_err += len(errs)
    cnt = collections.Counter()
    for _, it in items:
        t = it.get("type")
        if t == "multi_cross":
            t = f"multi_cross_{it.get('cross_type')}"
        cnt[(it.get("split"), t)] += 1
    print("\nCOUNTS:", dict(sorted(cnt.items())))
    if a.agent:
        ag = next((x for x in PLAN["agents"] if x["agent"] == a.agent), None)
        if ag:
            for split, q in ag["quota"].items():
                for t, need in q.items():
                    if t == "primary_split":
                        continue
                    have = cnt[(split, t)]
                    flag = "OK " if have >= need else "LOW"
                    print(f"QUOTA {flag} {split}/{t}: {have}/{need}")
    print(f"\nTOTAL items={len(items)} errors={n_err}")
    sys.exit(1 if n_err else 0)


if __name__ == "__main__":
    main()
