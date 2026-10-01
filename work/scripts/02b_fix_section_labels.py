"""Stage 2b: repair `section` metadata of chunks without re-chunking.

Some figure captions ("2.8 [2].", "4.31 Hình 4.31 ...") and exercise prompts ("4.5 Hãy trình bày ...") were
accepted as numbered headings by 02_preprocess.py (v1). Because of the monotonic-numbering rule, a false heading
with a high number also hid the real headings after it. Re-chunking would change chunk_ids already used as question
evidence, so this script:
  1. re-derives the heading sequence with STRICT_HEADINGS=True (captions/exercises rejected);
  2. locates each heading line inside the unchanged v1 chunk texts (headings are consecutive, in reading order);
  3. rewrites only `section` and `sections_covered` (chunk text, ids and pages unchanged).

Input : work/corpus/all_chunks.v1.jsonl (created from all_chunks.jsonl on first run), dataset/PTIT PDFs
Output: work/corpus/all_chunks.jsonl, work/corpus/chunks/<book>.jsonl, work/corpus/clean/<book>.paras.strict.jsonl,
        work/corpus/section_fix_report.json
"""
import collections
import importlib.util
import json
import os
import re
import shutil
import unicodedata

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
C = os.path.join(ROOT, "work", "corpus")
spec = importlib.util.spec_from_file_location("pre", os.path.join(ROOT, "work", "scripts", "02_preprocess.py"))
pre = importlib.util.module_from_spec(spec)
spec.loader.exec_module(pre)
pre.STRICT_HEADINGS = True


def n(s):
    return re.sub(r"\s+", " ", unicodedata.normalize("NFC", s)).strip().lower()


def main():
    src, v1 = os.path.join(C, "all_chunks.jsonl"), os.path.join(C, "all_chunks.v1.jsonl")
    if not os.path.exists(v1):
        shutil.copy(src, v1)
    chunks = [json.loads(l) for l in open(v1, encoding="utf-8")]
    by_book = collections.defaultdict(list)
    for c in chunks:
        by_book[c["book_id"]].append(c)
    report = {}
    for book, rel in pre.BOOKS.items():
        paras = pre.build_paragraphs(book, pre.extract_pages(os.path.join(ROOT, "dataset", "PTIT", rel)))
        with open(os.path.join(C, "clean", f"{book}.paras.strict.jsonl"), "w", encoding="utf-8") as f:
            for p in paras:
                f.write(json.dumps(p, ensure_ascii=False) + "\n")
        # heading events in reading order: chapter openers and (strict) numbered sections
        heads = []
        for p in paras:
            if p["heading"]:
                kind = "section" if p["section_no"] else "chapter"
                heads.append({"kind": kind, "no": p["section_no"], "title": p["section"], "chapter_no": p["chapter_no"],
                              "chapter": p["chapter"], "text": n(p["text"])})
        cs = by_book[book]
        # headings are stand-alone lines in the chunk text: match whole normalised lines, in reading order
        lines = [(j, li, n(l)) for j, c in enumerate(cs) for li, l in enumerate(c["text"].split("\n"))]
        idx = collections.defaultdict(list)
        for k, (_, _, l) in enumerate(lines):
            idx[l].append(k)
        cursor = 0
        for h in heads:
            h["loc"] = None
            cand = [k for k in idx.get(h["text"], []) if k >= cursor]
            if cand:
                k = cand[0]
                h["loc"] = (lines[k][0], lines[k][1])
                cursor = k + 1
            elif h["kind"] == "section" and len(h["text"]) >= 12:
                # fallback: in v1 the heading may have been merged into a paragraph -> substring search, next 6 chunks
                for k in range(cursor, len(lines)):
                    if lines[k][0] > lines[min(cursor, len(lines) - 1)][0] + 6:
                        break
                    if h["text"] in lines[k][2]:
                        h["loc"] = (lines[k][0], lines[k][1] + 0.25)
                        cursor = k
                        break
        located = sum(1 for h in heads if h["loc"])
        # headings not found (e.g. chapter openers removed as running headers in v1) anchor just before the next found one
        nxt = None
        for h in reversed(heads):
            if h["loc"] is None:
                h["loc"] = (nxt[0], nxt[1] - 0.5) if nxt else (len(cs) - 1, 10 ** 9)
            else:
                nxt = h["loc"]
        events = collections.defaultdict(list)
        for h in heads:
            events[h["loc"][0]].append(h)
        cur_ch, cur_chno, cur_sec, changed = None, None, "", 0
        for j, c in enumerate(cs):
            evs = sorted(events.get(j, []), key=lambda h: h["loc"][1])
            start_ch, start_chno, start_sec = cur_ch, cur_chno, cur_sec
            covered = set()
            for h in evs:
                if h["kind"] == "chapter":
                    cur_ch, cur_chno, cur_sec = h["chapter"], h["chapter_no"], ""
                else:
                    cur_sec = f"{h['no']} {h['title']}"
                    covered.add(h["no"])
                    cur_ch, cur_chno = h["chapter"], h["chapter_no"]
                if h["loc"][1] <= 1:  # the chunk opens with this heading (line 0/1, overlap-free)
                    start_ch, start_chno = cur_ch, cur_chno
                    start_sec = cur_sec
            if start_ch is None:
                start_ch, start_chno = c["chapter"], c["chapter_no"]
            if start_sec:
                covered.add(start_sec.split(" ", 1)[0])
            if (start_sec, start_chno) != (c["section"], c["chapter_no"]):
                changed += 1
            c["chapter"], c["chapter_no"], c["section"] = start_ch, start_chno, start_sec
            c["sections_covered"] = sorted(covered, key=lambda s: tuple(int(x) for x in s.split(".")))
        report[book] = {"headings": len(heads), "located": located, "chunks_relabelled": changed, "chapters": dict(collections.Counter(str(c["chapter_no"]) for c in cs))}
    with open(src, "w", encoding="utf-8") as f:
        for c in chunks:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")
    for b, cs in by_book.items():
        with open(os.path.join(C, "chunks", f"{b}.jsonl"), "w", encoding="utf-8") as f:
            for c in cs:
                f.write(json.dumps(c, ensure_ascii=False) + "\n")
    json.dump(report, open(os.path.join(C, "section_fix_report.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    for b, r in report.items():
        print(b, r)


if __name__ == "__main__":
    main()
