"""Stage 3d: corpus-wide check of `unanswerable` drafts.

prepare: BM25 (syllable uni+bigrams) top-K chunks over ALL 11 books for every unanswerable draft,
         written as batch files for verifier agents  -> work/qgen/unans_check/batch_<k>.md
apply  : read verifier verdicts work/qgen/unans_check/verdict_<k>.jsonl and store them in
         the draft as `auto_check`                      -> work/qgen/merged/ptit_draft.jsonl (in place)
Usage: 03d_unanswerable_check.py prepare|apply
"""
import glob
import json
import os
import re
import sys
import unicodedata

from rank_bm25 import BM25Okapi

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DRAFT = os.path.join(ROOT, "work", "qgen", "merged", "ptit_draft.jsonl")
OUT = os.path.join(ROOT, "work", "qgen", "unans_check")
TOP_K, BATCH = 8, 45


def toks(s):
    w = re.findall(r"\w+", unicodedata.normalize("NFC", s).lower())
    return w + [a + "_" + b for a, b in zip(w, w[1:])]


def prepare():
    os.makedirs(OUT, exist_ok=True)
    chunks = [json.loads(l) for l in open(os.path.join(ROOT, "work", "corpus", "all_chunks.jsonl"), encoding="utf-8")]
    bm = BM25Okapi([toks(c["text"]) for c in chunks])
    qs = [json.loads(l) for l in open(DRAFT, encoding="utf-8")]
    un = [q for q in qs if q["type"] == "unanswerable"]
    for k in range(0, len(un), BATCH):
        part = un[k:k + BATCH]
        with open(os.path.join(OUT, f"batch_{k // BATCH + 1:02d}.md"), "w", encoding="utf-8") as f:
            for q in part:
                scores = bm.get_scores(toks(q["question"]))
                top = sorted(range(len(chunks)), key=lambda i: -scores[i])[:TOP_K]
                cited = {e["chunk_id"] for e in q["evidence"]}
                f.write(f"## QUESTION {q['id']}\n**Câu hỏi:** {q['question']}\n"
                        f"**Lý do người sinh cho là không trả lời được:** {q.get('unanswerable_reason')}\n\n")
                for i in top:
                    c = chunks[i]
                    mark = " (chunk gây nhầm do người sinh chỉ ra)" if c["chunk_id"] in cited else ""
                    f.write(f"### {c['chunk_id']}{mark} | {c['book_id']} | trang {c['page_start']}-{c['page_end']}\n{c['text']}\n\n")
    print(f"{len(un)} unanswerable drafts -> {len(range(0, len(un), BATCH))} batches in {OUT}")


def apply():
    verdicts = {}
    for f in sorted(glob.glob(os.path.join(OUT, "verdict_*.jsonl"))):
        for line in open(f, encoding="utf-8"):
            if line.strip():
                v = json.loads(line)
                verdicts[v["id"]] = v
    qs = [json.loads(l) for l in open(DRAFT, encoding="utf-8")]
    n = 0
    for q in qs:
        if q["type"] == "unanswerable" and q["id"] in verdicts:
            v = verdicts[q["id"]]
            q["auto_check"] = {"verdict": v["verdict"], "chunk_id": v.get("chunk_id"), "note": v.get("note", "")}
            n += 1
    with open(DRAFT, "w", encoding="utf-8") as f:
        for q in qs:
            f.write(json.dumps(q, ensure_ascii=False) + "\n")
    miss = [q["id"] for q in qs if q["type"] == "unanswerable" and "auto_check" not in q]
    print(f"applied {n} verdicts; missing: {miss}")
    from collections import Counter
    print(Counter(q["auto_check"]["verdict"] for q in qs if "auto_check" in q))


if __name__ == "__main__":
    {"prepare": prepare, "apply": apply}[sys.argv[1]]()
