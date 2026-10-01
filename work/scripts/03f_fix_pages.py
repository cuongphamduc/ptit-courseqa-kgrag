"""Stage 3f: set each evidence `page` to the PDF page whose text contains the span (searching the chunk's page
range +-1). Falls back to the merge-time estimate. Writes work/qgen/merged/ptit_draft.jsonl in place and
work/qgen/merged/page_check.json."""
import collections
import json
import os
import re
import subprocess
import unicodedata

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DRAFT = os.path.join(ROOT, "work", "qgen", "merged", "ptit_draft.jsonl")
BOOKS = json.load(open(os.path.join(ROOT, "work", "corpus", "books.json"), encoding="utf-8"))
CH = {json.loads(l)["chunk_id"]: json.loads(l) for l in open(os.path.join(ROOT, "work", "corpus", "all_chunks.jsonl"), encoding="utf-8")}
_cache = {}


def n(s):
    return re.sub(r"\W+", "", unicodedata.normalize("NFC", s).replace("ƣ", "ư").replace("Ƣ", "Ư").lower())


def page_text(b, p):
    if (b, p) not in _cache:
        _cache[(b, p)] = n(subprocess.run(["pdftotext", "-f", str(p), "-l", str(p), os.path.join(ROOT, "dataset", "PTIT", BOOKS[b]), "-"],
                                          capture_output=True, text=True).stdout)
    return _cache[(b, p)]


def main():
    qs = [json.loads(l) for l in open(DRAFT, encoding="utf-8")]
    stat = collections.Counter()
    for q in qs:
        for e in q["evidence"]:
            c = CH[e["chunk_id"]]
            s = n(e["span"])
            keys = [s[:30], s[len(s) // 2:len(s) // 2 + 30], s[-30:]]
            pages = list(range(max(1, c["page_start"] - 1), c["page_end"] + 2))
            found = None
            for key in keys:
                found = next((p for p in pages if key and key in page_text(e["book_id"], p)), None)
                if found:
                    break
            if found:
                stat["located" if found == e["page"] else "corrected"] += 1
                e["page"] = found
            else:
                stat["not_located_kept_estimate"] += 1
    with open(DRAFT, "w", encoding="utf-8") as f:
        for q in qs:
            f.write(json.dumps(q, ensure_ascii=False) + "\n")
    json.dump(stat, open(os.path.join(ROOT, "work", "qgen", "merged", "page_check.json"), "w"), indent=1)
    print(dict(stat))


if __name__ == "__main__":
    main()
