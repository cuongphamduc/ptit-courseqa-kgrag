"""Stage 1 inventory: PTIT textbooks + UIT-ViQuAD2.0 (reproducible).

Outputs:
  work/inventory/ptit_books.csv      per-book stats
  work/inventory/ptit_overlap.csv    pairwise 8-gram shingle overlap
  work/inventory/viquad_splits.json  split sizes / unanswerable ratio
Run: work/.venv/bin/python work/scripts/01_inventory.py
"""
import csv
import glob
import itertools
import json
import os
import re
import subprocess
import unicodedata

import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
PTIT = os.path.join(ROOT, "dataset", "PTIT")
VIQUAD = os.path.join(ROOT, "work", "data", "UIT-ViQuAD2.0", "data")
OUT = os.path.join(ROOT, "work", "inventory")
os.makedirs(OUT, exist_ok=True)


def pdfinfo(path):
    out = subprocess.run(["pdfinfo", path], capture_output=True, text=True).stdout
    info = {}
    for line in out.splitlines():
        if ":" in line:
            k, v = line.split(":", 1)
            info[k.strip()] = v.strip()
    return info


def page_text(path, page):
    return subprocess.run(["pdftotext", "-f", str(page), "-l", str(page), path, "-"],
                          capture_output=True, text=True).stdout


def full_text(path):
    return subprocess.run(["pdftotext", path, "-"], capture_output=True, text=True).stdout


rows, shingles = [], {}
for path in sorted(glob.glob(os.path.join(PTIT, "*", "*.pdf"))):
    info = pdfinfo(path)
    pages = int(info.get("Pages", 0))
    low_text = sum(1 for p in range(1, pages + 1)
                   if len(re.sub(r"\s", "", page_text(path, p))) < 50)
    raw = full_text(path)
    text = unicodedata.normalize("NFC", raw)
    words = re.findall(r"\w+", text.lower().replace("ƣ", "ư"))
    name = os.path.basename(path)
    shingles[name] = {tuple(words[i:i + 8]) for i in range(len(words) - 8)}
    rows.append({
        "group": os.path.basename(os.path.dirname(path)),
        "file": name,
        "pages": pages,
        "size_mb": round(os.path.getsize(path) / 2**20, 1),
        "producer": info.get("Producer", ""),
        "words": len(words),
        "pages_lt50chars": low_text,
        "bad_char_U01A3": raw.count("ƣ") + raw.count("Ƣ"),
        "needs_ocr": "no" if low_text / max(pages, 1) < 0.05 else "check",
    })

with open(os.path.join(OUT, "ptit_books.csv"), "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0]))
    w.writeheader()
    w.writerows(rows)

with open(os.path.join(OUT, "ptit_overlap.csv"), "w", newline="", encoding="utf-8") as f:
    w = csv.writer(f)
    w.writerow(["book_a", "book_b", "overlap_8gram_min"])
    pairs = []
    for a, b in itertools.combinations(shingles, 2):
        inter = len(shingles[a] & shingles[b])
        pairs.append((a, b, round(inter / max(1, min(len(shingles[a]), len(shingles[b]))), 4)))
    for p in sorted(pairs, key=lambda x: -x[2]):
        w.writerow(p)

vq = {}
for split in ["train", "validation", "test"]:
    d = pd.read_parquet(os.path.join(VIQUAD, f"{split}-00000-of-00001.parquet"))
    labelled = int(d.answers.notna().sum())
    vq[split] = {
        "n_questions": len(d),
        "n_titles": int(d.title.nunique()),
        "n_contexts": int(d.context.nunique()),
        "n_with_answers_field": labelled,
        "n_unanswerable": int(d.is_impossible.sum()),
        "unanswerable_ratio": round(float(d.is_impossible.mean()), 4),
        "has_public_labels": labelled > 0,
    }
with open(os.path.join(OUT, "viquad_splits.json"), "w", encoding="utf-8") as f:
    json.dump(vq, f, ensure_ascii=False, indent=2)

print(pd.DataFrame(rows).to_string(index=False))
print(json.dumps(vq, ensure_ascii=False, indent=2))
