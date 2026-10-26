"""Stage 9 — build the public release packages for Hugging Face Datasets, GitHub and Zenodo.

Licence: CC BY 4.0. The full textbook text is NOT redistributed (author decision, 2026-10-01):
  - chunk text is replaced by chunk_id + pages + SHA-256 of the NFC text (rebuild with 02_preprocess.py from the PDFs);
  - annotation sheets are exported without the `chunks` sheet and without the retrieved `context` column;
  - the `texts` field of LightRAG retrieval logs is dropped;
  - evidence spans (short verbatim excerpts, median 39 words) are kept.
UIT-ViQuAD2.0 has no stated licence on its dataset card: only question ids are released, rebuilt by 04_viquad_subset.py.

Output: work/release/{core,huggingface/PTIT-CourseQA,github/ptit-courseqa-kgrag,zenodo}
"""
import csv
import glob
import hashlib
import json
import os
import shutil
import unicodedata
import zipfile

import openpyxl

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
W = os.path.join(ROOT, "work")
RES = os.path.join(ROOT, "results")
REL = os.path.join(W, "release")
TPL = os.path.join(W, "release_templates")
CORE = os.path.join(REL, "core")
VERSION = "1.0.1"

GROUP_EN = {"An toàn thông tin": "Information security", "Công nghệ thông tin": "Information technology",
            "Điện tử": "Electronics", "Marketing": "Marketing"}


def wjsonl(path, rows):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def rjsonl(path):
    return [json.loads(l) for l in open(path, encoding="utf-8")]


def wcsv(path, header, rows):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)


def sheet_to_csv(xlsx, sheet, out, drop=()):
    ws = openpyxl.load_workbook(xlsx, read_only=True)[sheet]
    rows = list(ws.iter_rows(values_only=True))
    keep = [i for i, h in enumerate(rows[0]) if h is not None and h not in drop]
    wcsv(out, [rows[0][i] for i in keep], [[r[i] for i in keep] for r in rows[1:] if any(v is not None for v in r)])


def build_core():
    shutil.rmtree(CORE, ignore_errors=True)
    # 1. benchmark
    for name in ("ptit_test", "ptit_dev", "ptit_accepted_pool"):
        shutil.copy(os.path.join(W, "benchmark", f"{name}.jsonl"), os.path.join(_mk(CORE, "data"), f"{name}.jsonl"))
    shutil.copy(os.path.join(W, "benchmark", "DATASHEET.md"), os.path.join(CORE, "DATASHEET_vi.md"))
    # 2. corpus index without text
    chunks = rjsonl(os.path.join(W, "corpus", "all_chunks.jsonl"))
    idx = []
    for c in chunks:
        t = unicodedata.normalize("NFC", c["text"])
        r = {k: v for k, v in c.items() if k != "text"}
        r["n_chars"] = len(t)
        r["text_sha256"] = hashlib.sha256(t.encode("utf-8")).hexdigest()
        idx.append(r)
    wjsonl(os.path.join(CORE, "corpus", "chunk_index.jsonl"), idx)
    books = json.load(open(os.path.join(W, "corpus", "books.json"), encoding="utf-8"))
    inv = {r["file"]: r for r in csv.DictReader(open(os.path.join(W, "inventory", "ptit_books.csv"), encoding="utf-8"))}
    meta = []
    for bid, rel in books.items():
        group, fn = rel.split("/", 1)
        stem = fn[:-4]
        title, year = stem.rsplit(" - ", 1)
        meta.append({"book_id": bid, "field_vi": group, "field_en": GROUP_EN[group], "title_vi": title, "year": int(year),
                     "pdf_file": rel, "pdf_pages": int(inv[fn]["pages"]),
                     "n_chunks": sum(1 for c in chunks if c["book_id"] == bid)})
    json.dump(meta, open(os.path.join(CORE, "corpus", "books.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    # 3. ViQuAD subset: ids only
    vq = _mk(CORE, "viquad_subset")
    for split, ids in (("test", "viquad_subset_ids.txt"), ("dev", "viquad_dev_ids.txt")):
        shutil.copy(os.path.join(W, "benchmark", ids), os.path.join(vq, f"{split}_ids.txt"))
        rows = rjsonl(os.path.join(W, "benchmark", f"viquad_{split}.jsonl"))
        wcsv(os.path.join(vq, f"{split}_index.csv"), ["id", "uit_id", "type", "is_impossible", "gold_context_id"],
             [[r["id"], r["uit_id"], r["type"], r["is_impossible"], r["gold_context_id"]] for r in rows])
    shutil.copy(os.path.join(RES, "configs", "viquad_config.json"), os.path.join(vq, "viquad_config.json"))
    # 4. annotation (labels only)
    an = os.path.join(ROOT, "annotation")
    ao = _mk(CORE, "annotation")
    shutil.copy(os.path.join(an, "GUIDELINES.md"), os.path.join(ao, "GUIDELINES_vi.md"))
    sheet_to_csv(os.path.join(an, "round1", "annotation_A.xlsx"), "questions", os.path.join(ao, "round1_annotator_A.csv"))
    sheet_to_csv(os.path.join(an, "round1", "annotation_B_blind.xlsx"), "questions", os.path.join(ao, "round1_annotator_B_blind.csv"))
    sheet_to_csv(os.path.join(an, "disagreements.xlsx"), "disagreements", os.path.join(ao, "round1_adjudication.csv"))
    for g in ("A", "B"):
        sheet_to_csv(os.path.join(an, "halluc", f"halluc_{g}.xlsx"), "answers", os.path.join(ao, f"hallucination_annotator_{g}.csv"),
                     drop=("context",))
    shutil.copy(os.path.join(RES, "halluc", "manual_labels_decoded.jsonl"), os.path.join(ao, "hallucination_labels_decoded.jsonl"))
    # 5. system outputs and statistics (results/, minus superseded runs, minus chunk text)
    out = _mk(CORE, "outputs")
    for src in glob.glob(os.path.join(RES, "**", "*"), recursive=True):
        rel = os.path.relpath(src, RES)
        if os.path.isdir(src) or rel.startswith(os.path.join("dev", "v1_superseded")):
            continue
        dst = os.path.join(out, rel)
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        if rel.startswith("retrieval") and rel.endswith(".jsonl"):
            wjsonl(dst, [{k: v for k, v in r.items() if k != "texts"} for r in rjsonl(src)])
        else:
            shutil.copy(src, dst)
    _check_no_fulltext(chunks)


def copy_tpl(name, dst):
    """copy a template, filling {{HF_OWNER}} / {{GH_OWNER}} / {{ZENODO_DOI}} from the environment when set"""
    t = open(os.path.join(TPL, name), encoding="utf-8").read()
    for k in ("HF_OWNER", "GH_OWNER", "ZENODO_DOI"):
        if os.environ.get(k):
            t = t.replace("{{" + k + "}}", os.environ[k])
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    open(dst, "w", encoding="utf-8").write(t)


def _mk(*p):
    d = os.path.join(*p)
    os.makedirs(d, exist_ok=True)
    return d


def _check_no_fulltext(chunks):
    """Excerpts are allowed (evidence spans, answers quoting the context); whole chunks are not.
    Each chunk is cut into non-overlapping 300-character windows. Fail if any file contains >= 50% of the windows of
    any chunk; report the share of all corpus windows that appear anywhere in the release."""
    win = {}
    for c in chunks:
        t = unicodedata.normalize("NFC", c["text"])
        win[c["chunk_id"]] = [t[s:s + 300] for s in range(0, len(t) - 299, 300)]
    found, worst = set(), (0.0, None, None)
    for f in glob.glob(os.path.join(CORE, "**", "*"), recursive=True):
        if os.path.isdir(f) or not f.endswith((".jsonl", ".json", ".csv", ".md", ".txt")):
            continue
        txt = unicodedata.normalize("NFC", open(f, encoding="utf-8").read())
        for cid, ws in win.items():
            hit = [i for i, w in enumerate(ws) if w in txt]
            found.update((cid, i) for i in hit)
            if ws and len(hit) / len(ws) > worst[0]:
                worst = (len(hit) / len(ws), cid, os.path.relpath(f, CORE))
    total = sum(len(ws) for ws in win.values())
    rep = {"windows_total": total, "windows_in_release": len(found), "share_in_release": round(len(found) / total, 4),
           "max_share_of_one_chunk_in_one_file": round(worst[0], 3), "chunk": worst[1], "file": worst[2]}
    json.dump(rep, open(os.path.join(REL, "fulltext_check.json"), "w"), indent=1)
    print("full-text check:", rep)
    if worst[0] >= 0.5:
        raise SystemExit("a whole chunk is reproduced in the release")


def build_hf():
    d = os.path.join(REL, "huggingface", "PTIT-CourseQA")
    shutil.rmtree(d, ignore_errors=True)
    shutil.copytree(CORE, d)
    copy_tpl("HF_README.md", os.path.join(d, "README.md"))
    copy_tpl("LICENSE", os.path.join(d, "LICENSE"))


CODE = [("work/rag", "rag"), ("work/scripts", "scripts"), ("paper/scripts/make_numbers.py", "paper/scripts/make_numbers.py"),
        ("paper/figures/scripts", "paper/figures/scripts"), ("work/qgen/INSTRUCTIONS.md", "prompts/question_generation_INSTRUCTIONS_vi.md"),
        ("annotation/GUIDELINES.md", "prompts/annotation_GUIDELINES_vi.md")]


def build_github():
    d = os.path.join(REL, "github", "ptit-courseqa-kgrag")
    shutil.rmtree(d, ignore_errors=True)
    for src, dst in CODE:
        s, t = os.path.join(ROOT, src), os.path.join(d, dst)
        if os.path.isdir(s):
            shutil.copytree(s, t, ignore=shutil.ignore_patterns("__pycache__", "*.pyc", "*.pid", "*.log"))
        else:
            os.makedirs(os.path.dirname(t), exist_ok=True)
            shutil.copy(s, t)
    # same layout as the working tree: scripts expect ROOT/work/{rag,scripts}
    os.makedirs(os.path.join(d, "work"), exist_ok=True)
    for sub in ("rag", "scripts"):
        shutil.move(os.path.join(d, sub), os.path.join(d, "work", sub))
    ctl = os.path.join(d, "work", "scripts", "ollama_ctl.sh")
    s = open(ctl).read().replace("W=" + W, 'W="$(cd "$(dirname "$0")/.." && pwd)"')
    open(ctl, "w").write(s)
    srv = os.path.join(d, "work", "scripts", "ollama_serve.sh")
    s = open(srv).read().replace("export OLLAMA_MODELS=/usr/share/ollama/.ollama/models",
                                 "export OLLAMA_MODELS=${OLLAMA_MODELS:-/usr/share/ollama/.ollama/models}")
    open(srv, "w").write(s.replace("exec /usr/local/bin/ollama serve", "exec ${OLLAMA_BIN:-ollama} serve"))
    for name in ("ptit_test", "ptit_dev"):
        shutil.copy(os.path.join(CORE, "data", f"{name}.jsonl"), os.path.join(_mk(d, "work", "benchmark"), f"{name}.jsonl"))
    shutil.copy(os.path.join(CORE, "corpus", "chunk_index.jsonl"), os.path.join(_mk(d, "release_data"), "chunk_index.jsonl"))
    shutil.copy(os.path.join(CORE, "corpus", "books.json"), os.path.join(d, "release_data", "books.json"))
    shutil.copy(os.path.join(W, "corpus", "books.json"), os.path.join(_mk(d, "work", "corpus"), "books.json"))
    for f in ("GH_README.md:README.md", "LICENSE:LICENSE", "CITATION.cff:CITATION.cff", "requirements.txt:requirements.txt",
              "gitignore:.gitignore"):
        a, b = f.split(":")
        copy_tpl(a, os.path.join(d, b))
    blob = "\n".join(open(p, encoding="utf-8", errors="ignore").read() for p in glob.glob(os.path.join(d, "**", "*"), recursive=True)
                     if os.path.isfile(p) and p.endswith((".py", ".sh", ".md", ".json", ".txt", ".cff")))
    assert ROOT not in blob, "absolute path left in code release"


def build_zenodo():
    d = os.path.join(REL, "zenodo")
    shutil.rmtree(d, ignore_errors=True)
    os.makedirs(d)
    for name, src in ((f"PTIT-CourseQA-v{VERSION}-data.zip", CORE),
                      (f"ptit-courseqa-kgrag-v{VERSION}-code.zip", os.path.join(REL, "github", "ptit-courseqa-kgrag"))):
        with zipfile.ZipFile(os.path.join(d, name), "w", zipfile.ZIP_DEFLATED) as z:
            top = name[:-4]
            for f in sorted(glob.glob(os.path.join(src, "**", "*"), recursive=True) + glob.glob(os.path.join(src, ".*"))):
                if os.path.isfile(f):
                    z.write(f, os.path.join(top, os.path.relpath(f, src)))
            if src == CORE:
                hf = os.path.join(REL, "huggingface", "PTIT-CourseQA")
                z.write(os.path.join(hf, "README.md"), os.path.join(top, "README.md"))
                z.write(os.path.join(hf, "LICENSE"), os.path.join(top, "LICENSE"))
    copy_tpl("zenodo_metadata.json", os.path.join(d, "zenodo_metadata.json"))


def manifest():
    rows = []
    for f in sorted(glob.glob(os.path.join(REL, "**", "*"), recursive=True)):
        if os.path.isfile(f) and not f.endswith("MANIFEST.csv") and "/core/" not in f:
            rows.append([os.path.relpath(f, REL), os.path.getsize(f), hashlib.md5(open(f, "rb").read()).hexdigest()])
    wcsv(os.path.join(REL, "MANIFEST.csv"), ["path", "bytes", "md5"], rows)
    for top in ("huggingface", "github", "zenodo"):
        n = sum(r[1] for r in rows if r[0].startswith(top))
        print(f"{top:12s} {sum(1 for r in rows if r[0].startswith(top)):5d} files {n / 1e6:8.1f} MB")


if __name__ == "__main__":
    build_core()
    build_hf()
    build_github()
    build_zenodo()
    manifest()
