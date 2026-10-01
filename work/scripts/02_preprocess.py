"""Stage 2: extract, clean, structure and chunk the PTIT textbooks.

Input : dataset/PTIT/**.pdf (read-only), work/corpus/books.json
Output: work/corpus/clean/<book>.paras.jsonl   cleaned paragraphs with page/chapter/section
        work/corpus/chunks/<book>.jsonl        500-600 token chunks with metadata
        work/corpus/all_chunks.jsonl
        work/corpus/stats.json
Run   : work/.venv/bin/python work/scripts/02_preprocess.py
"""
import collections
import json
import os
import re
import subprocess
import unicodedata

from tokenizers import Tokenizer

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
CORPUS = os.path.join(ROOT, "work", "corpus")
BOOKS = json.load(open(os.path.join(CORPUS, "books.json"), encoding="utf-8"))
TOKENIZER_ID = "Qwen/Qwen3.5-4B"  # tokenizer of the primary SLM
TOK = Tokenizer.from_pretrained(TOKENIZER_ID)

MIN_TOK, MAX_TOK, OVERLAP_TOK = 500, 600, 60
# STRICT_HEADINGS=False reproduces the chunking used for the benchmark (v1). 02b_fix_section_labels.py sets it
# to True to re-derive section metadata, rejecting figure captions / citations / exercise prompts as headings.
STRICT_HEADINGS = False
BAD_TITLE = re.compile(
    r"^([a-zàáảãạăâđèéẻẽẹêìíỉĩịòóỏõọôơùúủũụưỳýỷỹỵ\[(\-–]|Hình\b|Bảng\b|Hãy\b|Trình bày\b|Giải thích\b|Nêu\b|"
    r"Cho\b|Tại sao\b|Vì sao\b|So sánh\b|Phân biệt\b|Liệt kê\b|\(Bài tập)")


def bad_title(title):
    t = title.strip()
    return bool(BAD_TITLE.match(t)) or t.endswith("?") or bool(re.match(r"^\[\d+\]", t))
ROMAN = {"I": 1, "II": 2, "III": 3, "IV": 4, "V": 5, "VI": 6, "VII": 7, "VIII": 8, "IX": 9, "X": 10}

RE_CHAPTER = re.compile(r"^(CHƯƠNG|Chương)\s*([0-9]{1,2}|[IVX]{1,4})\b\s*[:.\-–]?\s*(.*)$")
RE_SECTION = re.compile(r"^([0-9]{1,2}(?:\.[0-9]{1,2}){1,3})\.?\s+(\S.*)$")
RE_PAGENUM = re.compile(r"^[-–—\s]*[0-9ivxIVX]{1,4}[-–—\s]*$")
RE_WATERMARK = re.compile(r"^(PT|IT|PTIT)$")
RE_TOC_LEADER = re.compile(r"\.{5,}\s*\d*\s*$")
RE_EXCLUDE_START = re.compile(
    r"^(TÀI LIỆU THAM KHẢO|Tài liệu tham khảo|CÂU HỎI ÔN TẬP|Câu hỏi ôn tập|CÂU HỎI VÀ BÀI TẬP|"
    r"Câu hỏi và bài tập|BÀI TẬP CHƯƠNG|Bài tập chương|DANH MỤC|MỤC LỤC)\b")
RE_APPENDIX = re.compile(r"^(PHỤ LỤC|Phụ lục)\b")
DEF_CUE = re.compile(r"(định nghĩa|khái niệm|được hiểu là|là gì)", re.I)


def ntok(text):
    return len(TOK.encode(text, add_special_tokens=False).ids)


def norm(s):
    s = unicodedata.normalize("NFC", s)
    s = s.replace("ƣ", "ư").replace("Ƣ", "Ư")
    s = re.sub(r"[-]", "•", s)  # private-use bullet glyphs
    s = s.replace("­", "").replace("​", "")
    return s


def extract_pages(pdf):
    raw = subprocess.run(["pdftotext", pdf, "-"], capture_output=True, text=True).stdout
    pages = raw.split("\f")
    if pages and not pages[-1].strip():
        pages = pages[:-1]
    return [norm(p) for p in pages]


def header_footer_keys(pages):
    """Running headers/footers: identical top/bottom-2 lines on >= 3 pages, or digit-masked
    lines on >= 5 pages (e.g. 'Trang 12'). Distinct chapter openers ('CHƯƠNG 1', 'CHƯƠNG 2')
    are protected from the masked rule."""
    exact, masked = collections.Counter(), collections.Counter()
    for p in pages:
        ls = [l.strip() for l in p.splitlines() if l.strip()]
        for l in set(ls[:2] + ls[-2:]):
            exact[l] += 1
            if re.search(r"\d", l) and not RE_CHAPTER.match(l):
                masked[re.sub(r"\d+", "#", l)] += 1
    keys = {k for k, c in exact.items() if c >= 3 and len(k) > 2}
    keys |= {k for k, c in masked.items() if c >= 5 and len(k) > 2}
    return keys


def clean_page_lines(page, hf):
    ls = [l.rstrip() for l in page.splitlines()]
    nonempty = [i for i, l in enumerate(ls) if l.strip()]
    edge = set(nonempty[:2] + nonempty[-2:])
    out = []
    for i, l in enumerate(ls):
        s = l.strip()
        if s and (RE_PAGENUM.match(s) or RE_WATERMARK.match(s)):
            continue
        if i in edge and (s in hf or re.sub(r"\d+", "#", s) in hf):
            if STRICT_HEADINGS and RE_CHAPTER.match(s) and s not in _SEEN_CHAPTER_LINES:
                _SEEN_CHAPTER_LINES.add(s)
                out.append(s)
            continue
        out.append(s)
    return out


_SEEN_CHAPTER_LINES = set()


def is_next_number(nums, last):
    """Strict mode: accept a section number only if it continues the numbering (allowing a skip of <= 2)."""
    parent = nums[:-1]
    if tuple(last[:len(parent)]) != tuple(parent):
        # allow skipping into a deeper level from an ancestor, e.g. chapter 4 -> 4.1.1
        k = len(last)
        return tuple(parent[:k]) == tuple(last) and all(x == 1 for x in parent[k:]) and nums[-1] == 1
    prev = last[len(parent)] if len(last) > len(parent) else 0
    return prev < nums[-1] <= prev + 3


RE_NUM_ONLY = re.compile(r"^[0-9]{1,2}(?:\.[0-9]{1,2}){1,3}\.?$")


def is_upper_line(s):
    letters = [c for c in s if c.isalpha()]
    return len(letters) >= 3 and all(c.isupper() for c in letters)


def join_heading_lines(lines):
    """Join '5.1' + 'Title' and multi-line upper-case chapter titles into single lines."""
    out, i = [], 0
    while i < len(lines):
        l = lines[i]
        j = i + 1
        while j < len(lines) and not lines[j]:
            j += 1
        nxt = lines[j] if j < len(lines) else ""
        if RE_NUM_ONLY.match(l) and nxt and len(nxt) < 110 and (nxt[0].isupper() or nxt[0] == "Đ") and not RE_NUM_ONLY.match(nxt):
            out.append(l.rstrip(".") + " " + nxt)
            i = j + 1
            continue
        mc = RE_CHAPTER.match(l)
        if mc and mc.group(3) and is_upper_line(mc.group(3)) and nxt and is_upper_line(nxt) and len(nxt) < 80 \
                and not RE_CHAPTER.match(nxt) and not RE_SECTION.match(nxt):
            out.append(l + " " + nxt)
            i = j + 1
            continue
        out.append(l)
        i += 1
    return out


def chapter_num(tok):
    return int(tok) if tok.isdigit() else ROMAN.get(tok, -1)


def is_toc_page(lines):
    return sum(1 for l in lines if RE_TOC_LEADER.search(l)) >= 4


def build_paragraphs(book, pages):
    """Return list of paragraph dicts in reading order with structure labels."""
    hf = header_footer_keys(pages)
    _SEEN_CHAPTER_LINES.clear()
    page_lines = [join_heading_lines(clean_page_lines(p, hf)) for p in pages]

    # Pre-scan numbered section headings for chapter look-ahead.
    sec_first = []  # (page_idx, chapter_number_of_section)
    for pi, lines in enumerate(page_lines):
        if is_toc_page(lines):
            continue
        for l in lines:
            m = RE_SECTION.match(l)
            if m and len(l) < 130:
                sec_first.append((pi, int(m.group(1).split(".")[0])))

    def chapter_confirmed(pi, n):
        return any(p >= pi and p <= pi + 4 and c == n for p, c in sec_first)

    paras, cur = [], None
    chap_no, chap_title, sec_no, sec_title = 0, "", "", ""
    last_sec = (0,)
    mode = "front"  # front | body | excluded | appendix
    pending_title = False
    for pi, lines in enumerate(page_lines):
        page = pi + 1
        if is_toc_page(lines):
            continue
        buf = []

        def flush():
            nonlocal buf
            if buf:
                text = " ".join(buf)
                text = re.sub(r"(\w)- (\w)", lambda m: m.group(1) + m.group(2) if m.group(2).islower() else m.group(0), text)
                text = re.sub(r"\s+", " ", text).strip()
                if text and mode in ("body", "appendix"):
                    cno = "PL" if (STRICT_HEADINGS and mode == "appendix") else chap_no
                    paras.append({"page": page, "text": text, "chapter_no": cno, "chapter": chap_title,
                                  "section_no": sec_no, "section": sec_title, "heading": False,
                                  "mode": mode})
                buf = []

        for l in lines:
            if not l:
                flush()
                continue
            if pending_title:
                chap_title = f"Chương {chap_no}. {l}"
                pending_title = False
                if paras and paras[-1]["heading"] and paras[-1]["chapter_no"] == chap_no:
                    paras[-1]["text"] += " " + l
                    paras[-1]["chapter"] = chap_title
                continue
            mc = RE_CHAPTER.match(l)
            if mc and len(l) < 130 and not mc.group(3)[:1].islower() and not l.endswith(","):
                n = chapter_num(mc.group(2))
                if n == chap_no + 1 and chapter_confirmed(pi, n):
                    flush()
                    chap_no, sec_no, sec_title, last_sec = n, "", "", (n,)
                    title = mc.group(3).strip()
                    chap_title = f"Chương {n}. {title}" if title else f"Chương {n}"
                    pending_title = not title
                    mode = "body"
                    paras.append({"page": page, "text": l, "chapter_no": chap_no, "chapter": chap_title,
                                  "section_no": "", "section": "", "heading": True, "mode": mode})
                    continue
            ms = RE_SECTION.match(l)
            if ms and len(l) < 130 and mode != "front":
                nums = tuple(int(x) for x in ms.group(1).split("."))
                if nums[0] == chap_no and nums > last_sec and not (STRICT_HEADINGS and (bad_title(ms.group(2)) or not is_next_number(nums, last_sec))):
                    flush()
                    last_sec = nums
                    sec_no, sec_title = ms.group(1), ms.group(2).strip()
                    if mode == "excluded":
                        mode = "body"
                    paras.append({"page": page, "text": l, "chapter_no": chap_no, "chapter": chap_title,
                                  "section_no": sec_no, "section": sec_title, "heading": True, "mode": mode})
                    continue
            if mode != "front" and len(l) < 60 and RE_EXCLUDE_START.match(l):
                flush()
                mode = "excluded"
                continue
            if mode != "front" and len(l) < 40 and RE_APPENDIX.match(l):
                flush()
                mode = "appendix"
                chap_no_app = "PL"
                chap_title, sec_no, sec_title = "Phụ lục", "", ""
                paras.append({"page": page, "text": l, "chapter_no": chap_no_app, "chapter": chap_title,
                              "section_no": "", "section": "", "heading": True, "mode": mode})
                continue
            buf.append(l)
        flush()
    return merge_cross_page(paras)


def merge_cross_page(paras):
    """Merge a paragraph split by a page break (no final punctuation + lowercase start)."""
    out = []
    for p in paras:
        if (out and not p["heading"] and not out[-1]["heading"] and p["page"] == out[-1].get("page_end", out[-1]["page"]) + 1
                and not re.search(r"[.:;!?)\]]$", out[-1]["text"]) and p["text"][:1].islower()
                and p["section_no"] == out[-1]["section_no"]):
            out[-1]["text"] += " " + p["text"]
            out[-1]["page_end"] = p["page"]
        else:
            p.setdefault("page_end", p["page"])
            out.append(p)
    return out


SENT_SPLIT = re.compile(r"(?<=[.!?;])\s+(?=[A-ZĐÀ-Ỹ0-9•\-–(\"“])")


def split_long(p, limit):
    """Split an over-long paragraph into sentence groups <= limit tokens."""
    sents = SENT_SPLIT.split(p["text"])
    pieces, cur = [], ""
    for s in sents:
        cand = (cur + " " + s).strip()
        if cur and ntok(cand) > limit:
            pieces.append(cur)
            cur = s
        else:
            cur = cand
    if cur:
        pieces.append(cur)
    final = []
    for piece in pieces:  # hard split by words if a single sentence is too long
        while ntok(piece) > limit:
            words = piece.split()
            cut = max(1, int(len(words) * limit / ntok(piece) * 0.95))
            final.append(" ".join(words[:cut]))
            piece = " ".join(words[cut:])
        final.append(piece)
    return [dict(p, text=t) for t in final]


def tail_overlap(units):
    """Trailing sentences of the previous chunk (<= OVERLAP_TOK tokens) used as light overlap."""
    if not units or units[-1]["heading"]:
        return None
    sents = SENT_SPLIT.split(units[-1]["text"])
    tail = ""
    for s in reversed(sents):
        cand = (s + " " + tail).strip()
        if ntok(cand) > OVERLAP_TOK:
            break
        tail = cand
    return dict(units[-1], text=tail, overlap=True) if tail else None


def chunk_book(book, paras):
    units = []
    for p in paras:
        p["ntok"] = ntok(p["text"])
        if p["ntok"] > MAX_TOK - OVERLAP_TOK:
            for q in split_long(p, MAX_TOK - OVERLAP_TOK):
                q["ntok"] = ntok(q["text"])
                units.append(q)
        else:
            units.append(p)

    chunks, cur = [], []

    def cur_tok():
        return sum(u["ntok"] for u in cur) + max(0, len(cur) - 1)

    def emit():
        nonlocal cur
        body = [u for u in cur if not u.get("overlap")]
        if not body:
            cur = []
            return
        first = next((u for u in body if not u["heading"]), body[0])
        text = "\n".join(u["text"] for u in cur)
        sec = first["section_no"] + " " + first["section"] if first["section_no"] else ""
        chunks.append({
            "chunk_id": f"{book}-{len(chunks) + 1:04d}",
            "book_id": book,
            "chapter": first["chapter"],
            "chapter_no": first["chapter_no"],
            "section": sec.strip(),
            "sections_covered": sorted({u["section_no"] for u in body if u["section_no"]}),
            "page_start": min(u["page"] for u in cur),
            "page_end": max(u.get("page_end", u["page"]) for u in cur),
            "n_tokens": ntok(text),
            "text": text,
        })
        ov = tail_overlap(body)
        cur = [ov] if ov else []
        if ov:
            ov["ntok"] = ntok(ov["text"])

    def fill_split(u, room):
        """Split a non-definition paragraph at sentence boundaries so the head fits into `room` tokens."""
        if u["heading"] or DEF_CUE.search(u["text"][:80]) or room < 40:
            return None, u
        sents = SENT_SPLIT.split(u["text"])
        head = ""
        for k, s in enumerate(sents):
            cand = (head + " " + s).strip()
            if ntok(cand) > room:
                break
            head = cand
        else:
            return None, u
        if not head:
            return None, u
        rest = u["text"][len(head):].strip()
        h = dict(u, text=head, ntok=ntok(head))
        r = dict(u, text=rest, ntok=ntok(rest))
        return h, r

    queue = list(units)
    while queue:
        u = queue.pop(0)
        # never cross chapters
        if cur and u["chapter_no"] != cur[-1]["chapter_no"]:
            emit()
            cur = [x for x in cur if not x.get("overlap")]
        # prefer to break at a heading once the chunk is reasonably full
        if u["heading"] and cur and cur_tok() >= MIN_TOK:
            emit()
        if cur and cur_tok() + u["ntok"] + 1 > MAX_TOK and cur_tok() < MIN_TOK \
                and u["chapter_no"] == cur[-1]["chapter_no"]:
            h, r = fill_split(u, MAX_TOK - cur_tok() - 1)
            if h:
                cur.append(h)
                emit()
                queue.insert(0, r)
                continue
        if cur and cur_tok() + u["ntok"] + 1 > MAX_TOK:
            # keep a definition / list-introducing paragraph together with what follows
            last = cur[-1]
            if (not last["heading"] and last["text"].endswith(":") and len(cur) > 1
                    and DEF_CUE.search(last["text"]) and cur_tok() - last["ntok"] >= MIN_TOK * 0.5):
                cur.pop()
                emit()
                if cur_tok() + last["ntok"] + u["ntok"] + 2 > MAX_TOK:
                    cur = []  # drop overlap to make room for the definition + its continuation
                cur.append(last)
                if cur_tok() + u["ntok"] + 1 > MAX_TOK:
                    emit()
            else:
                emit()
        cur.append(u)
    if cur and any(not x.get("overlap") for x in cur):
        emit()
    # merge a tiny trailing chunk of a chapter into its predecessor when it fits
    merged = []
    for c in chunks:
        if (merged and merged[-1]["chapter_no"] == c["chapter_no"]
                and ((c["n_tokens"] < 150 and merged[-1]["n_tokens"] + c["n_tokens"] <= MAX_TOK)
                     or c["n_tokens"] < 50)):
            m = merged[-1]
            m["text"] += "\n" + c["text"]
            m["page_end"] = max(m["page_end"], c["page_end"])
            m["sections_covered"] = sorted(set(m["sections_covered"]) | set(c["sections_covered"]))
            m["n_tokens"] = ntok(m["text"])
        else:
            merged.append(c)
    # fragments < 50 tokens at a chapter start: merge forward into the next chunk of the same chapter
    final = []
    for c in merged:
        if final and final[-1]["n_tokens"] < 50 and final[-1]["chapter_no"] == c["chapter_no"]:
            f = final.pop()
            c["text"] = f["text"] + "\n" + c["text"]
            c["page_start"] = min(c["page_start"], f["page_start"])
            c["sections_covered"] = sorted(set(c["sections_covered"]) | set(f["sections_covered"]))
            c["n_tokens"] = ntok(c["text"])
        final.append(c)
    merged = [c for c in final if c["n_tokens"] >= 50]
    for i, c in enumerate(merged):
        c["chunk_id"] = f"{book}-{i + 1:04d}"
    return merged


def main():
    os.makedirs(os.path.join(CORPUS, "clean"), exist_ok=True)
    os.makedirs(os.path.join(CORPUS, "chunks"), exist_ok=True)
    stats, all_chunks = {}, []
    for book, rel in BOOKS.items():
        pdf = os.path.join(ROOT, "dataset", "PTIT", rel)
        pages = extract_pages(pdf)
        paras = build_paragraphs(book, pages)
        with open(os.path.join(CORPUS, "clean", f"{book}.paras.jsonl"), "w", encoding="utf-8") as f:
            for p in paras:
                f.write(json.dumps(p, ensure_ascii=False) + "\n")
        chunks = chunk_book(book, paras)
        with open(os.path.join(CORPUS, "chunks", f"{book}.jsonl"), "w", encoding="utf-8") as f:
            for c in chunks:
                f.write(json.dumps(c, ensure_ascii=False) + "\n")
        all_chunks += chunks
        used_pages = {p for x in paras for p in range(x["page"], x["page_end"] + 1)}
        toks = [c["n_tokens"] for c in chunks]
        chapters = collections.OrderedDict()
        for x in paras:
            chapters.setdefault(str(x["chapter_no"]), x["chapter"])
        stats[book] = {
            "file": rel,
            "pdf_pages": len(pages),
            "content_pages": len(used_pages),
            "ocr_pages": 0,
            "paragraphs": len(paras),
            "headings": sum(1 for x in paras if x["heading"]),
            "chapters": chapters,
            "tokens_content": sum(ntok(x["text"]) for x in paras),
            "chunks": len(chunks),
            "chunk_tokens_min": min(toks),
            "chunk_tokens_median": sorted(toks)[len(toks) // 2],
            "chunk_tokens_max": max(toks),
            "chunks_in_500_600": sum(1 for t in toks if 500 <= t <= 600),
        }
        print(book, {k: v for k, v in stats[book].items() if k not in ("chapters", "file")})
    with open(os.path.join(CORPUS, "all_chunks.jsonl"), "w", encoding="utf-8") as f:
        for c in all_chunks:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")
    json.dump({"tokenizer": TOKENIZER_ID, "min_tok": MIN_TOK, "max_tok": MAX_TOK, "overlap_tok": OVERLAP_TOK,
               "books": stats}, open(os.path.join(CORPUS, "stats.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main()
