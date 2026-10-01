"""Vietnamese text utilities: normalisation, word segmentation, refusal detection, EM/F1."""
import functools
import re
import string
import unicodedata
from collections import Counter

REFUSAL = "Không tìm thấy thông tin trong tài liệu."
# answers that decline instead of using the exact refusal sentence are also counted as refusals
_REFUSAL_PATTERNS = [
    r"không tìm thấy thông tin", r"không có thông tin", r"tài liệu không (đề cập|nêu|cung cấp|chứa|có)",
    r"không (được )?đề cập", r"không đủ thông tin", r"không thể trả lời", r"tôi không biết",
]
_REFUSAL_RE = re.compile("|".join(_REFUSAL_PATTERNS))


def nfc(s):
    return unicodedata.normalize("NFC", s or "")


def normalize_answer(s):
    """SQuAD-style normalisation adapted to Vietnamese: NFC, lower-case, drop punctuation, collapse spaces.
    Diacritics are kept (they distinguish words in Vietnamese)."""
    s = nfc(s).lower()
    s = "".join(" " if (ch in string.punctuation or unicodedata.category(ch).startswith("P")) else ch for ch in s)
    return " ".join(s.split())


def is_refusal(answer):
    return not normalize_answer(answer) or bool(_REFUSAL_RE.search(normalize_answer(answer)))


def exact_match(pred, golds):
    p = normalize_answer(pred)
    return float(any(p == normalize_answer(g) for g in golds))


def f1_score(pred, golds):
    """syllable-level F1, max over gold answers"""
    pt = normalize_answer(pred).split()
    best = 0.0
    for g in golds:
        gt = normalize_answer(g).split()
        if not pt or not gt:
            best = max(best, float(pt == gt))
            continue
        overlap = sum((Counter(pt) & Counter(gt)).values())
        if overlap:
            pr, rc = overlap / len(pt), overlap / len(gt)
            best = max(best, 2 * pr * rc / (pr + rc))
    return best


@functools.lru_cache(maxsize=200000)
def segment(s):
    """Vietnamese word segmentation (pyvi); multi-syllable words joined with '_'."""
    from pyvi import ViTokenizer
    return ViTokenizer.tokenize(nfc(s))


def bm25_tokens(s):
    toks = normalize_answer(segment(s).replace("_", "\u0001")).replace("\u0001", "_").split()
    return toks
