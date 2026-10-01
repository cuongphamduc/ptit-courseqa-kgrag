"""Stage 3a: plan question generation.

- Assign chunks to dev/test evidence pools (seed 42, stratified by chapter) so dev and test
  questions never share evidence chunks.
- Compute per-agent quotas (30% over-generation) for the 4 question types.
- Export agent-readable book files with chunk headers.

Output: work/qgen/plan.json, work/qgen/books/<book>.txt, results/configs/qgen_config.json
"""
import collections
import json
import math
import os
import random

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
CORPUS = os.path.join(ROOT, "work", "corpus")
QGEN = os.path.join(ROOT, "work", "qgen")
SEED = 42
DEV_FRACTION = 0.18
MIN_DEV_CHUNKS = 12
OVERGEN = 1.30
TARGET = {"test": 1000, "dev": 200}
TYPE_RATIO = {"single": 0.35, "multi_intra": 0.30, "multi_cross": 0.20, "unanswerable": 0.15}

# Cross-book pairs (from the GATE-1 inventory, section 3) with topical hints.
PAIRS = [
    ("DT-NHUNG", "DT-VXL", "kiến trúc/thanh ghi/chế độ của vi xử lý ARM, ngắt, vi điều khiển 8051, bộ nhớ, lập trình hợp ngữ trong hệ nhúng"),
    ("DT-NHUNG", "DT-VLSI", "bộ xử lý nhúng, FPGA/ASIC, công nghệ CMOS, thiết kế mạch tích hợp, tiêu thụ công suất"),
    ("ATTT-CS", "ATTT-TMDT", "mã hóa, chữ ký số, chứng thực, an toàn thông tin mạng, khung pháp lý về an toàn thông tin"),
    ("CNTT-PTTK", "ATTT-TMDT", "hệ thống thương mại điện tử, giao dịch điện tử, hợp đồng điện tử, thanh toán trực tuyến, bảo vệ thông tin khách hàng"),
    ("MKT-CRM", "MKT-QTBH", "chăm sóc khách hàng, dịch vụ khách hàng, lực lượng bán hàng, cơ sở dữ liệu khách hàng, khách hàng trung thành"),
    ("MKT-PT", "MKT-CRM", "phân tích khách hàng, phân khúc, giá trị vòng đời khách hàng, dữ liệu khách hàng, marketing trực tuyến"),
]
PAIR_TEST, PAIR_DEV = 20, 4  # generated per pair (split evenly between the two primary books)


def main():
    rng = random.Random(SEED)
    books = json.load(open(os.path.join(CORPUS, "books.json"), encoding="utf-8"))
    chunks = [json.loads(l) for l in open(os.path.join(CORPUS, "all_chunks.jsonl"), encoding="utf-8")]
    by_book = collections.defaultdict(list)
    for c in chunks:
        by_book[c["book_id"]].append(c)

    # --- dev/test pools, stratified by chapter; dev chunks drawn as short contiguous runs (2-3 chunks)
    pools = {}
    for b, cs in by_book.items():
        n_dev = max(MIN_DEV_CHUNKS, round(len(cs) * DEV_FRACTION))
        by_chap = collections.defaultdict(list)
        for c in cs:
            by_chap[str(c["chapter_no"])].append(c["chunk_id"])
        dev = set()
        chap_order = list(by_chap)
        # allocate proportional quota per chapter
        quota = {ch: max(1, round(n_dev * len(ids) / len(cs))) for ch, ids in by_chap.items()}
        for ch in chap_order:
            ids = by_chap[ch]
            need = min(quota[ch], len(ids) - 1) if len(ids) > 1 else 0
            while need > 0:
                run = min(need, rng.choice([2, 3]))
                start = rng.randrange(0, max(1, len(ids) - run + 1))
                for k in range(start, min(len(ids), start + run)):
                    if ids[k] not in dev and need > 0:
                        dev.add(ids[k])
                        need -= 1
                if len(dev) >= len(cs) - 1:
                    break
        pools[b] = {c["chunk_id"]: ("dev" if c["chunk_id"] in dev else "test") for c in cs}

    # --- quotas
    n_books = len(books)
    per_book_test = math.ceil(TARGET["test"] * OVERGEN / n_books)  # 119 -> keep 118 as below
    per_book_dev = math.ceil(TARGET["dev"] * OVERGEN / n_books)
    def type_quota(total):
        q = {t: round(total * r) for t, r in TYPE_RATIO.items()}
        q["single"] += total - sum(q.values())
        return q
    tq_test, tq_dev = type_quota(per_book_test), type_quota(per_book_dev)

    pair_count = collections.Counter()
    for a, b, _ in PAIRS:
        pair_count[a] += 1
        pair_count[b] += 1

    agents = []
    for b in books:
        ct = tq_test["multi_cross"] - (PAIR_TEST // 2) * pair_count[b]
        cd = tq_dev["multi_cross"] - (PAIR_DEV // 2) * pair_count[b]
        agents.append({
            "agent": f"book:{b}", "kind": "book", "books": [b],
            "quota": {
                "test": {"single": tq_test["single"], "multi_intra": tq_test["multi_intra"],
                         "multi_cross_chapter": max(ct, 0), "unanswerable": tq_test["unanswerable"]},
                "dev": {"single": tq_dev["single"], "multi_intra": tq_dev["multi_intra"],
                        "multi_cross_chapter": max(cd, 0), "unanswerable": tq_dev["unanswerable"]},
            },
            "output": f"work/qgen/raw/{b}.jsonl",
        })
    for a, b, hint in PAIRS:
        agents.append({
            "agent": f"pair:{a}+{b}", "kind": "pair", "books": [a, b], "hint": hint,
            "quota": {"test": {"multi_cross_book": PAIR_TEST, "primary_split": {a: PAIR_TEST // 2, b: PAIR_TEST // 2}},
                      "dev": {"multi_cross_book": PAIR_DEV, "primary_split": {a: PAIR_DEV // 2, b: PAIR_DEV // 2}}},
            "output": f"work/qgen/raw/pair_{a}+{b}.jsonl",
        })

    os.makedirs(os.path.join(QGEN, "books"), exist_ok=True)
    os.makedirs(os.path.join(QGEN, "raw"), exist_ok=True)
    for b, cs in by_book.items():
        with open(os.path.join(QGEN, "books", f"{b}.txt"), "w", encoding="utf-8") as f:
            f.write(f"# {b}: {books[b]}\n\n")
            for c in cs:
                f.write(f"### CHUNK {c['chunk_id']} | split={pools[b][c['chunk_id']]} | {c['chapter']} | "
                        f"{c['section']} | trang PDF {c['page_start']}-{c['page_end']}\n{c['text']}\n\n")

    plan = {"seed": SEED, "dev_fraction": DEV_FRACTION, "overgen": OVERGEN, "target": TARGET,
            "type_ratio": TYPE_RATIO, "pools": pools, "agents": agents,
            "pool_sizes": {b: collections.Counter(p.values()) for b, p in pools.items()}}
    json.dump(plan, open(os.path.join(QGEN, "plan.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    os.makedirs(os.path.join(ROOT, "results", "configs"), exist_ok=True)
    cfg = {k: v for k, v in plan.items() if k != "pools"}
    cfg["generator"] = "claude-opus-5-5 (Claude Code subagent)"
    cfg["pairs"] = PAIRS
    json.dump(cfg, open(os.path.join(ROOT, "results", "configs", "qgen_config.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)

    tot = collections.Counter()
    for a in agents:
        for split, q in a["quota"].items():
            for t, n in q.items():
                if t != "primary_split":
                    tot[(split, t)] += n
    print("pool sizes:", {b: dict(v) for b, v in plan["pool_sizes"].items()})
    print("per-book test quota:", tq_test, " dev:", tq_dev)
    for k in sorted(tot):
        print(k, tot[k])
    for a in agents:
        print(a["agent"], a["quota"])


if __name__ == "__main__":
    main()
