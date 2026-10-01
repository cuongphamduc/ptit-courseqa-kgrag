"""Stage 5a: choose the embedding model for S2/S3/S4/S5 on DEV only.

Criterion (fixed before running): mean chunk recall@5 on PTIT dev answerable questions.
Also reported: recall@{1,3,10,20}, all-gold@5, MRR on PTIT dev and ViQuAD dev; BM25 (S1) as reference;
index build time and peak GPU memory of each encoder.
Output: results/dev/embedding_selection.json, results/configs/embedding_choice.json
"""
import json
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from rag import data, retrieval  # noqa: E402

KS = (1, 3, 5, 10, 20)
OUT = os.path.join(data.ROOT, "results")


def evaluate(ranked, qs):
    res = {f"recall@{k}": [] for k in KS}
    res["all_gold@5"], res["mrr"] = [], []
    for q in qs:
        if not q["gold_chunks"]:
            continue
        ids = [cid for cid, _ in ranked[q["id"]]]
        g = set(q["gold_chunks"])
        for k in KS:
            res[f"recall@{k}"].append(len(g & set(ids[:k])) / len(g))
        res["all_gold@5"].append(float(g <= set(ids[:5])))
        res["mrr"].append(next((1 / (i + 1) for i, c in enumerate(ids) if c in g), 0.0))
    return {k: round(float(np.mean(v)), 4) for k, v in res.items()} | {"n": len(res["mrr"])}


def main():
    sets = {"ptit_dev": ("ptit", "dev"), "viquad_dev": ("viquad", "dev")}
    report = {}
    for tag, (ds, sp) in sets.items():
        qs = data.questions(ds, sp)
        cname = data.corpus_for(ds, sp)
        bm = retrieval.BM25(cname)
        report.setdefault("bm25", {})[tag] = evaluate({q["id"]: bm.search(q["question"], 20) for q in qs}, qs)
    for name in retrieval.EMBEDDERS:
        torch.cuda.reset_peak_memory_stats()
        entry = {}
        for tag, (ds, sp) in sets.items():
            qs = data.questions(ds, sp)
            d = retrieval.Dense(data.corpus_for(ds, sp), name)
            t = time.perf_counter()
            qv = retrieval.encode(name, [q["question"] for q in qs], "q")
            q_s = (time.perf_counter() - t) / len(qs)
            entry[tag] = evaluate({q["id"]: d.search(None, 20, qvec=v) for q, v in zip(qs, qv)}, qs) | {
                "index_build_s": round(d.build_s, 1), "windows": d.n_windows, "query_encode_s_mean_batched": round(q_s, 4)}
        entry["peak_gpu_mem_gib"] = round(torch.cuda.max_memory_allocated() / 2 ** 30, 3)
        report[name] = entry
        retrieval.unload_encoders()
        print(name, json.dumps(entry), flush=True)
    os.makedirs(os.path.join(OUT, "dev"), exist_ok=True)
    json.dump(report, open(os.path.join(OUT, "dev", "embedding_selection.json"), "w"), indent=1)
    best = max(retrieval.EMBEDDERS, key=lambda n: (report[n]["ptit_dev"]["recall@5"], report[n]["ptit_dev"]["mrr"]))
    json.dump({"criterion": "PTIT dev mean chunk recall@5 (tie-break MRR)", "chosen": best,
               "hf": retrieval.EMBEDDERS[best]["hf"], "value": report[best]["ptit_dev"]["recall@5"]},
              open(os.path.join(OUT, "configs", "embedding_choice.json"), "w"), indent=1)
    print("BM25", report["bm25"])
    print("chosen:", best)


if __name__ == "__main__":
    main()
