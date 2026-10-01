"""Stage 6: automatic metrics from results/predictions and results/retrieval.

Per (dataset_split, system, slm), overall and per question type:
  em_ans / f1_ans     EM / syllable-F1 on answerable questions (a refusal scores 0)
  em_all / f1_all     SQuAD2-style over all questions (unanswerable correct iff the system refuses)
  refusal_unans       correct refusals on unanswerable questions; halluc_unans = 1 - refusal_unans
  false_refusal_ans   refusals on answerable questions
  recall@k            mean share of gold chunks in the retrieved context (answerable questions)
  gen_latency / prompt_tokens / output_tokens means
Writes results/metrics/<dataset_split>.json and per-question scores results/metrics/per_question/<...>.jsonl
(used by the significance tests).
Usage: 06_eval.py <dataset_split> [<dataset_split> ...]
"""
import collections
import glob
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from rag import data  # noqa: E402
from rag.text import exact_match, f1_score, is_refusal  # noqa: E402

RES = os.path.join(data.ROOT, "results")


def score(q, pred):
    ref = is_refusal(pred)
    if q["unanswerable"]:
        return {"em": float(ref), "f1": float(ref), "refusal": float(ref), "answerable": False}
    em = 0.0 if ref else exact_match(pred, q["golds"])
    f1 = 0.0 if ref else f1_score(pred, q["golds"])
    return {"em": em, "f1": f1, "refusal": float(ref), "answerable": True}


def aggregate(rows):
    ans = [r for r in rows if r["answerable"]]
    una = [r for r in rows if not r["answerable"]]
    m = lambda xs: round(float(np.mean(xs)), 4) if xs else None
    out = {"n": len(rows), "n_ans": len(ans), "n_unans": len(una),
           "em_ans": m([r["em"] for r in ans]), "f1_ans": m([r["f1"] for r in ans]),
           "em_all": m([r["em"] for r in rows]), "f1_all": m([r["f1"] for r in rows]),
           "refusal_unans": m([r["refusal"] for r in una]), "halluc_unans": m([1 - r["refusal"] for r in una]),
           "false_refusal_ans": m([r["refusal"] for r in ans])}
    for key in ("recall", "gen_latency_s", "prompt_tokens", "output_tokens", "retrieval_latency_s"):
        vals = [r[key] for r in (ans if key == "recall" else rows) if r.get(key) is not None]
        if vals:
            out[key] = m(vals)
    return out


def main(ds_split):
    dataset, split = ds_split.split("_")
    qs = {q["id"]: q for q in data.questions(dataset, split)}
    report = {}
    os.makedirs(os.path.join(RES, "metrics", "per_question"), exist_ok=True)
    for path in sorted(glob.glob(os.path.join(RES, "predictions", ds_split, "*.jsonl"))):
        name = os.path.basename(path)[:-6]
        system, slm = name.split("__")
        preds = {json.loads(l)["id"]: json.loads(l) for l in open(path, encoding="utf-8")}
        if len(preds) < len(qs):
            print(f"skip incomplete {name}: {len(preds)}/{len(qs)}")
            continue
        ret = {}
        rp = os.path.join(RES, "retrieval", ds_split, f"{system}.jsonl")
        if os.path.exists(rp):
            ret = {json.loads(l)["id"]: json.loads(l) for l in open(rp, encoding="utf-8")}
        rows = []
        for qid, q in qs.items():
            p = preds[qid]
            r = score(q, p["answer"]) | {"id": qid, "type": q["type"], "gen_latency_s": p["gen_latency_s"],
                                         "prompt_tokens": p["prompt_tokens"], "output_tokens": p["output_tokens"]}
            if qid in ret:
                r["retrieval_latency_s"] = ret[qid]["retrieval_latency_s"]
                if q["gold_chunks"]:
                    r["recall"] = len(set(q["gold_chunks"]) & set(ret[qid]["chunk_ids"])) / len(q["gold_chunks"])
                r["route"] = ret[qid].get("route")
            rows.append(r)
        with open(os.path.join(RES, "metrics", "per_question", f"{ds_split}__{name}.jsonl"), "w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        by_type = collections.defaultdict(list)
        for r in rows:
            by_type[r["type"]].append(r)
        report.setdefault(system, {})[slm] = {"overall": aggregate(rows), "by_type": {t: aggregate(v) for t, v in sorted(by_type.items())}}
    json.dump(report, open(os.path.join(RES, "metrics", f"{ds_split}.json"), "w"), indent=1)
    return report


if __name__ == "__main__":
    for a in sys.argv[1:]:
        rep = main(a)
        for s, v in rep.items():
            for slm, x in v.items():
                o = x["overall"]
                print(f"{a} {s:14s} {slm:12s} F1ans={o['f1_ans']} EMans={o['em_ans']} F1all={o['f1_all']} refU={o['refusal_unans']} "
                      f"falseRef={o['false_refusal_ans']} R={o.get('recall')}")
