"""Stage 6: faithfulness / groundedness by LLM-as-judge (rag/judge.py, qwen3.6 — not one of the evaluated SLMs).

Scope (compute budget): a fixed judge sample per test set, the same questions for every system and SLM:
  PTIT test   the 150 Stage-7 questions (07a sampling, seed 42) + 150 further questions stratified by type = 300
  ViQuAD test 300 questions stratified by answerable/unanswerable, seed 42
Judged: RAG systems S1-S5 x all SLMs, plus the S5 ablations for the primary SLM. Refusals are not judged
(they carry no claim); they are counted separately.
Output: results/judge/<ds>_test/<system>__<slm>.jsonl, results/judge/summary.json,
        work/benchmark/judge_sample_<ds>.txt
"""
import collections
import glob
import importlib.util
import json
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from rag import data, judge  # noqa: E402
from rag.text import is_refusal  # noqa: E402

RES = os.path.join(data.ROOT, "results")
N = 300


def sample_ids(ds):
    path = os.path.join(data.BENCH, f"judge_sample_{ds}.txt")
    if os.path.exists(path):
        return open(path).read().split()
    if ds == "ptit":
        spec = importlib.util.spec_from_file_location("h", os.path.join(data.WORK, "scripts", "07a_export_halluc.py"))
        h = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(h)
        first = h.sample_questions(random.Random(h.SEED))
        more = h.sample_questions(random.Random(h.SEED + 1), n=N - len(first), exclude={q["id"] for q in first})
        ids = [q["id"] for q in first + more]
    else:
        rng = random.Random(42)
        qs = list(data.questions(ds, "test"))
        ids = []
        for flag in (False, True):
            part = sorted((q["id"] for q in qs if q["unanswerable"] == flag))
            ids += rng.sample(part, round(N * len(part) / len(qs)))
    open(path, "w").write("\n".join(sorted(ids)) + "\n")
    return sorted(ids)


def main(datasets):
    summary = {}
    for ds in datasets:
        ids = set(sample_ids(ds))
        qs = {q["id"]: q for q in data.questions(ds, "test")}
        chunks = {c["chunk_id"]: c for c in data.corpus(data.corpus_for(ds, "test"))}
        for pred_path in sorted(glob.glob(os.path.join(RES, "predictions", f"{ds}_test", "*.jsonl"))):
            name = os.path.basename(pred_path)[:-6]
            system, slm = name.split("__")
            if system == "S0":
                continue
            ret = {json.loads(l)["id"]: json.loads(l) for l in open(os.path.join(RES, "retrieval", f"{ds}_test", f"{system}.jsonl"), encoding="utf-8")}
            preds = {json.loads(l)["id"]: json.loads(l) for l in open(pred_path, encoding="utf-8")}
            out = os.path.join(RES, "judge", f"{ds}_test", f"{name}.jsonl")
            os.makedirs(os.path.dirname(out), exist_ok=True)
            done = {json.loads(l)["id"] for l in open(out, encoding="utf-8")} if os.path.exists(out) else set()
            with open(out, "a", encoding="utf-8") as f:
                for qid in sorted(ids - done):
                    ans = preds[qid]["answer"]
                    if is_refusal(ans):
                        rec = {"id": qid, "refusal": True}
                    else:
                        r = ret[qid]
                        texts = r.get("texts") or [chunks[c]["text"] for c in r["chunk_ids"]]
                        ctx = "\n\n".join(f"[{i + 1}] {h + chr(10) if h else ''}{t}" for i, (h, t) in
                                          enumerate(zip(r.get("headers") or [""] * len(texts), texts)))
                        if r.get("facts"):
                            ctx += "\n\nCÁC QUAN HỆ LIÊN QUAN:\n" + "\n".join(f"- {x}" for x in r["facts"])
                        rec = {"id": qid, "refusal": False, **judge.judge(ctx, qs[qid]["question"], ans)}
                    f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    f.flush()
            rows = [json.loads(l) for l in open(out, encoding="utf-8")]
            ans_rows = [r for r in rows if not r["refusal"] and r.get("score") is not None]
            by_type = collections.defaultdict(list)
            for r in ans_rows:
                by_type[qs[r["id"]]["type"]].append(r["score"])
            summary.setdefault(f"{ds}_test", {}).setdefault(system, {})[slm] = {
                "n_sample": len(rows), "n_judged": len(ans_rows), "refusals": sum(r["refusal"] for r in rows),
                "parse_failures": sum(1 for r in rows if not r["refusal"] and r.get("score") is None),
                "faithfulness": round(sum(r["score"] for r in ans_rows) / max(1, len(ans_rows)), 4),
                "unsupported_rate": round(sum(r["score"] == 0.0 for r in ans_rows) / max(1, len(ans_rows)), 4),
                "by_type": {t: round(sum(v) / len(v), 4) for t, v in sorted(by_type.items())}}
            print(ds, name, summary[f"{ds}_test"][system][slm]["faithfulness"], flush=True)
    json.dump(summary, open(os.path.join(RES, "judge", "summary.json"), "w"), indent=1)


if __name__ == "__main__":
    main(sys.argv[1:] or ["ptit", "viquad"])
