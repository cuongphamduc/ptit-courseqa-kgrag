"""Stage 6: significance of S5 vs each baseline (S0-S4), per SLM and dataset, on per-question scores.

  F1   paired bootstrap (10,000 resamples, seed 42): mean difference, 95% percentile CI, two-sided p
       (share of resampled differences on the other side of 0, doubled); computed for f1_all and f1_ans
  EM   exact McNemar test (binomial on discordant pairs) over all questions (em_all)
  halluc  exact McNemar on unanswerable questions (hallucinated = answered instead of refusing)
Holm correction over the 5 baselines within each (dataset, SLM, metric) family.
Output: results/stats/<dataset_split>.json
"""
import json
import os
import sys

import numpy as np
from scipy.stats import binomtest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from rag import data  # noqa: E402

RES = os.path.join(data.ROOT, "results")
B, SEED = 10000, 42
BASELINES = ["S0", "S1", "S2", "S3", "S4"]


def load(ds, system, slm):
    p = os.path.join(RES, "metrics", "per_question", f"{ds}__{system}__{slm}.jsonl")
    return {json.loads(l)["id"]: json.loads(l) for l in open(p, encoding="utf-8")} if os.path.exists(p) else None


def bootstrap(a, b, rng):
    d = np.asarray(a) - np.asarray(b)
    idx = rng.integers(0, len(d), size=(B, len(d)))
    means = d[idx].mean(axis=1)
    lo, hi = np.percentile(means, [2.5, 97.5])
    p = min(1.0, 2 * min((means <= 0).mean(), (means >= 0).mean()))
    return {"diff": round(float(d.mean()), 4), "ci95": [round(float(lo), 4), round(float(hi), 4)], "p": float(max(p, 1 / B)), "n": len(d)}


def mcnemar(x, y):
    """x, y: binary outcomes (1 = correct) of S5 and baseline"""
    b = int(sum(1 for i, j in zip(x, y) if i == 1 and j == 0))
    c = int(sum(1 for i, j in zip(x, y) if i == 0 and j == 1))
    p = binomtest(b, b + c, 0.5).pvalue if b + c else 1.0
    return {"s5_only": b, "baseline_only": c, "p": float(p), "n": len(x)}


def holm(ps):
    order = sorted(range(len(ps)), key=lambda i: ps[i])
    adj, run = [0.0] * len(ps), 0.0
    for r, i in enumerate(order):
        run = max(run, min(1.0, (len(ps) - r) * ps[i]))
        adj[i] = run
    return adj


def main(ds, systems=("S5",), baselines=BASELINES, suffix=""):
    from glob import glob
    slms = sorted({os.path.basename(p).split("__")[2][:-6] for p in glob(os.path.join(RES, "metrics", "per_question", f"{ds}__S5__*.jsonl"))})
    out = {}
    for target in systems:
        for slm in slms:
            s5 = load(ds, target, slm)
            fam = {}
            for base in baselines:
                bl = load(ds, base, slm)
                if s5 is None or bl is None:
                    continue
                ids = sorted(s5)
                ans = [i for i in ids if s5[i]["answerable"]]
                una = [i for i in ids if not s5[i]["answerable"]]
                rng = np.random.default_rng(SEED)
                fam[base] = {
                    "f1_all": bootstrap([s5[i]["f1"] for i in ids], [bl[i]["f1"] for i in ids], rng),
                    "f1_ans": bootstrap([s5[i]["f1"] for i in ans], [bl[i]["f1"] for i in ans], rng),
                    "em_all": mcnemar([s5[i]["em"] for i in ids], [bl[i]["em"] for i in ids]),
                    "halluc_unans": mcnemar([s5[i]["refusal"] for i in una], [bl[i]["refusal"] for i in una]),
                }
            for metric in ("f1_all", "f1_ans", "em_all", "halluc_unans"):
                names = list(fam)
                for n, a in zip(names, holm([fam[n][metric]["p"] for n in names])):
                    fam[n][metric]["p_holm"] = round(a, 6)
            out.setdefault(target, {})[slm] = fam
    os.makedirs(os.path.join(RES, "stats"), exist_ok=True)
    json.dump(out, open(os.path.join(RES, "stats", f"{ds}{suffix}.json"), "w"), indent=1)
    return out


ABLATIONS = ["S5-noframe", "S5-norouting", "S5-novector", "S5-depth2", "S5-depth3"]

if __name__ == "__main__":
    for ds in sys.argv[1:]:
        main(ds)
        main(ds, baselines=ABLATIONS, suffix="_ablations")  # S5 vs its ablations (primary SLM only has them)
