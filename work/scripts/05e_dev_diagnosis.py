"""Dev-only retrieval diagnosis behind the S5 fusion redesign (no generation):
text-branch recall@8 / recall@50, graph-branch recall@8, and the share of gold chunks missed by the text top-8 that
the graph ranking recovers in its top-20, on PTIT dev answerable questions (graph settings of the v1 design:
depth 1, gamma 0.3, entity threshold 0.5). Output: results/dev/s5_diagnosis.json
"""
import collections
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from rag import data, retrieval  # noqa: E402
from rag.kg import S5Retriever  # noqa: E402
from rag.router import Router  # noqa: E402


def main():
    cfg = json.load(open(os.path.join(data.ROOT, "results", "configs", "systems_config.json")))
    qs = [q for q in data.questions("ptit", "dev") if q["gold_chunks"]]
    qv = retrieval.encode(cfg["embedding"], [q["question"] for q in qs], "q")
    r = S5Retriever("ptit", cfg["embedding"], router=Router.load(), k=8)
    st = collections.defaultdict(list)
    for q, v in zip(qs, qv):
        g = set(q["gold_chunks"])
        t = [c for c, _ in r.text_rank(q["question"], v)]
        w = r.kg.expand(r.kg.link(q["question"], v, 10, 0.5), 1, 0.3)
        gs = collections.Counter()
        for e, we in w.items():
            for c in r.kg.ents[e]["chunks"]:
                gs[c] += we
        gr = [c for c, _ in gs.most_common(50)]
        grp = "multi_hop" if q["type"].startswith("multi") else "single"
        for name, val in (("text_recall@8", len(g & set(t[:8])) / len(g)), ("text_recall@50", len(g & set(t[:50])) / len(g)),
                          ("graph_recall@8", len(g & set(gr[:8])) / len(g))):
            st[(grp, name)].append(val)
        miss = g - set(t[:8])
        if miss:
            st[(grp, "missed_by_text8_found_in_graph20")].append(len(miss & set(gr[:20])) / len(miss))
    out = {f"{grp}:{name}": {"mean": round(float(np.mean(v)), 4), "n": len(v)} for (grp, name), v in sorted(st.items())}
    json.dump(out, open(os.path.join(data.ROOT, "results", "dev", "s5_diagnosis.json"), "w"), indent=1)
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
