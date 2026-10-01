"""Stage 5d: tune on PTIT DEV only (primary SLM), then freeze the configuration for the single test run.

steps (run in order):
  k        context size k in {3,5,8}: S3 end-to-end on dev -> k for every system (same context budget for all)
  s5grid   S5 retrieval grid (no generation): fusion (graph slots / weighted RRF) x depth x gamma x per_section x ent_min_sim,
           criterion = mean chunk recall@k on multi-hop dev questions (tie-break: all dev answerable)
  s5facts  S5 end-to-end with n_facts in {0, 10}
  s4mode   S4 LightRAG mode in {mix, hybrid} end-to-end
  freeze   write the chosen values into results/configs/systems_config.json and set frozen=true
Outputs: results/dev/tune_<step>.json
"""
import copy
import itertools
import json
import os
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from rag import data, llm, retrieval  # noqa: E402

RES = os.path.join(data.ROOT, "results")
CFGP = os.path.join(RES, "configs", "systems_config.json")
PY = sys.executable
RUN = os.path.join(data.WORK, "scripts", "06_run.py")
EVAL = os.path.join(data.WORK, "scripts", "06_eval.py")


def cfg():
    return json.load(open(CFGP))


def variant(tag, changes):
    c = copy.deepcopy(cfg())
    for k, v in changes.items():
        if isinstance(v, dict):
            c.setdefault(k, {}).update(v)
        else:
            c[k] = v
    p = os.path.join(RES, "dev", f"config{tag}.json")
    json.dump(c, open(p, "w"), indent=1)
    return p


def run(system, tag, cfg_path, slm=llm.PRIMARY):
    base = [PY, RUN, "--system", system, "--dataset", "ptit", "--split", "dev", "--config", cfg_path, "--tag", tag]
    if system != "S0":
        subprocess.run([PY, RUN, "retrieve"] + base[2:], check=True)
    subprocess.run([PY, RUN, "generate"] + base[2:] + ["--slm", slm], check=True)


def metrics(name, slm=llm.PRIMARY):
    subprocess.run([PY, EVAL, "ptit_dev"], check=True, capture_output=True)
    return json.load(open(os.path.join(RES, "metrics", "ptit_dev.json")))[name][slm]["overall"]


def save(step, obj):
    json.dump(obj, open(os.path.join(RES, "dev", f"tune_{step}.json"), "w"), indent=1, ensure_ascii=False)
    print(json.dumps(obj, indent=1, ensure_ascii=False))


def step_k():
    out = {}
    for k in (3, 5, 8):
        tag = f"_k{k}"
        run("S3", tag, variant(tag, {"k": k}))
        out[k] = metrics(f"S3{tag}")
    best = max(out, key=lambda k: (out[k]["f1_all"], out[k]["f1_ans"]))
    save("k", {"criterion": "S3 dev F1_all (tie-break F1_ans), primary SLM", "results": out, "chosen": best})


def step_s5grid():
    from rag.kg import S5Retriever
    from rag.router import Router
    c = cfg()
    k = json.load(open(os.path.join(RES, "dev", "tune_k.json")))["chosen"]
    qs = [q for q in data.questions("ptit", "dev") if q["gold_chunks"]]
    qv = retrieval.encode(c["embedding"], [q["question"] for q in qs], "q")
    r = S5Retriever("ptit", c["embedding"], router=Router.load(), k=k)
    fusions = [("slots", 1, 1.0), ("slots", 2, 1.0), ("slots", 3, 1.0), ("rrf", 0, 0.25), ("rrf", 0, 0.5), ("rrf", 0, 1.0)]
    grid = list(itertools.product(fusions, (1, 2, 3), (0.3, 0.5), (2, 3, 99), (0.5, 0.6)))
    res = {}
    for (fusion, m, gw), depth, gamma, per, sim in grid:
        r.p.update(fusion=fusion, graph_slots=m, graph_weight=gw, depth=depth, gamma=gamma, per_section=per, ent_min_sim=sim)
        rec_multi, rec_all = [], []
        for q, v in zip(qs, qv):
            got = set(r.retrieve(q["question"], v)["chunk_ids"])
            x = len(got & set(q["gold_chunks"])) / len(q["gold_chunks"])
            rec_all.append(x)
            if q["type"].startswith("multi"):
                rec_multi.append(x)
        res[f"{fusion}_m{m}_w{gw}_d{depth}_g{gamma}_p{per}_s{sim}"] = {"recall_multi": round(sum(rec_multi) / len(rec_multi), 4),
                                                                      "recall_all": round(sum(rec_all) / len(rec_all), 4)}
    # tie-break: recall_all, then the simplest graph use (fewer hops, fewer slots)
    best = max(res, key=lambda n: (res[n]["recall_multi"], res[n]["recall_all"], -int(n.split("_d")[1][0]), -int(n.split("_m")[1][0])))
    f, m, w, d, g, p, s = best.split("_")
    chosen = {"fusion": f, "graph_slots": int(m[1:]), "graph_weight": float(w[1:]), "depth": int(d[1:]), "gamma": float(g[1:]),
              "per_section": int(p[1:]), "ent_min_sim": float(s[1:])}
    save("s5grid", {"criterion": "S5 dev mean chunk recall@k on multi-hop questions (tie-break: all answerable)", "k": k,
                    "results": res, "chosen": chosen})


def step_s5facts():
    k = json.load(open(os.path.join(RES, "dev", "tune_k.json")))["chosen"]
    g = json.load(open(os.path.join(RES, "dev", "tune_s5grid.json")))["chosen"]
    out = {}
    for nf in (0, 10):
        tag = f"_f{nf}"
        run("S5", tag, variant(tag, {"k": k, "S5": {**g, "n_facts": nf}}))
        out[nf] = metrics(f"S5{tag}")
    best = max(out, key=lambda n: (out[n]["f1_all"], out[n]["f1_ans"]))
    save("s5facts", {"criterion": "S5 dev F1_all, primary SLM", "results": out, "chosen": best})


def step_s4mode():
    k = json.load(open(os.path.join(RES, "dev", "tune_k.json")))["chosen"]
    out = {}
    for mode in ("mix", "hybrid"):
        tag = f"_{mode}"
        run("S4", tag, variant(tag, {"k": k, "S4": {"mode": mode}}))
        out[mode] = metrics(f"S4{tag}")
    best = max(out, key=lambda m: (out[m]["f1_all"], out[m]["f1_ans"]))
    save("s4mode", {"criterion": "S4 dev F1_all, primary SLM", "results": out, "chosen": best})


def step_freeze():
    c = cfg()
    ld = lambda s: json.load(open(os.path.join(RES, "dev", f"tune_{s}.json")))["chosen"]
    c["k"] = ld("k")
    c["S5"].update(ld("s5grid"))
    c["S5"]["n_facts"] = ld("s5facts")
    c["S4"]["mode"] = ld("s4mode")
    c["frozen"] = True
    c["note"] = "frozen after dev tuning (results/dev/tune_*.json); used unchanged for PTIT test and ViQuAD test"
    json.dump(c, open(CFGP, "w"), indent=1)
    print(json.dumps(c, indent=1))


if __name__ == "__main__":
    os.makedirs(os.path.join(RES, "dev"), exist_ok=True)
    {"k": step_k, "s5grid": step_s5grid, "s5facts": step_s5facts, "s4mode": step_s4mode, "freeze": step_freeze}[sys.argv[1]]()
