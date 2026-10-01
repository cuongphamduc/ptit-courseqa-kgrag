"""Stage 6 (RQ3): cost on limited hardware, from files written during the runs.

offline  index/graph build time: BM25 and dense (timed here / cached build_s), S5 SLM extraction (sum of per-chunk
         latencies in kg_extract.jsonl), LightRAG wall time (lightrag_build_time.json);
         graph size: S5 entities / relations / mentions / frame nodes; LightRAG nodes / edges (graphml)
online   per system and SLM (test runs, sequential, alone on the GPU): retrieval latency, generation latency and
         their per-question sum (mean, p95), prompt tokens
memory   Ollama footprint per SLM (weights + KV at num_ctx 8192), retrieval-process peak GPU memory and RSS
Output: results/costs.json
"""
import glob
import json
import os
import sys
import time
import xml.etree.ElementTree as ET

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from rag import data, retrieval  # noqa: E402
from rag.kg import KG  # noqa: E402

RES = os.path.join(data.ROOT, "results")
IDX = os.path.join(data.WORK, "index")


def graphml_size(path):
    n = e = 0
    for _, el in ET.iterparse(path):
        tag = el.tag.split("}")[-1]
        n += tag == "node"
        e += tag == "edge"
        el.clear()
    return n, e


def main():
    emb = json.load(open(os.path.join(RES, "configs", "embedding_choice.json")))["chosen"]
    out = {"offline": {}, "online": {}, "memory": {}}
    for corpus in ("ptit", "viquad_test"):
        t = time.perf_counter()
        retrieval.BM25(corpus)
        bm25_s = time.perf_counter() - t
        dense_s = float(np.load(os.path.join(IDX, corpus, f"{emb}.npz"))["build_s"])
        kg = KG(corpus, emb)
        st = kg.stats()
        lr_time = [json.loads(l) for l in open(os.path.join(IDX, corpus, "lightrag_build_time.json"))]
        orig = os.path.join(IDX, corpus, "lightrag_build_time.contaminated.json")
        if os.path.exists(orig):  # ViQuAD: the extraction cost is the first build (rebuild reused its LLM cache)
            lr_time = [json.loads(l) for l in open(orig)] + lr_time
        n, e = graphml_size(os.path.join(IDX, corpus, "lightrag", "graph_chunk_entity_relation.graphml"))
        out["offline"][corpus] = {
            "chunks": len(data.corpus(corpus)), "bm25_build_s": round(bm25_s, 1), "dense_build_s": round(dense_s, 1),
            "s5_extraction_s": round(st["extraction"]["latency_s"], 1), "s5_extraction_output_tokens": st["extraction"]["output_tokens"],
            "s5_json_ok": f"{st['extraction']['json_ok']}/{st['extraction']['chunks']}",
            "s5_graph": {k: st[k] for k in ("entities", "relations", "mentions", "chunks_with_entities", "frame_nodes")},
            "lightrag_build_runs": lr_time, "lightrag_graph": {"nodes": n, "edges": e}}
    for ds in ("ptit_test", "viquad_test"):
        ret = {}
        for p in glob.glob(os.path.join(RES, "retrieval", ds, "*.jsonl")):
            ret[os.path.basename(p)[:-6]] = {json.loads(l)["id"]: json.loads(l)["retrieval_latency_s"] for l in open(p)}
        for p in sorted(glob.glob(os.path.join(RES, "predictions", ds, "*.jsonl"))):
            system, slm = os.path.basename(p)[:-6].split("__")
            rows = [json.loads(l) for l in open(p)]
            g = np.array([r["gen_latency_s"] for r in rows])
            rl = np.array([ret.get(system, {}).get(r["id"], 0.0) for r in rows])
            tot = g + rl
            out["online"].setdefault(ds, {}).setdefault(system, {})[slm] = {
                "retrieval_mean_s": round(float(rl.mean()), 4), "retrieval_p95_s": round(float(np.percentile(rl, 95)), 4),
                "generation_mean_s": round(float(g.mean()), 3), "generation_p95_s": round(float(np.percentile(g, 95)), 3),
                "total_mean_s": round(float(tot.mean()), 3), "total_p95_s": round(float(np.percentile(tot, 95)), 3),
                "prompt_tokens_mean": round(float(np.mean([r["prompt_tokens"] for r in rows if r["prompt_tokens"] is not None])), 1),
                "prompt_tokens_missing": sum(r["prompt_tokens"] is None for r in rows),
                "prompt_tokens_max": int(max(r["prompt_tokens"] or 0 for r in rows))}
    runs = [json.loads(l) for l in open(os.path.join(RES, "resources", "runs.jsonl"))]
    for r in runs:
        if r["kind"] == "generate" and r["dataset"].endswith("_test"):
            for tag, size in r["ollama_size_gib"].items():
                out["memory"].setdefault("slm_ollama_gib", {})[r["slm"]] = max(size, out["memory"].get("slm_ollama_gib", {}).get(r["slm"], 0))
        if r["kind"] == "retrieve" and r["dataset"].endswith("_test"):
            m = out["memory"].setdefault("retrieval", {}).setdefault(r["system"], {"gpu_peak_gib": 0, "max_rss_gib": 0})
            m["gpu_peak_gib"] = max(m["gpu_peak_gib"], r["gpu_peak_gib"])
            m["max_rss_gib"] = max(m["max_rss_gib"], r["max_rss_gib"])
    json.dump(out, open(os.path.join(RES, "costs.json"), "w"), indent=1)
    print(json.dumps(out["offline"], indent=1))
    print(json.dumps(out["memory"], indent=1))
    for ds in out["online"]:
        for s, v in out["online"][ds].items():
            x = v.get("qwen3.5-4b")
            if x:
                print(ds, s, x)


if __name__ == "__main__":
    main()
