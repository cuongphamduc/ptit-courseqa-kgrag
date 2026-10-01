"""Stage 5/6 runner.

  retrieve  --system S1..S5|ablation --dataset ptit|viquad --split dev|test
      -> results/retrieval/<dataset>_<split>/<system>.jsonl   (chunk ids, facts, route, retrieval latency)
  generate  --system S0..S5|ablation --slm <name> --dataset ... --split ...
      -> results/predictions/<dataset>_<split>/<system>__<slm>.jsonl   (answer, tokens, generation latency)

Test-set rule: a test run needs results/configs/systems_config.json with "frozen": true (chosen on dev) and refuses
to overwrite a complete test file. Every run is resumable (already answered ids are skipped).
Config for dev runs: --config <json> (defaults to the systems_config.json draft).
"""
import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from rag import data, llm, prompts, retrieval, systems  # noqa: E402

RES = os.path.join(data.ROOT, "results")
CFG = os.path.join(RES, "configs", "systems_config.json")


def load_cfg(path):
    cfg = json.load(open(path or CFG))
    return cfg


def done_ids(path):
    if not os.path.exists(path):
        return set()
    return {json.loads(l)["id"] for l in open(path, encoding="utf-8") if l.strip()}


def guard_test(args, cfg, out, n_total):
    if args.split != "test":
        return
    if not cfg.get("frozen"):
        sys.exit("test run refused: systems_config.json is not frozen (tune on dev first)")
    if len(done_ids(out)) >= n_total:
        sys.exit(f"test run refused: {out} is already complete (test runs once)")


def log_resources(kind, args, extra):
    """append a resource record: peak memory of the SLM (Ollama) or of the retrieval process"""
    import resource
    rec = {"kind": kind, "system": args.system + args.tag, "dataset": f"{args.dataset}_{args.split}",
           "max_rss_gib": round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2 ** 20, 3), **extra}
    os.makedirs(os.path.join(RES, "resources"), exist_ok=True)
    with open(os.path.join(RES, "resources", "runs.jsonl"), "a") as f:
        f.write(json.dumps(rec) + "\n")


def cmd_retrieve(args):
    cfg = load_cfg(args.config)
    qs = data.questions(args.dataset, args.split)
    out = os.path.join(RES, "retrieval", f"{args.dataset}_{args.split}", f"{args.system}{args.tag}.jsonl")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    guard_test(args, cfg, out, len(qs))
    done = done_ids(out)
    todo = [q for q in qs if q["id"] not in done]
    if not todo:
        return
    r = systems.make(args.system, data.corpus_for(args.dataset, args.split), cfg)
    emb = cfg["embedding"]
    retrieval.encode(emb, ["khởi động"], "q")  # warm-up (model load excluded from per-question latency)
    with open(out, "a", encoding="utf-8") as f:
        for i, q in enumerate(todo):
            t = time.perf_counter()
            qvec = retrieval.encode(emb, [q["question"]], "q")[0] if args.system != "S1" else None
            res = r.retrieve(q["question"], qvec) if args.system != "S4" else r.retrieve(q["question"])
            dt = time.perf_counter() - t
            rec = {"id": q["id"], "chunk_ids": res["chunk_ids"], "facts": res.get("facts", []), "route": res.get("route"),
                   "headers": [h for h, _ in res["passages"]], "retrieval_latency_s": round(dt, 4)}
            if args.system == "S4":  # LightRAG returns the chunk text itself
                rec["texts"] = [t for _, t in res["passages"]]
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            if i % 100 == 0:
                print(f"{args.system} {args.dataset}_{args.split}: {i + 1}/{len(todo)}", flush=True)
    import torch
    log_resources("retrieve", args, {"gpu_peak_gib": round(torch.cuda.max_memory_allocated() / 2 ** 30, 3),
                                     "n": len(todo)})
    if args.system == "S4":
        r.close()


def cmd_generate(args):
    cfg = load_cfg(args.config)
    qs = data.questions(args.dataset, args.split)
    out = os.path.join(RES, "predictions", f"{args.dataset}_{args.split}", f"{args.system}{args.tag}__{args.slm}.jsonl")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    guard_test(args, cfg, out, len(qs))
    done = done_ids(out)
    ret = {}
    if args.system != "S0":
        rp = os.path.join(RES, "retrieval", f"{args.dataset}_{args.split}", f"{args.system}{args.tag}.jsonl")
        ret = {json.loads(l)["id"]: json.loads(l) for l in open(rp, encoding="utf-8")}
    chunks = {c["chunk_id"]: c for c in data.corpus(data.corpus_for(args.dataset, args.split))}
    llm.chat(args.slm, "xin chào", num_predict=2)  # load model outside the timed calls
    with open(out, "a", encoding="utf-8") as f:
        for i, q in enumerate(q for q in qs if q["id"] not in done):
            if args.system == "S0":
                p = prompts.closed_book_prompt(q["question"])
            else:
                r = ret[q["id"]]
                texts = r.get("texts") or [chunks[c]["text"] for c in r["chunk_ids"]]
                heads = r.get("headers") or [""] * len(texts)
                p = prompts.rag_prompt(q["question"], list(zip(heads, texts)), r.get("facts"))
            g = llm.chat(args.slm, p, num_predict=cfg.get("num_predict", 160))
            f.write(json.dumps({"id": q["id"], "answer": g["text"], "prompt_tokens": g["prompt_tokens"], "output_tokens": g["output_tokens"],
                                "gen_latency_s": g["latency_s"]}, ensure_ascii=False) + "\n")
            f.flush()
            if i % 100 == 0:
                print(f"{args.system}/{args.slm} {args.dataset}_{args.split}: {i + 1}", flush=True)
    log_resources("generate", args, {"slm": args.slm, "ollama_size_gib": {k: round(v / 2 ** 30, 3) for k, v in llm.loaded_models().items()},
                                     "num_ctx": llm.NUM_CTX})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["retrieve", "generate"])
    ap.add_argument("--system", required=True)
    ap.add_argument("--slm", default=llm.PRIMARY)
    ap.add_argument("--dataset", required=True, choices=["ptit", "viquad"])
    ap.add_argument("--split", required=True, choices=["dev", "test"])
    ap.add_argument("--config", default=None)
    ap.add_argument("--tag", default="", help="suffix for dev tuning variants, e.g. _k3")
    args = ap.parse_args()
    {"retrieve": cmd_retrieve, "generate": cmd_generate}[args.cmd](args)


if __name__ == "__main__":
    main()
