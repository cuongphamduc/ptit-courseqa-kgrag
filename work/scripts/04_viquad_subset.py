"""Stage 4: ViQuAD2.0 evaluation subset (original labels, no re-annotation).

The public test split has no labels (work/inventory/viquad_splits.json), so - as decided at GATE 1 - the evaluation
subset is drawn from the *validation* split:
  - test: 1,000 questions, stratified by (title x is_impossible) with largest-remainder allocation, so the
    unanswerable ratio of the split is kept; retrieval corpus = all unique contexts of the validation split;
  - dev (for tuning only): 200 questions from the *train* split. Train titles are drawn at random until their
    contexts reach the size of the validation corpus; the dev corpus is all contexts of those titles.
Each context (one Wikipedia paragraph) is one retrieval chunk.

Output: work/benchmark/{viquad_test.jsonl, viquad_dev.jsonl, viquad_subset_ids.txt, viquad_dev_ids.txt,
        viquad_corpus_test.jsonl, viquad_corpus_dev.jsonl}, results/configs/viquad_config.json, results/viquad_stats.json
"""
import collections
import hashlib
import json
import os
import random

import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA = os.path.join(ROOT, "work", "data", "UIT-ViQuAD2.0", "data")
OUT = os.path.join(ROOT, "work", "benchmark")
SEED, N_TEST, N_DEV = 42, 1000, 200


def load(split):
    f = next(x for x in os.listdir(DATA) if x.startswith(split))
    return pd.read_parquet(os.path.join(DATA, f))


def ctx_id(title, context):
    return "VQ-" + hashlib.md5((title + "\n" + context).encode("utf-8")).hexdigest()[:12]


def allocate(total, weights):
    s = sum(weights.values())
    exact = {k: total * w / s for k, w in weights.items()}
    alloc = {k: int(x) for k, x in exact.items()}
    for k in sorted(exact, key=lambda k: (-(exact[k] - int(exact[k])), str(k)))[:total - sum(alloc.values())]:
        alloc[k] += 1
    return alloc


def stratified(df, n, rng):
    cells = collections.defaultdict(list)
    for i, r in df.iterrows():
        cells[(r["title"], bool(r["is_impossible"]))].append(i)
    alloc = allocate(n, {k: len(v) for k, v in cells.items()})
    idx = []
    for k in sorted(cells, key=str):
        idx += rng.sample(sorted(cells[k]), alloc[k])
    return df.loc[sorted(idx)]


def record(r, split):
    imp = bool(r["is_impossible"])
    ans = [] if imp else [str(a) for a in r["answers"]["text"]]
    plaus = [] if r["plausible_answers"] is None else [str(a) for a in r["plausible_answers"]["text"]]
    return {"id": r["id"], "uit_id": r["uit_id"], "split": split, "title": r["title"], "question": r["question"],
            "answers": ans, "answer": ans[0] if ans else None, "is_impossible": imp, "plausible_answers": plaus,
            "type": "unanswerable" if imp else "answerable", "gold_context_id": ctx_id(r["title"], r["context"])}


def corpus(df):
    seen = {}
    for _, r in df.drop_duplicates(["title", "context"]).iterrows():
        seen[ctx_id(r["title"], r["context"])] = {"chunk_id": ctx_id(r["title"], r["context"]), "title": r["title"], "text": r["context"]}
    return list(seen.values())


def write(path, rows):
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def main():
    rng = random.Random(SEED)
    val, train = load("validation"), load("train")

    test = stratified(val, N_TEST, rng)
    test_corpus = corpus(val)

    titles = sorted(train["title"].unique())
    rng.shuffle(titles)
    target_ctx, chosen, n_ctx = val["context"].nunique(), [], 0
    for t in titles:
        if n_ctx >= target_ctx:
            break
        chosen.append(t)
        n_ctx += train.loc[train["title"] == t, "context"].nunique()
    tr = train[train["title"].isin(chosen)]
    dev = stratified(tr, N_DEV, rng)
    dev_corpus = corpus(tr)

    write(os.path.join(OUT, "viquad_test.jsonl"), [record(r, "test") for _, r in test.iterrows()])
    write(os.path.join(OUT, "viquad_dev.jsonl"), [record(r, "dev") for _, r in dev.iterrows()])
    write(os.path.join(OUT, "viquad_corpus_test.jsonl"), test_corpus)
    write(os.path.join(OUT, "viquad_corpus_dev.jsonl"), dev_corpus)
    open(os.path.join(OUT, "viquad_subset_ids.txt"), "w").write("\n".join(test["id"]) + "\n")
    open(os.path.join(OUT, "viquad_dev_ids.txt"), "w").write("\n".join(dev["id"]) + "\n")

    assert set(test_corpus_ids := {c["chunk_id"] for c in test_corpus}) >= {record(r, "t")["gold_context_id"] for _, r in test.iterrows()}
    cfg = {"seed": SEED, "source": "UIT-ViQuAD2.0 (Hugging Face parquet, work/data/UIT-ViQuAD2.0)",
           "test_from": "validation (public test split has no labels)", "n_test": N_TEST,
           "stratify": "title x is_impossible, largest remainder", "test_corpus": "all unique contexts of validation",
           "dev_from": "train", "n_dev": N_DEV, "dev_titles": chosen, "dev_corpus": "all unique contexts of dev_titles",
           "chunking": "one context = one chunk"}
    json.dump(cfg, open(os.path.join(ROOT, "results", "configs", "viquad_config.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)

    def st(df, pool):
        return {"n": len(df), "unanswerable": int(df["is_impossible"].sum()), "unanswerable_ratio": round(float(df["is_impossible"].mean()), 4),
                "source_unanswerable_ratio": round(float(pool["is_impossible"].mean()), 4), "titles": int(df["title"].nunique()),
                "question_len_mean": round(float(df["question"].str.split().str.len().mean()), 2),
                "answer_len_mean": round(float(pd.Series([len(str(a["text"][0]).split()) for a, imp in zip(df["answers"], df["is_impossible"]) if not imp]).mean()), 2)}
    stats = {"test": {**st(test, val), "corpus_chunks": len(test_corpus)}, "dev": {**st(dev, tr), "corpus_chunks": len(dev_corpus)}}
    json.dump(stats, open(os.path.join(ROOT, "results", "viquad_stats.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(json.dumps(stats, indent=1))


if __name__ == "__main__":
    main()
