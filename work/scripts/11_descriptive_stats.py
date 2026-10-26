"""Descriptive statistics of the S5 graph and of the benchmark questions (reviewer request, after GATE 5).

No model is run: the graph is rebuilt from work/index/ptit/kg_extract.jsonl exactly as rag/kg.py builds it, and the
questions are the frozen PTIT test set. Nothing here was used for tuning.

graph      entities mentioned in one chunk only; entities mentioned in >= 2 books; relations per chunk; entities
           without any relation
evidence   answerable multi-hop test questions with >= 2 distinct gold chunks: share whose gold chunks are linked in the
           graph (a) by a shared entity, (b) by a shared entity or one relation edge between their entities;
           overall and for multi_intra / multi_cross-chapter / multi_cross-book
lexical    share of the question's content syllables (document frequency < 10% of chunks) found in the text of its
           gold chunks, against a random chunk of the same book (seed 42); answerable test questions
Output: results/descriptive_stats.json
"""
import collections
import itertools
import json
import os
import random
import statistics
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(ROOT, "work"))
from rag import data  # noqa: E402
from rag.kg import KG  # noqa: E402
from rag.text import normalize_answer  # noqa: E402

random.seed(42)
kg = KG("ptit", "vn-embedding")
chunks = kg.chunks
book_of = {c["chunk_id"]: c["meta"]["book_id"] for c in chunks}

# ---------- graph ----------
ent_books = {k: {book_of[c] for c in v["chunks"]} for k, v in kg.ents.items()}
n_ent = len(kg.ents)
rels_per_chunk = collections.Counter(t[3] for t in kg.triples)
graph = {
    "entities": n_ent, "relations": len(kg.triples),
    "share_entities_one_chunk": round(sum(len(v["chunks"]) == 1 for v in kg.ents.values()) / n_ent, 4),
    "share_entities_multi_book": round(sum(len(b) >= 2 for b in ent_books.values()) / n_ent, 4),
    "share_entities_no_relation": round(sum(1 for k in kg.ents if not kg.rels.get(k)) / n_ent, 4),
    "relations_per_chunk_mean": round(len(kg.triples) / len(chunks), 2),
    "entities_per_chunk_mean": round(statistics.mean(len(kg.chunk_ents.get(c["chunk_id"], ())) for c in chunks), 2),
}

# ---------- links between gold chunks ----------
neigh = collections.defaultdict(set)  # entity -> entities one relation edge away
for s, _, o, _ in kg.triples:
    neigh[s].add(o)
    neigh[o].add(s)


def linked(a, b, one_hop):
    ea, eb = kg.chunk_ents.get(a, set()), kg.chunk_ents.get(b, set())
    if ea & eb:
        return True
    return one_hop and any(neigh[e] & eb for e in ea)


qs = [q for q in data.questions("ptit", "test") if not q["unanswerable"]]
groups = collections.defaultdict(list)
for q in qs:
    if q["type"] == "single" or len(q["gold_chunks"]) < 2:
        continue
    g = "multi_intra" if q["type"] == "multi_intra" else f"multi_cross_{q['cross_type']}"
    pairs = list(itertools.combinations(q["gold_chunks"], 2))
    # a question counts as linked when its gold chunks form one connected group under the link relation
    for name, hop in (("shared_entity", False), ("entity_or_one_relation", True)):
        comp = {c: c for c in q["gold_chunks"]}

        def find(x):
            while comp[x] != x:
                x = comp[x]
            return x
        for a, b in pairs:
            if linked(a, b, hop):
                comp[find(a)] = find(b)
        groups[(g, name)].append(len({find(c) for c in q["gold_chunks"]}) == 1)
evidence = {}
for g in ("multi_intra", "multi_cross_chapter", "multi_cross_book"):
    evidence[g] = {"n": len(groups[(g, "shared_entity")]),
                   **{name: round(statistics.mean(groups[(g, name)]), 4) for name in ("shared_entity", "entity_or_one_relation")}}
alln = {name: [v for g in ("multi_intra", "multi_cross_chapter", "multi_cross_book") for v in groups[(g, name)]]
        for name in ("shared_entity", "entity_or_one_relation")}
evidence["all_multi_hop"] = {"n": len(alln["shared_entity"]), **{k: round(statistics.mean(v), 4) for k, v in alln.items()}}

# ---------- lexical alignment ----------
toks = {c["chunk_id"]: set(normalize_answer(c["text"]).split()) for c in chunks}
df = collections.Counter(t for s in toks.values() for t in s)
content = lambda s: {t for t in set(normalize_answer(s).split()) if df[t] < 0.10 * len(chunks)}
by_book = collections.defaultdict(list)
for c in chunks:
    by_book[c["meta"]["book_id"]].append(c["chunk_id"])
gold_cov, rand_cov, n_content = [], [], []
for q in qs:
    qt = content(q["question"])
    if not qt:
        continue
    gold = set().union(*(toks[c] for c in q["gold_chunks"]))
    others = [c for c in by_book[q["book_id"]] if c not in q["gold_chunks"]]
    rnd = set().union(*(toks[c] for c in random.sample(others, len(q["gold_chunks"]))))
    gold_cov.append(len(qt & gold) / len(qt))
    rand_cov.append(len(qt & rnd) / len(qt))
    n_content.append(len(qt))
lexical = {"n": len(gold_cov), "content_syllables_per_question_mean": round(statistics.mean(n_content), 2),
           "coverage_gold_mean": round(statistics.mean(gold_cov), 4), "coverage_random_same_book_mean": round(statistics.mean(rand_cov), 4),
           "df_threshold": 0.10}

out = {"graph": graph, "gold_chunk_links": evidence, "lexical": lexical,
       "source": "work/scripts/11_descriptive_stats.py (kg_extract.jsonl, ptit_test.jsonl, all_chunks.jsonl)"}
json.dump(out, open(os.path.join(ROOT, "results", "descriptive_stats.json"), "w"), indent=1, ensure_ascii=False)
print(json.dumps(out, indent=1, ensure_ascii=False))
