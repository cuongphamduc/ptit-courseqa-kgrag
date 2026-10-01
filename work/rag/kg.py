"""S5: structure-framed knowledge graph + hybrid graph/vector retrieval with question-type routing.

Graph (built from kg_extract.jsonl + corpus metadata, no extra LLM calls):
  - knowledge layer: entity nodes (canonicalised SLM entities), relation edges (SLM triples), entity-chunk mentions;
  - structural frame: book > chapter > section > chunk from the corpus metadata (ViQuAD: article title > paragraph).
Retrieval for question q:
  1. text branch: BM25 + dense (RRF) ranking of chunks;
  2. route = router(q) in {single, multi}; the single route returns the text-branch top-k;
  3. multi route: link q to seed entities (alias string match + entity-embedding similarity), expand over relation
     edges up to `depth` hops (weight decay gamma^hop, IDF over mention counts), score chunks by the reached entities,
     fuse with the text branch (fusion="slots": text top-(k-m) + m best graph chunks not yet selected; or weighted
     RRF), and pick top-k with at most `per_section` chunks per section (frame);
     the strongest relations among reached entities are added as facts.
Every chunk is shown with its frame path (book > chapter > section) as header.
Ablations: use_frame=False (no section cap, no headers), routing=False (multi route for all), use_text=False
(graph only: seeds from entity linking), depth in {1,2,3}.
"""
import collections
import json
import math
import os
import re

import numpy as np

from . import data, retrieval
from .text import nfc, normalize_answer

DEFAULTS = {"k": 5, "depth": 2, "gamma": 0.5, "per_section": 2, "n_facts": 10, "ent_topm": 10, "ent_min_sim": 0.55,
            "rrf_c": 60, "text_depth": 50, "use_frame": True, "routing": True, "use_text": True,
            "fusion": "slots", "graph_slots": 2, "graph_weight": 1.0}


def canon(name):
    return normalize_answer(re.sub(r"\s+", " ", nfc(name)))


def split_alias(name):
    """'Kho dữ liệu (Data Warehouse)' -> ('kho dữ liệu', ['data warehouse'])"""
    m = re.match(r"^(.*?)\s*\((.+)\)\s*$", nfc(name))
    if m and m.group(1).strip():
        return canon(m.group(1)), [canon(a) for a in re.split(r"[,;/]", m.group(2)) if canon(a)]
    return canon(name), []


class KG:
    def __init__(self, corpus_name, embed_model):
        self.corpus_name, self.embed_model = corpus_name, embed_model
        self.chunks = data.corpus(corpus_name)
        self.cidx = {c["chunk_id"]: i for i, c in enumerate(self.chunks)}
        self.ents = {}                      # key -> {names, types, desc, chunks:set}
        self.alias = {}                     # alias -> key
        self.rels = collections.defaultdict(list)   # key -> [(rel, other_key, chunk_id, direction)]
        self.triples = []                   # (src_key, rel, tgt_key, chunk_id)
        path = os.path.join(data.WORK, "index", corpus_name, "kg_extract.jsonl")
        self.extract_stats = collections.Counter()
        for line in open(path, encoding="utf-8"):
            r = json.loads(line)
            self.extract_stats["chunks"] += 1
            self.extract_stats["json_ok"] += r["ok"]
            self.extract_stats["latency_s"] += r["latency_s"]
            self.extract_stats["output_tokens"] += r["output_tokens"] or 0
            local = {}
            for e in r["entities"]:
                if not isinstance(e, dict) or not e.get("name") or not canon(e["name"]):
                    continue
                key = self._add_entity(e)
                self.ents[key]["chunks"].add(r["chunk_id"])
                local[canon(e["name"])] = key
                local[split_alias(e["name"])[0]] = key
            for t in r["relations"]:
                if not isinstance(t, dict):
                    continue
                s, o = self._resolve(t.get("source", ""), local), self._resolve(t.get("target", ""), local)
                if s and o and s != o:
                    rel = nfc(t.get("relation", "")).strip()
                    self.triples.append((s, rel, o, r["chunk_id"]))
                    self.rels[s].append((rel, o, r["chunk_id"], "out"))
                    self.rels[o].append((rel, s, r["chunk_id"], "in"))
        n = len(self.chunks)
        self.idf = {k: math.log(1 + n / len(v["chunks"])) for k, v in self.ents.items() if v["chunks"]}
        self.max_idf = max(self.idf.values()) if self.idf else 1.0
        self.chunk_ents = collections.defaultdict(set)
        for k, v in self.ents.items():
            for c in v["chunks"]:
                self.chunk_ents[c].add(k)
        # alias automaton: aliases of >= 2 syllables, or single tokens of >= 3 characters containing a digit/uppercase acronym
        self.alias_list = sorted((a for a in self.alias if len(a.split()) >= 2 or (len(a) >= 3 and re.search(r"\d|[a-z]", a) and len(a) <= 12)),
                                 key=len, reverse=True)
        self._ent_emb = None

    def _add_entity(self, e):
        main, aliases = split_alias(e["name"])
        key = self.alias.get(main) or main
        if key not in self.ents:
            self.ents[key] = {"names": collections.Counter(), "types": collections.Counter(), "desc": [], "chunks": set()}
        ent = self.ents[key]
        ent["names"][nfc(e["name"]).strip()] += 1
        ent["types"][e.get("type", "KHÁC")] += 1
        if e.get("description") and len(ent["desc"]) < 3:
            ent["desc"].append(nfc(e["description"]).strip())
        self.alias.setdefault(main, key)
        for a in aliases:
            self.alias.setdefault(a, key)
        return key

    def _resolve(self, name, local):
        c = canon(name)
        if c in local:
            return local[c]
        main, _ = split_alias(name)
        return local.get(main) or self.alias.get(c) or self.alias.get(main)

    # ---------- frame (structure) ----------
    def frame(self, cid):
        c = self.chunks[self.cidx[cid]]
        m = c["meta"]
        if self.corpus_name == "ptit":
            return f"{c['title']} › {m.get('chapter') or ''} › {m.get('section') or ''}".rstrip(" ›")
        return c["title"]

    def section_key(self, cid):
        m = self.chunks[self.cidx[cid]]["meta"]
        if self.corpus_name == "ptit":
            return (m["book_id"], m.get("chapter_no"), (m.get("section") or "").split(" ", 1)[0])
        return (m["book_id"],)

    # ---------- entity linking ----------
    def entity_embeddings(self):
        if self._ent_emb is None:
            path = os.path.join(data.WORK, "index", self.corpus_name, f"kg_entities_{self.embed_model}.npz")
            self.ent_keys = sorted(self.ents)
            if os.path.exists(path) and list(np.load(path)["keys"]) == self.ent_keys:
                self._ent_emb = np.load(path)["emb"]
            else:
                texts = [f"{self.ents[k]['names'].most_common(1)[0][0]}: {(self.ents[k]['desc'] or [''])[0]}" for k in self.ent_keys]
                self._ent_emb = retrieval.encode(self.embed_model, texts, "d", batch_size=64)
                np.savez(path, emb=self._ent_emb, keys=np.array(self.ent_keys))
        return self._ent_emb

    def link(self, question, qvec, topm, min_sim):
        qn = f" {canon(question)} "
        seeds = {}
        for a in self.alias_list:
            if f" {a} " in qn:
                seeds[self.alias[a]] = 1.0
        sims = self.entity_embeddings() @ qvec
        for i in np.argsort(-sims)[:topm]:
            if sims[i] >= min_sim:
                k = self.ent_keys[i]
                seeds[k] = max(seeds.get(k, 0.0), float(sims[i]))
        return seeds

    def expand(self, seeds, depth, gamma):
        """BFS over relation edges; returns {entity: weight} with weight = seed * gamma^hop * idf/max_idf"""
        reached = {k: (w, 0) for k, w in seeds.items()}
        frontier = dict(seeds)
        for hop in range(1, depth + 1):
            nxt = {}
            for k, w in frontier.items():
                for _, o, _, _ in self.rels.get(k, []):
                    if o not in reached:
                        nxt[o] = max(nxt.get(o, 0.0), w)
            for o, w in nxt.items():
                reached[o] = (w, hop)
            frontier = nxt
        return {k: w * gamma ** h * self.idf.get(k, 0.0) / self.max_idf for k, (w, h) in reached.items()}

    def stats(self):
        return {"entities": len(self.ents), "relations": len(self.triples), "mentions": sum(len(v["chunks"]) for v in self.ents.values()),
                "chunks_with_entities": len(self.chunk_ents), "aliases": len(self.alias),
                "frame_nodes": len({self.section_key(c["chunk_id"]) for c in self.chunks}) + len({c["meta"]["book_id"] for c in self.chunks}),
                "extraction": dict(self.extract_stats)}


class S5Retriever:
    def __init__(self, corpus_name, embed_model, router=None, **params):
        self.p = {**DEFAULTS, **params}
        self.kg = KG(corpus_name, embed_model)
        self.bm25 = retrieval.BM25(corpus_name)
        self.dense = retrieval.Dense(corpus_name, embed_model)
        self.router = router
        self.embed_model = embed_model

    def text_rank(self, question, qvec):
        n = self.p["text_depth"]
        return retrieval.rrf([self.bm25.search(question, n), self.dense.search(question, n, qvec=qvec)], k=n, c=self.p["rrf_c"])

    def retrieve(self, question, qvec=None):
        p, kg = self.p, self.kg
        if qvec is None:
            qvec = retrieval.encode(self.embed_model, [question], "q")[0]
        route = "multi"
        if p["routing"] and p["use_text"] and self.router is not None:
            route = self.router.route(question, qvec)
        text = self.text_rank(question, qvec) if p["use_text"] else []
        facts, seeds = [], {}
        if route == "single":
            ranked = [cid for cid, _ in text]
        else:
            seeds = kg.link(question, qvec, p["ent_topm"], p["ent_min_sim"])
            w = kg.expand(seeds, p["depth"], p["gamma"])
            gscore = collections.Counter()
            for e, we in w.items():
                for c in kg.ents[e]["chunks"]:
                    gscore[c] += we
            graph = gscore.most_common(p["text_depth"])
            if not text:
                ranked = [cid for cid, _ in graph]
            elif p["fusion"] == "slots":
                # text top-(k - m) is kept; m slots go to the best graph chunks not already selected
                head = [cid for cid, _ in text[:p["k"] - p["graph_slots"]]]
                extra = [cid for cid, _ in graph if cid not in head][:p["graph_slots"]]
                ranked = head + extra + [cid for cid, _ in text[p["k"] - p["graph_slots"]:] if cid not in extra]
            else:  # weighted reciprocal rank fusion
                agg = collections.Counter()
                for wgt, lst in ((1.0, text), (p["graph_weight"], graph)):
                    for rank, (cid, _) in enumerate(lst):
                        agg[cid] += wgt / (p["rrf_c"] + rank + 1)
                ranked = [cid for cid, _ in agg.most_common(p["text_depth"])]
            ranked_w = sorted(((kg_w, s, r, o) for s, r, o, _ in kg.triples
                               if s in w and o in w for kg_w in [w[s] + w[o]]), key=lambda x: -x[0])
            seen = set()
            for _, s, r, o in ranked_w:
                f = f"{kg.ents[s]['names'].most_common(1)[0][0]} — {r} — {kg.ents[o]['names'].most_common(1)[0][0]}"
                if f not in seen:
                    seen.add(f)
                    facts.append(f)
                if len(facts) >= p["n_facts"]:
                    break
        chosen, per = [], collections.Counter()
        for cid in ranked:
            if p["use_frame"] and route == "multi":
                sk = kg.section_key(cid)
                if per[sk] >= p["per_section"]:
                    continue
                per[sk] += 1
            chosen.append(cid)
            if len(chosen) >= p["k"]:
                break
        heads = [f"(Nguồn: {kg.frame(c)})" if p["use_frame"] else "" for c in chosen]
        texts = [kg.chunks[kg.cidx[c]]["text"] for c in chosen]
        return {"route": route, "chunk_ids": chosen, "passages": list(zip(heads, texts)), "facts": facts, "n_seeds": len(seeds)}
