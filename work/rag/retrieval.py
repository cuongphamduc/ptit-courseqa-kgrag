"""Retrievers for S1 (BM25), S2 (dense) and S3 (hybrid, reciprocal rank fusion).

Dense encoders whose maximum input is shorter than a chunk (multilingual-e5-large: 512, vietnamese-bi-encoder: 256
sub-word tokens) encode each chunk as consecutive sentence-packed windows; the chunk score is the maximum window score.
Chunk embeddings are cached in work/index/<corpus>/<model>.npz.
"""
import os
import re
import time

import numpy as np

from . import data
from .text import bm25_tokens, nfc, segment

INDEX = os.path.join(data.WORK, "index")

EMBEDDERS = {
    "bge-m3": {"hf": "BAAI/bge-m3", "max_len": 1024, "q": "", "d": ""},
    "vn-embedding": {"hf": "AITeamVN/Vietnamese_Embedding", "max_len": 1024, "q": "", "d": ""},
    "qwen3-emb-0.6b": {"hf": "Qwen/Qwen3-Embedding-0.6B", "max_len": 1024,
                       "q": "Instruct: Given a question, retrieve passages that answer the question\nQuery:", "d": ""},
    "me5-large": {"hf": "intfloat/multilingual-e5-large", "max_len": 512, "q": "query: ", "d": "passage: "},
    "vn-bi-encoder": {"hf": "bkai-foundation-models/vietnamese-bi-encoder", "max_len": 256, "q": "", "d": "", "segment": True},
}


class BM25:
    name = "bm25"

    def __init__(self, corpus_name):
        from rank_bm25 import BM25Okapi
        self.chunks = data.corpus(corpus_name)
        self.ids = [c["chunk_id"] for c in self.chunks]
        t = time.perf_counter()
        self.bm25 = BM25Okapi([bm25_tokens(c["text"]) for c in self.chunks])
        self.build_s = time.perf_counter() - t

    def search(self, query, k=100):
        s = self.bm25.get_scores(bm25_tokens(query))
        top = np.argsort(-s)[:k]
        return [(self.ids[i], float(s[i])) for i in top]


_models = {}


def load_encoder(name, device="cuda"):
    if name not in _models:
        import torch
        from sentence_transformers import SentenceTransformer
        spec = EMBEDDERS[name]
        m = SentenceTransformer(spec["hf"], device=device, model_kwargs={"torch_dtype": torch.float16}, trust_remote_code=False)
        m.max_seq_length = spec["max_len"]
        _models[name] = m
    return _models[name]


def unload_encoders():
    import gc

    import torch
    _models.clear()
    gc.collect()
    torch.cuda.empty_cache()


def _prep(name, text):
    text = nfc(text)
    return segment(text) if EMBEDDERS[name].get("segment") else text


def _windows(name, text):
    """split text into sentence-packed windows that fit the encoder's max length (minus prefix and specials)"""
    m = load_encoder(name)
    spec = EMBEDDERS[name]
    budget = spec["max_len"] - 8 - len(m.tokenizer.tokenize(spec["d"])) if spec["d"] else spec["max_len"] - 8
    ntok = lambda s: len(m.tokenizer.tokenize(_prep(name, s)))
    if ntok(text) <= budget:
        return [text]
    sents = [s for s in re.split(r"(?<=[.!?;:])\s+|\n+", text) if s.strip()]
    wins, cur = [], ""
    for s in sents:
        while ntok(s) > budget:  # a single over-long sentence: cut by words
            words = s.split()
            lo = max(1, int(len(words) * budget / ntok(s)))
            head, s = " ".join(words[:lo]), " ".join(words[lo:])
            if cur:
                wins.append(cur)
                cur = ""
            wins.append(head)
        cand = (cur + " " + s).strip()
        if cur and ntok(cand) > budget:
            wins.append(cur)
            cur = s
        else:
            cur = cand
    if cur:
        wins.append(cur)
    return wins


def encode(name, texts, kind, batch_size=16):
    m = load_encoder(name)
    spec = EMBEDDERS[name]
    pre = spec["q"] if kind == "q" else spec["d"]
    texts = [pre + _prep(name, t) for t in texts]
    return m.encode(texts, batch_size=batch_size, normalize_embeddings=True, convert_to_numpy=True, show_progress_bar=False).astype(np.float32)


class Dense:
    def __init__(self, corpus_name, model):
        self.name = f"dense:{model}"
        self.model = model
        self.chunks = data.corpus(corpus_name)
        self.ids = [c["chunk_id"] for c in self.chunks]
        path = os.path.join(INDEX, corpus_name, f"{model}.npz")
        if os.path.exists(path):
            z = np.load(path)
            self.emb, self.owner, self.build_s = z["emb"], z["owner"], float(z["build_s"])
        else:
            os.makedirs(os.path.dirname(path), exist_ok=True)
            t = time.perf_counter()
            wins, owner = [], []
            for i, c in enumerate(self.chunks):
                for w in _windows(model, c["text"]):
                    wins.append(w)
                    owner.append(i)
            self.emb = encode(model, wins, "d")
            self.owner = np.array(owner)
            self.build_s = time.perf_counter() - t
            np.savez(path, emb=self.emb, owner=self.owner, build_s=self.build_s)
        self.n_windows = len(self.owner)

    def scores(self, qvec):
        s = self.emb @ qvec
        best = np.full(len(self.chunks), -1e9, dtype=np.float32)
        np.maximum.at(best, self.owner, s)
        return best

    def search(self, query, k=100, qvec=None):
        if qvec is None:
            qvec = encode(self.model, [query], "q")[0]
        s = self.scores(qvec)
        top = np.argsort(-s)[:k]
        return [(self.ids[i], float(s[i])) for i in top]


def rrf(rankings, k=100, c=60):
    """reciprocal rank fusion of several [(id, score)] lists"""
    agg = {}
    for r in rankings:
        for rank, (cid, _) in enumerate(r):
            agg[cid] = agg.get(cid, 0.0) + 1.0 / (c + rank + 1)
    return sorted(agg.items(), key=lambda x: -x[1])[:k]
