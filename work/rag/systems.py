"""System registry: S0-S5 and S5 ablations -> retriever objects with a common retrieve(question, qvec) API."""
from . import data, retrieval

SYSTEMS = ["S0", "S1", "S2", "S3", "S4", "S5"]
ABLATIONS = {
    "S5-noframe": {"use_frame": False},
    "S5-norouting": {"routing": False},
    "S5-novector": {"use_text": False},
    "S5-depth1": {"depth": 1},
    "S5-depth2": {"depth": 2},
    "S5-depth3": {"depth": 3},
}


class _Text:
    """S1 / S2 / S3: top-k chunks, no headers, no facts"""

    def __init__(self, kind, corpus_name, embed_model, k, rrf_c=60, depth=50):
        self.kind, self.k, self.rrf_c, self.depth = kind, k, rrf_c, depth
        self.chunks = {c["chunk_id"]: c for c in data.corpus(corpus_name)}
        self.bm25 = retrieval.BM25(corpus_name) if kind in ("S1", "S3") else None
        self.dense = retrieval.Dense(corpus_name, embed_model) if kind in ("S2", "S3") else None
        self.embed_model = embed_model

    def retrieve(self, question, qvec=None):
        if self.kind == "S1":
            ranked = self.bm25.search(question, self.k)
        elif self.kind == "S2":
            ranked = self.dense.search(question, self.k, qvec=qvec)
        else:
            ranked = retrieval.rrf([self.bm25.search(question, self.depth), self.dense.search(question, self.depth, qvec=qvec)],
                                   k=self.k, c=self.rrf_c)
        ids = [cid for cid, _ in ranked][:self.k]
        return {"chunk_ids": ids, "passages": [("", self.chunks[c]["text"]) for c in ids], "facts": []}


def make(system, corpus_name, cfg):
    """cfg: dict from results/configs/systems_config.json"""
    emb = cfg["embedding"]
    if system in ("S1", "S2", "S3"):
        return _Text(system, corpus_name, emb, cfg["k"], cfg.get("rrf_c", 60))
    if system == "S4":
        from .lightrag_sys import LightRAGRetriever
        return LightRAGRetriever(corpus_name, emb, chunk_top_k=cfg["k"], **cfg.get("S4", {}))
    if system == "S5" or system in ABLATIONS:
        from .kg import S5Retriever
        from .router import Router
        params = {"k": cfg["k"], **cfg.get("S5", {}), **ABLATIONS.get(system, {})}
        return S5Retriever(corpus_name, emb, router=Router.load(), **params)
    raise ValueError(system)
