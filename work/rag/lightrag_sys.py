"""S4: LightRAG (lightrag-hku) as the off-the-shelf GraphRAG baseline.

- Graph built with the primary SLM (qwen3.5:4b, thinking off) and the embedding chosen on dev.
- Each corpus chunk is inserted as one document (doc id and file_path = chunk_id) with chunk_token_size large enough
  that LightRAG does not re-split it, so every system retrieves over the same units.
- One extraction pass per chunk (entity_extract_max_gleaning=0), the same LLM budget as S5's extraction.
- Retrieval only: aquery_data() returns LightRAG's selected entities, relations and chunks (keyword extraction by the
  primary SLM); the answer is generated with the shared prompt (prompts.rag_prompt) for every SLM.
"""
import asyncio
import os

import numpy as np

from . import data, llm, retrieval

LIGHTRAG_DIR = os.path.join(data.WORK, "index")


def _llm_func(host):
    import ollama
    client = ollama.AsyncClient(host=host, timeout=900)

    async def f(prompt, system_prompt=None, history_messages=None, keyword_extraction=False, entity_extraction=False, **kw):
        msgs = []
        if system_prompt:
            msgs.append({"role": "system", "content": system_prompt})
        msgs += list(history_messages or [])
        msgs.append({"role": "user", "content": prompt})
        fmt = None
        rf = kw.get("response_format")
        if keyword_extraction or (isinstance(rf, dict) and rf.get("type") == "json_object"):
            fmt = "json"
        elif isinstance(rf, dict) and rf.get("type") == "json_schema":
            fmt = rf["json_schema"].get("schema", rf["json_schema"])
        r = await client.chat(model=llm.SLMS[llm.PRIMARY]["tag"], messages=msgs, think=False, format=fmt, keep_alive="30m",
                              options={"temperature": 0.0, "seed": llm.SEED, "num_ctx": llm.NUM_CTX, "num_predict": 3000})
        return r.message.content or ""
    return f


def _embed_func(model):
    from lightrag.utils import EmbeddingFunc

    async def f(texts, context="document", **kw):
        kind = "q" if context == "query" else "d"
        return await asyncio.to_thread(lambda: retrieval.encode(model, list(texts), kind))
    return EmbeddingFunc(embedding_dim=1024, func=f, max_token_size=retrieval.EMBEDDERS[model]["max_len"], model_name=model,
                         supports_asymmetric=True)


def make(corpus_name, embed_model, host=llm.HOST):
    from lightrag import LightRAG
    wd = os.path.join(LIGHTRAG_DIR, corpus_name, "lightrag")
    os.makedirs(wd, exist_ok=True)
    rag = LightRAG(working_dir=wd, llm_model_func=_llm_func(host), llm_model_name=llm.SLMS[llm.PRIMARY]["tag"],
                   embedding_func=_embed_func(embed_model), chunk_token_size=4000, chunk_overlap_token_size=0,
                   entity_extract_max_gleaning=0, llm_model_max_async=2, embedding_func_max_async=1, max_parallel_insert=2,
                   addon_params={"language": "Vietnamese"}, enable_llm_cache=True)
    return rag


async def build(corpus_name, embed_model, host):
    rag = make(corpus_name, embed_model, host)
    await rag.initialize_storages()
    chunks = data.corpus(corpus_name)
    B = 20
    for i in range(0, len(chunks), B):
        part = chunks[i:i + B]
        await rag.ainsert([c["text"] for c in part], ids=[c["chunk_id"] for c in part], file_paths=[c["chunk_id"] for c in part])
        print(f"lightrag {corpus_name}: {min(i + B, len(chunks))}/{len(chunks)}", flush=True)
    await rag.finalize_storages()


class LightRAGRetriever:
    def __init__(self, corpus_name, embed_model, mode="mix", top_k=20, chunk_top_k=5, max_entity_tokens=1500,
                 max_relation_tokens=1500, max_total_tokens=6000):
        from lightrag import QueryParam
        self.rag = make(corpus_name, embed_model)
        self.loop = asyncio.new_event_loop()
        self.loop.run_until_complete(self.rag.initialize_storages())
        self.param = dict(mode=mode, top_k=top_k, chunk_top_k=chunk_top_k, max_entity_tokens=max_entity_tokens,
                          max_relation_tokens=max_relation_tokens, max_total_tokens=max_total_tokens, enable_rerank=False)
        self.QueryParam = QueryParam

    def retrieve(self, question):
        res = self.loop.run_until_complete(self.rag.aquery_data(question, self.QueryParam(**self.param)))
        d = res.get("data", {})
        facts = [f"{e['entity_name']}: {e.get('description', '')}".strip() for e in d.get("entities", [])]
        facts += [f"{r['src_id']} — {r.get('keywords', '')} — {r['tgt_id']}: {r.get('description', '')}" for r in d.get("relationships", [])]
        chunks = [(c.get("file_path"), c.get("content", "")) for c in d.get("chunks", [])]
        return {"chunk_ids": [cid for cid, _ in chunks], "passages": [("", t) for _, t in chunks], "facts": facts}

    def close(self):
        self.loop.run_until_complete(self.rag.finalize_storages())
