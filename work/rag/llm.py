"""Ollama client for the evaluated SLMs.

A private Ollama server (work/scripts/ollama_serve.sh, port 11435) reads the system model store read-only, so
parallelism and context length are controlled by this project. Decoding is deterministic (temperature 0, seed 42);
thinking is switched off for the Qwen models.
"""
import os
import time

import ollama

HOST = os.environ.get("PAPER_OLLAMA_HOST", "http://127.0.0.1:11435")
SEED = 42
NUM_CTX = int(os.environ.get("PAPER_NUM_CTX", "8192"))

SLMS = {
    "qwen3.5-4b": {"tag": "qwen3.5:4b", "think": False},
    "gemma4-e4b": {"tag": "gemma4:e4b-it-qat", "think": False},
    "llama3.2-3b": {"tag": "llama3.2:3b", "think": None},
    # the GGUF ships no chat template ({{ .Prompt }}): wrap with the prompt format of the PhoGPT README
    "phogpt-4b": {"tag": "hf.co/vinai/PhoGPT-4B-Chat-gguf:Q4_K_M", "think": None,
                  "raw_template": "### Câu hỏi: {prompt}\n### Trả lời:", "stop": ["### Câu hỏi:", "\n###"]},
    "qwen3-8b": {"tag": "qwen3:8b", "think": False},
}
PRIMARY = "qwen3.5-4b"

_client = None


def client():
    global _client
    if _client is None:
        _client = ollama.Client(host=HOST, timeout=600)
    return _client


def chat(slm, prompt, num_predict=256, num_ctx=None, fmt=None, temperature=0.0):
    """Single-turn generation. Returns dict(text, prompt_tokens, output_tokens, latency_s, load_s)."""
    m = SLMS[slm]
    kw = {}
    if m["think"] is not None:
        kw["think"] = m["think"]
    if fmt is not None:
        kw["format"] = fmt
    opts = {"temperature": temperature, "seed": SEED, "num_ctx": num_ctx or NUM_CTX, "num_predict": num_predict}
    if m.get("stop"):
        opts["stop"] = m["stop"]
    t0 = time.perf_counter()
    if m.get("raw_template"):
        r = client().generate(model=m["tag"], prompt=m["raw_template"].format(prompt=prompt), raw=True, keep_alive="30m", options=opts, **kw)
        text = r.response
    else:
        r = client().chat(model=m["tag"], messages=[{"role": "user", "content": prompt}], keep_alive="30m", options=opts, **kw)
        text = r.message.content
    dt = time.perf_counter() - t0
    return {"text": (text or "").strip(), "prompt_tokens": r.prompt_eval_count, "output_tokens": r.eval_count,
            "latency_s": round(dt, 4), "load_s": round((r.load_duration or 0) / 1e9, 4)}


def loaded_models():
    """memory footprint reported by Ollama (weights + KV cache + compute buffers) of the loaded models, bytes"""
    return {m.model: m.size for m in client().ps().models}
