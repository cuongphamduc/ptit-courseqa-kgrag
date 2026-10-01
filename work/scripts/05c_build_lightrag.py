"""Stage 5c: build the S4 LightRAG graphs (primary SLM on the private Ollama server, port 11435) for the PTIT
corpus and the ViQuAD test corpus. LightRAG keeps doc status, so a rerun resumes. Wall time is logged."""
import asyncio
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from rag import data, lightrag_sys  # noqa: E402

EMB = json.load(open(os.path.join(data.ROOT, "results", "configs", "embedding_choice.json")))["chosen"]
for name in sys.argv[1:] or ["ptit", "viquad_test"]:
    t = time.time()
    asyncio.run(lightrag_sys.build(name, EMB, "http://127.0.0.1:11435"))
    with open(os.path.join(data.WORK, "index", name, "lightrag_build_time.json"), "a") as f:
        f.write(json.dumps({"corpus": name, "wall_s": round(time.time() - t, 1), "note": "sequential, alone on the GPU"}) + "\n")
