"""Stage 5b: S5 entity/relation extraction with the primary SLM (qwen3.5:4b) for the PTIT corpus and the ViQuAD
test corpus. Resumable; output work/index/<corpus>/kg_extract.jsonl (per-chunk latency and token counts included)."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from rag import kg_extract  # noqa: E402

for name in sys.argv[1:] or ["ptit", "viquad_test"]:
    kg_extract.build(name)
