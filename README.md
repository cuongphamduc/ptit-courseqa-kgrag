# ptit-courseqa-kgrag

Code for the PTIT-CourseQA benchmark and the controlled comparison of retrieval-augmented generation systems for
Vietnamese small language models (SLMs), as reported in:

> P. D. Cuong, "Knowledge-Graph RAG for Vietnamese Small Language Models on University Course Materials: A Benchmark
> and a Controlled Evaluation", submitted to the *Journal of Science and Technology on Information and Communications* (JSTIC).

- Benchmark, annotations and all system outputs: https://huggingface.co/datasets/cuongphamduc/PTIT-CourseQA
- Archived release (data + this code): https://doi.org/10.5281/zenodo.23086349

## Systems

| Id | System |
|---|---|
| S0 | closed book (no retrieval) |
| S1 | BM25 over word-segmented text (pyvi) |
| S2 | dense retrieval (`AITeamVN/Vietnamese_Embedding`, chosen on dev) |
| S3 | hybrid: reciprocal-rank fusion of S1 and S2 |
| S4 | LightRAG 1.5.7, `mix` mode, graph built by the primary SLM |
| S5 | structure-framed KG-RAG: KG extracted by the primary SLM inside a book/chapter/section frame, question-type router, text + graph slot fusion |

Readers: Qwen3.5-4B (primary), Gemma-4-E4B, Llama-3.2-3B, PhoGPT-4B, Qwen3-8B, served by Ollama 0.32.5
(temperature 0, `num_ctx` 8192). Judge: qwen3.6 (36B MoE). Exact tags: `outputs/configs/models_hardware.json` in the
dataset release.

## Layout

```
work/rag/            library: text processing, retrieval (BM25, dense, RRF), KG extraction and graph retrieval (S5),
                     LightRAG wrapper (S4), router, prompts, Ollama client, judge
work/scripts/        numbered pipeline, one script per stage (docstrings give inputs/outputs)
work/benchmark/      ptit_test.jsonl, ptit_dev.jsonl
work/corpus/books.json
release_data/        chunk index without text (ids, pages, SHA-256) and book metadata
paper/scripts/       make_numbers.py: every number in the manuscript, generated from results/
paper/figures/scripts/  figure code
prompts/             question-generation instructions and annotation guidelines (Vietnamese)
```

## Reproducing

1. **Environment.** Python 3.11, `pip install -r requirements.txt` (PyTorch built for your CUDA), `pdftotext`
   (poppler 24.02), Ollama with the five reader models and the judge. `work/scripts/ollama_ctl.sh start|stop` runs a
   private Ollama server on port 11435.
2. **Corpus.** The textbooks are copyrighted by PTIT and are not distributed. Place the 11 PDFs under
   `dataset/PTIT/<field>/<file>` as listed in `work/corpus/books.json`, then
   ```
   python work/scripts/02_preprocess.py
   python work/scripts/02b_fix_section_labels.py
   ```
   and check the chunk hashes against `release_data/chunk_index.jsonl`.
3. **ViQuAD subset.** Download https://huggingface.co/datasets/taidng/UIT-ViQuAD2.0 to `work/data/UIT-ViQuAD2.0/`
   and run `python work/scripts/04_viquad_subset.py`.
4. **Indexes and tuning (dev only).** `05a_select_embedding.py`, `05b_build_kg_extract.py`, then
   `06_pipeline.py build tune`, `06_pipeline.py freeze`.
5. **Test (run once).** `06_pipeline.py test ablate eval`, then `06_judge.py`, `06_costs.py`.
   `06_run.py` refuses a test run unless the dev-chosen configuration is frozen, and never overwrites a complete test file.
6. **Hallucination study.** `07a_export_halluc.py` (annotation sheets), `07b_halluc_analysis.py` (after labelling).

Stage 3 (`03a`–`03g`) documents how the benchmark was built; question drafting used Claude Code subagents following
`prompts/question_generation_INSTRUCTIONS_vi.md`, so it is not re-runnable as a script. The released benchmark files
are the final output of that stage.

All randomness uses seed 42. The figure scripts optionally import `audit_panel_alignment` from the *nature-figure*
skill for a layout check; remove that call if it is not available.

## Licence

CC BY 4.0 (`LICENSE`). The licence does not cover the PTIT textbooks, which are not included.

## Citation

See `CITATION.cff`.
