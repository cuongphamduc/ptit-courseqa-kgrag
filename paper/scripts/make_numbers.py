"""Generate every number used in the manuscript from ./results (and ./work/benchmark via results/benchmark_stats.json).

Outputs
  ../numbers.tex                 \\newcommand macros used in the prose (one source of truth)
  ../tables/table_main.tex       F1_all, 6 systems x 5 SLMs, PTIT test and ViQuAD test (table*)
  ../tables/table_primary.tex    qwen3.5-4b detail on PTIT test
  ../../results/corpus_stats.json   corpus totals copied from work/corpus/stats.json (so the paper cites ./results only)
Run: work/.venv/bin/python paper/scripts/make_numbers.py
"""
import json
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
PAPER = os.path.dirname(HERE)
ROOT = os.path.dirname(PAPER)
R = os.path.join(ROOT, "results")
J = lambda *p: json.load(open(os.path.join(R, *p), encoding="utf-8"))
SLMS = ["qwen3.5-4b", "gemma4-e4b", "llama3.2-3b", "phogpt-4b", "qwen3-8b"]
SLM_TEX = {"qwen3.5-4b": "Qwen3.5-4B", "gemma4-e4b": "Gemma-4-E4B", "llama3.2-3b": "Llama-3.2-3B", "phogpt-4b": "PhoGPT-4B",
           "qwen3-8b": "Qwen3-8B"}
SYS = ["S0", "S1", "S2", "S3", "S4", "S5"]
SYS_TEX = {"S0": "S0 no retrieval", "S1": "S1 BM25", "S2": "S2 dense", "S3": "S3 hybrid", "S4": "S4 LightRAG", "S5": "S5 (ours)"}

M = {}


DIGITS = dict(zip("0123456789", ["Zero", "One", "Two", "Three", "Four", "Five", "Six", "Seven", "Eight", "Nine"]))


def put(name, value, fmt="{:.3f}"):
    name = "".join(DIGITS.get(c, c) for c in name)  # LaTeX command names cannot contain digits
    M[name] = fmt.format(value) if not isinstance(value, str) else value


def pct(x, d=1):
    return f"{100 * x:.{d}f}"


def thousands(n):
    return f"{int(n):,}".replace(",", "{,}")


def main():
    # corpus totals -> results/corpus_stats.json
    cs = json.load(open(os.path.join(ROOT, "work", "corpus", "stats.json"), encoding="utf-8"))
    books = cs["books"]
    corpus = {"books": len(books), "pdf_pages": sum(b["pdf_pages"] for b in books.values()),
              "ocr_pages": sum(b["ocr_pages"] for b in books.values()), "tokens": sum(b["tokens_content"] for b in books.values()),
              "chunks": sum(b["chunks"] for b in books.values()), "chunks_in_500_600": sum(b["chunks_in_500_600"] for b in books.values()),
              "tokenizer": cs["tokenizer"], "min_tok": cs["min_tok"], "max_tok": cs["max_tok"], "overlap_tok": cs["overlap_tok"],
              "source": "work/corpus/stats.json (02_preprocess.py)"}
    json.dump(corpus, open(os.path.join(R, "corpus_stats.json"), "w"), indent=1)
    put("nBooks", str(corpus["books"]))
    put("nPdfPages", thousands(corpus["pdf_pages"]))
    put("nTokens", thousands(corpus["tokens"]))
    put("nChunks", thousands(corpus["chunks"]))
    put("pctChunksInRange", pct(corpus["chunks_in_500_600"] / corpus["chunks"]))

    bs = J("benchmark_stats.json")
    fin = bs["final"]
    put("nDraft", thousands(bs["draft"]["test"]["n"] + bs["draft"]["dev"]["n"]))
    put("nDraftTest", thousands(bs["draft"]["test"]["n"]))
    put("nReject", str(bs["annotation_A_labels"]["reject"]))
    put("nEdit", str(bs["annotation_A_labels"]["edit"]))
    put("nTypeChanged", str(bs["type_changed_to_single"]))
    put("nPoolTest", thousands(bs["accepted_pool"]["test"]["n"]))
    put("nTest", thousands(fin["test"]["n"]))
    put("nDev", str(fin["dev"]["n"]))
    for t, key in (("single", "Single"), ("multi_intra", "Intra"), ("multi_cross", "Cross"), ("unanswerable", "Unans")):
        put(f"nTest{key}", str(fin["test"]["type"][t]))
    put("nCrossChapter", str(fin["test"]["cross_type"]["chapter"]))
    put("nCrossBook", str(fin["test"]["cross_type"]["book"]))
    put("minBookTest", str(min(fin["test"]["book"].values())))
    put("maxBookTest", str(max(fin["test"]["book"].values())))
    put("ansLenMedian", str(bs["descriptive"]["test"]["answer_len_median"]))
    put("sharedChunks", str(bs["evidence_chunks_shared_test_dev"]))

    ag = J("annotation_round1_agreement.json")
    put("nOverlapRoundOne", str(ag["n_overlap"]))
    put("kappaLabel", ag["label"]["kappa"], "{:.2f}")
    put("kappaType", ag["type_correct"]["kappa"], "{:.2f}")
    put("rawLabel", pct(ag["label"]["raw_agreement"]))
    put("rawEvidence", pct(ag["evidence_correct"]["raw_agreement"]))
    put("nDisagree", str(len(ag["disagreements"])))

    vq = J("viquad_stats.json")
    put("nViquad", thousands(vq["test"]["n"]))
    put("nViquadUnans", str(vq["test"]["unanswerable"]))
    put("pctViquadUnans", pct(vq["test"]["unanswerable_ratio"]))
    put("pctViquadSrcUnans", pct(vq["test"]["source_unanswerable_ratio"], 2))
    put("nViquadPassages", str(vq["test"]["corpus_chunks"]))

    es = J("dev", "embedding_selection.json")
    ch = J("configs", "embedding_choice.json")["chosen"]
    put("devRecallFiveEmb", es[ch]["ptit_dev"]["recall@5"])
    put("devRecallFiveBM", es["bm25"]["ptit_dev"]["recall@5"])
    emb_vals = sorted(es[m]["ptit_dev"]["recall@5"] for m in es if m != "bm25")
    put("devRecallFiveEmbMin", emb_vals[0])
    cfg = J("configs", "systems_config.json")
    put("kChunks", str(cfg["k"]))
    tk = J("dev", "tune_k.json")["results"]
    put("devFOneSthreeKeight", tk["8"]["f1_all"])
    put("devFOneSfive", J("dev", "tune_s5facts.json")["results"]["0"]["f1_all"])
    diag = J("dev", "s5_diagnosis.json")
    put("diagTextEight", diag["multi_hop:text_recall@8"]["mean"], "{:.2f}")
    put("diagTextFifty", diag["multi_hop:text_recall@50"]["mean"], "{:.2f}")
    put("diagGraphEight", diag["multi_hop:graph_recall@8"]["mean"], "{:.2f}")
    put("diagRecovered", pct(diag["multi_hop:missed_by_text8_found_in_graph20"]["mean"], 0))
    rt = J("dev", "router.json")
    put("routerCV", rt["cv_accuracy"], "{:.3f}")
    put("routerTestAcc", rt["ptit_test_route_accuracy"], "{:.2f}")
    put("nVqSingleRoute", str(rt["routes_test"]["viquad_test"]["single"]))

    met = {ds: J("metrics", f"{ds}.json") for ds in ("ptit_test", "viquad_test")}
    st = {ds: J("stats", f"{ds}.json")["S5"] for ds in ("ptit_test", "viquad_test")}
    mh = J("stats", "ptit_test_multihop.json")
    sta = {ds: J("stats", f"{ds}_ablations.json")["S5"]["qwen3.5-4b"] for ds in ("ptit_test", "viquad_test")}
    p = met["ptit_test"]
    q = lambda s, slm="qwen3.5-4b", ds="ptit_test": met[ds][s][slm]["overall"]
    for s in SYS:
        put(f"FOneAll{s}", q(s)["f1_all"])
        put(f"FOneAllVq{s}", q(s, ds="viquad_test")["f1_all"])
        put(f"RefU{s}", q(s)["refusal_unans"])
    put("RecallCrossSfour", p["S4"]["qwen3.5-4b"]["by_type"]["multi_cross"]["recall"], "{:.2f}")
    put("RecallCrossSfive", p["S5"]["qwen3.5-4b"]["by_type"]["multi_cross"]["recall"], "{:.2f}")
    # S5 vs S4 on PTIT across SLMs
    sig4 = [slm for slm in SLMS if st["ptit_test"][slm]["S4"]["f1_all"]["p_holm"] < 0.05 and st["ptit_test"][slm]["S4"]["f1_all"]["diff"] > 0]
    d4 = [st["ptit_test"][slm]["S4"]["f1_all"]["diff"] for slm in sig4]
    put("nSigVsSfour", str(len(sig4)))
    put("dVsSfourMin", min(d4))
    put("dVsSfourMax", max(d4))
    sig13 = [(slm, b) for slm in SLMS for b in ("S1", "S3") for ds in ("ptit_test", "viquad_test") if st[ds][slm][b]["f1_all"]["p_holm"] < 0.05]
    put("nSigVsSoneSthree", str(len(sig13)))
    put("nCompSoneSthree", str(len(SLMS) * 2 * 2))  # 5 SLMs x {S1, S3} x {PTIT, ViQuAD}
    put("FOneAllRagMin", min(q(s)["f1_all"] for s in ("S1", "S3", "S5")))
    put("FOneAllRagMax", max(q(s)["f1_all"] for s in ("S1", "S3", "S5")))
    mh_sig13 = [(slm, b) for slm in SLMS for b in ("S1", "S3") for sub in ("multi_hop", "multi_cross_book") if mh[sub][slm][b]["p_holm"] < 0.05]
    put("nSigMultiVsSoneSthree", str(len(mh_sig13)))
    put("nMultiHop", str(mh["multi_hop"]["qwen3.5-4b"]["S0"]["n"]))
    for ab, key in (("S5-novector", "Novector"), ("S5-norouting", "Norouting"), ("S5-noframe", "Noframe")):
        for ds, dk in (("ptit_test", ""), ("viquad_test", "Vq")):
            v = sta[ds][ab]["f1_all"]
            put(f"abl{key}{dk}", f"{v['diff']:+.3f}")
    # S5 vs S2 on ViQuAD, primary SLM, F1_ans
    v = st["viquad_test"]["qwen3.5-4b"]["S2"]["f1_ans"]
    put("vqSfiveSTwoAns", f"{v['diff']:+.3f}")
    put("vqSfiveSTwoAnsP", v["p_holm"], "{:.3f}")
    # refusal on ViQuAD
    ref_vq = [met["viquad_test"][s][slm]["overall"]["refusal_unans"] for s in SYS[1:] for slm in ("qwen3.5-4b", "gemma4-e4b", "qwen3-8b")]
    put("vqRefMin", min(ref_vq), "{:.2f}")
    put("vqRefMax", max(ref_vq), "{:.2f}")

    hl = J("halluc", "manual_halluc.json")
    for s in ("S0", "S2", "S4", "S5"):
        r_ = hl["rates"][s]
        put(f"hal{s}", pct(r_["halluc_rate_all"]))
        put(f"hal{s}Lo", pct(r_["halluc_rate_all_ci95"][0]))
        put(f"hal{s}Hi", pct(r_["halluc_rate_all_ci95"][1]))
    for b in ("S2", "S4"):
        put(f"halP{b}", hl["mcnemar_s5_vs"][b]["p_holm"], "{:.2f}")
    put("kappaHalFive", hl["agreement"]["kappa_5class"], "{:.2f}")
    put("kappaHalBin", hl["agreement"]["kappa_binary_halluc"], "{:.2f}")
    put("rhoJudge", hl["validation"]["spearman_manual_vs_judge"]["rho"], "{:.2f}")
    put("rhoFOne", hl["validation"]["spearman_manual_vs_f1"]["rho"], "{:.2f}")
    put("kappaRefusal", hl["validation"]["refusal_detector_vs_manual"]["kappa"], "{:.2f}")
    put("nHalQ", "150")
    put("nHalRows", str(hl["n_A"]))
    put("nHalB", str(hl["n_B"]))

    jd = J("judge", "summary.json")
    put("judgeSfive", jd["ptit_test"]["S5"]["qwen3.5-4b"]["faithfulness"])
    put("judgeSthree", jd["ptit_test"]["S3"]["qwen3.5-4b"]["faithfulness"])
    put("judgeSfour", jd["ptit_test"]["S4"]["qwen3.5-4b"]["faithfulness"])

    co = J("costs.json")
    off = co["offline"]["ptit"]
    put("hExtractSfive", off["s5_extraction_s"] / 3600, "{:.1f}")
    put("hBuildLightrag", off["lightrag_build_runs"][0]["wall_s"] / 3600, "{:.1f}")
    put("sBuildBM", off["bm25_build_s"], "{:.1f}")
    put("sBuildDense", off["dense_build_s"], "{:.1f}")
    put("nEntities", thousands(off["s5_graph"]["entities"]))
    put("nRelations", thousands(off["s5_graph"]["relations"]))
    put("nLrNodes", thousands(off["lightrag_graph"]["nodes"]))
    put("nLrEdges", thousands(off["lightrag_graph"]["edges"]))
    on = co["online"]["ptit_test"]
    for s in SYS:
        put(f"lat{s}", on[s]["qwen3.5-4b"]["total_mean_s"], "{:.2f}")
        put(f"latP{s}", on[s]["qwen3.5-4b"]["total_p95_s"], "{:.2f}")
    mem = co["memory"]["slm_ollama_gib"]
    put("memSlmMin", min(mem.values()), "{:.1f}")
    put("memSlmMax", max(mem.values()), "{:.1f}")
    rg = [v["gpu_peak_gib"] for k, v in co["memory"]["retrieval"].items() if k in SYS]
    put("memRetMax", max(rg), "{:.1f}")
    put("rssRet", max(v["max_rss_gib"] for v in co["memory"]["retrieval"].values()), "{:.1f}")

    # descriptive statistics of the S5 graph and the questions -> results/descriptive_stats.json (11_descriptive_stats.py)
    ds = J("descriptive_stats.json")
    g, lk, lx = ds["graph"], ds["gold_chunk_links"], ds["lexical"]
    put("entPerChunk", g["entities_per_chunk_mean"], "{:.1f}")
    put("relPerChunk", g["relations_per_chunk_mean"], "{:.1f}")
    put("pctEntOneChunk", pct(g["share_entities_one_chunk"]))
    put("pctEntMultiBook", pct(g["share_entities_multi_book"]))
    put("pctEntNoRel", pct(g["share_entities_no_relation"]))
    for key, name in (("all_multi_hop", "All"), ("multi_intra", "Intra"), ("multi_cross_chapter", "Chapter"), ("multi_cross_book", "Book")):
        put(f"pctLink{name}", pct(lk[key]["shared_entity"], 0))
        put(f"pctLinkHop{name}", pct(lk[key]["entity_or_one_relation"], 0))
    put("pctLexGold", pct(lx["coverage_gold_mean"], 0))
    put("pctLexRand", pct(lx["coverage_random_same_book_mean"], 0))
    put("contentSyl", lx["content_syllables_per_question_mean"], "{:.1f}")
    put("qLenMean", bs["descriptive"]["test"]["question_len_mean"], "{:.1f}")
    put("mhChunks", bs["descriptive"]["test"]["multi_hop_distinct_chunks_mean"], "{:.2f}")

    # Vietnamese-formatted copies for the Vietnamese abstract (decimal comma, thousands point)
    for k in ("nTest", "nViquad", "FOneAllSZero", "FOneAllRagMin", "FOneAllRagMax", "halSZero", "halSTwo", "halSFour"):
        M[k + "Vn"] = M[k].replace("{,}", "\\THOUSANDS").replace(".", "{,}").replace("\\THOUSANDS", ".")

    with open(os.path.join(PAPER, "numbers.tex"), "w") as f:
        f.write("% generated by paper/scripts/make_numbers.py from ./results -- do not edit by hand\n")
        for k, v in M.items():
            f.write(f"\\newcommand{{\\{k}}}{{{v}}}\n")

    # ---------- Table: F1_all, systems x SLMs ----------
    os.makedirs(os.path.join(PAPER, "tables"), exist_ok=True)
    SHORT = {"qwen3.5-4b": "Q3.5-4B", "gemma4-e4b": "G4-E4B", "llama3.2-3b": "L3.2-3B", "phogpt-4b": "Pho-4B", "qwen3-8b": "Q3-8B"}
    L = [r"\begin{table*}[!t]", r"\centering", r"\caption{F1$_\text{all}$ on the two test sets (1{,}000 questions each; an unanswerable question counts as correct only when the system abstains). "
         r"SLMs: Qwen3.5-4B, Gemma-4-E4B, Llama-3.2-3B, PhoGPT-4B-Chat, Qwen3-8B. "
         r"Best value per column in bold. $^\dagger$/$^\ddagger$: S5 is significantly better/worse than that baseline (paired bootstrap, 10{,}000 resamples, Holm-adjusted $p<0.05$).}",
         r"\label{tab:main}", r"\setlength{\tabcolsep}{3.5pt}", r"\footnotesize",
         r"\begin{tabular}{l" + "c" * 5 + "|" + "c" * 5 + "}", r"\hline",
         r" & \multicolumn{5}{c|}{PTIT-CourseQA test} & \multicolumn{5}{c}{UIT-ViQuAD2.0 subset} \\",
         "System & " + " & ".join(SHORT[s] for s in SLMS) + " & " + " & ".join(SHORT[s] for s in SLMS) + r" \\", r"\hline"]
    best = {(ds, slm): max(met[ds][s][slm]["overall"]["f1_all"] for s in SYS) for ds in met for slm in SLMS}
    for s in SYS:
        cells = []
        for ds in ("ptit_test", "viquad_test"):
            for slm in SLMS:
                x = met[ds][s][slm]["overall"]["f1_all"]
                c = f"{x:.3f}"
                if abs(x - best[(ds, slm)]) < 1e-9:
                    c = r"\textbf{" + c + "}"
                if s != "S5":
                    t = st[ds][slm][s]["f1_all"]
                    if t["p_holm"] < 0.05:
                        c += r"$^\dagger$" if t["diff"] > 0 else r"$^\ddagger$"
                cells.append(c)
        L.append(SYS_TEX[s] + " & " + " & ".join(cells) + r" \\")
    L += [r"\hline", r"\end{tabular}", r"\end{table*}"]
    open(os.path.join(PAPER, "tables", "table_main.tex"), "w").write("\n".join(L) + "\n")

    # ---------- Table: primary SLM detail on PTIT ----------
    L = [r"\begin{table}[!t]", r"\centering",
         r"\caption{Qwen3.5-4B on PTIT-CourseQA test: F1 on answerable questions, correct abstention on unanswerable questions (Abst.), "
         r"false abstention on answerable questions (FAbst.), gold-chunk recall of the context (R@8) and mean end-to-end latency on the GB10 (s).}",
         r"\label{tab:primary}", r"\setlength{\tabcolsep}{3.2pt}", r"\small", r"\begin{tabular}{lcccccc}", r"\hline",
         r"System & F1$_\text{ans}$ & F1$_\text{all}$ & Abst. & FAbst. & R@8 & Lat. \\", r"\hline"]
    for s in SYS:
        o = q(s)
        rc = "--" if o.get("recall") is None else f"{o['recall']:.3f}"
        L.append(f"{s} & {o['f1_ans']:.3f} & {o['f1_all']:.3f} & {o['refusal_unans']:.3f} & {o['false_refusal_ans']:.3f} & {rc} & {on[s]['qwen3.5-4b']['total_mean_s']:.2f}" + r" \\")
    L += [r"\hline", r"\end{tabular}", r"\end{table}"]
    open(os.path.join(PAPER, "tables", "table_primary.tex"), "w").write("\n".join(L) + "\n")
    print(len(M), "macros")


if __name__ == "__main__":
    main()
