"""GATE 4: build results/SUMMARY.md from the result files only (every number is read from ./results or ./work/benchmark).
Also computes the RQ1 subset tests (S5 vs baselines on multi-hop and cross-book questions, paired bootstrap, Holm)
into results/stats/ptit_test_multihop.json.
"""
import collections
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from rag import data  # noqa: E402

R = os.path.join(data.ROOT, "results")
SLMS = ["qwen3.5-4b", "gemma4-e4b", "llama3.2-3b", "phogpt-4b", "qwen3-8b"]
SYS = ["S0", "S1", "S2", "S3", "S4", "S5"]
ABL = ["S5-noframe", "S5-norouting", "S5-novector", "S5-depth2", "S5-depth3"]
J = lambda *p: json.load(open(os.path.join(R, *p)))
f3 = lambda x: "–" if x is None else f"{x:.3f}"


def pq(ds, system, slm):
    p = os.path.join(R, "metrics", "per_question", f"{ds}__{system}__{slm}.jsonl")
    return {json.loads(l)["id"]: json.loads(l) for l in open(p)}


def boot(a, b, seed=42, B=10000):
    d = np.asarray(a) - np.asarray(b)
    rng = np.random.default_rng(seed)
    m = d[rng.integers(0, len(d), size=(B, len(d)))].mean(axis=1)
    p = min(1.0, 2 * min((m <= 0).mean(), (m >= 0).mean()))
    return {"diff": round(float(d.mean()), 4), "ci95": [round(float(np.percentile(m, 2.5)), 4), round(float(np.percentile(m, 97.5)), 4)],
            "p": float(max(p, 1 / B)), "n": len(d)}


def holm(ps):
    order = sorted(range(len(ps)), key=lambda i: ps[i])
    adj, run = [0.0] * len(ps), 0.0
    for r, i in enumerate(order):
        run = max(run, min(1.0, (len(ps) - r) * ps[i]))
        adj[i] = run
    return adj


def multihop_tests():
    qs = {q["id"]: q for q in data.questions("ptit", "test")}
    subsets = {"multi_hop": [i for i, q in qs.items() if q["type"] in ("multi_intra", "multi_cross")],
               "multi_cross_book": [i for i, q in qs.items() if q["type"] == "multi_cross" and q["cross_type"] == "book"]}
    out = {}
    for name, ids in subsets.items():
        for slm in SLMS:
            s5 = pq("ptit_test", "S5", slm)
            fam = {b: boot([s5[i]["f1"] for i in ids], [pq("ptit_test", b, slm)[i]["f1"] for i in ids]) for b in SYS[:-1]}
            for b, a in zip(fam, holm([v["p"] for v in fam.values()])):
                fam[b]["p_holm"] = round(a, 6)
            out.setdefault(name, {})[slm] = fam
    json.dump(out, open(os.path.join(R, "stats", "ptit_test_multihop.json"), "w"), indent=1)
    return out


def sig(v):
    p = v["p_holm"]
    return f"{v['diff']:+.3f} [{v['ci95'][0]:+.3f}, {v['ci95'][1]:+.3f}]{'*' if p < 0.05 else ''}"


def main():
    L = []
    w = L.append
    bs, agr = J("benchmark_stats.json"), J("annotation_round1_agreement.json")
    vq, cfg = J("viquad_stats.json"), J("configs", "systems_config.json")
    met = {ds: J("metrics", f"{ds}.json") for ds in ("ptit_test", "viquad_test")}
    st = {ds: J("stats", f"{ds}.json")["S5"] for ds in ("ptit_test", "viquad_test")}
    sta = {ds: J("stats", f"{ds}_ablations.json")["S5"]["qwen3.5-4b"] for ds in ("ptit_test", "viquad_test")}
    mh = multihop_tests()
    jd = J("judge", "summary.json")
    hl = J("halluc", "manual_halluc.json")
    co = J("costs.json")
    dev = {s: J("dev", f"tune_{s}.json") for s in ("k", "s5grid", "s5facts", "s4mode")}

    w("# SUMMARY — KG-RAG cho SLM tiếng Việt trên học liệu đại học (GATE 4)\n")
    w("Mọi số liệu trong tài liệu này được sinh tự động bằng `work/scripts/08_summary.py`, từ các file trong `results/` và `work/benchmark/`. "
      "Ký hiệu `*` nghĩa là p < 0,05 sau hiệu chỉnh Holm. Δ = S5 − baseline.\n")

    w("## 1. Dữ liệu\n")
    fin = bs["final"]
    w(f"- **PTIT test:** {fin['test']['n']} câu. Phân bố loại: {fin['test']['type']}; câu liên chương/liên sách: {fin['test']['cross_type']}.")
    w(f"- **PTIT dev:** {fin['dev']['n']} câu. Test và dev không dùng chung chunk bằng chứng nào ({bs['evidence_chunks_shared_test_dev']} chunk chung).")
    w(f"- **Kho truy xuất:** {co['offline']['ptit']['chunks']} chunk, 11 giáo trình.")
    w(f"- **ViQuAD test:** {vq['test']['n']} câu lấy từ split validation, {vq['test']['unanswerable']} câu không có đáp án. Kho: {vq['test']['corpus_chunks']} đoạn văn.")
    a = agr
    w(f"- **Độ đồng thuận gán nhãn vòng 1** (A so với B gán mù, {a['n_overlap']} câu):")
    w(f"  - κ(label) = {a['label']['kappa']:.2f}, tỷ lệ trùng thô {a['label']['raw_agreement']:.1%};")
    w(f"  - κ(type_correct) = {a['type_correct']['kappa']:.2f};")
    w(f"  - evidence_correct: tỷ lệ trùng thô {a['evidence_correct']['raw_agreement']:.1%} (κ không xác định có nghĩa, vì phân bố nhãn quá lệch).")
    w("  - Có 12 câu bất đồng; tác giả phân xử cả 12 theo A. Nguồn gốc các bản B trước: xem DATASHEET.\n")

    w("## 2. Cấu hình chốt trên dev\n")
    w(f"- **k = {cfg['k']} chunk cho mọi hệ thống.** Kết quả S3 trên dev: " +
      "; ".join(f"k={k}: F1_all {v['f1_all']:.3f}" for k, v in dev['k']['results'].items()) + ".")
    w(f"- **Embedding:** {cfg['embedding']}.")
    w(f"- **S4:** mode {cfg['S4']['mode']}.")
    w(f"- **S5:** trộn `{cfg['S5']['fusion']}` (chèn {cfg['S5']['graph_slots']} chunk từ đồ thị), độ sâu {cfg['S5']['depth']}, n_facts = {cfg['S5']['n_facts']}.")
    w(f"- **S5 bản đầu (RRF ngang trọng số) kém S3 trên dev** (xem `results/dev/v1_superseded/`). Thiết kế cách trộn được sửa dựa trên chẩn đoán trên dev. "
      f"S5 cuối cùng ngang S3 trên dev: F1_all {dev['s5facts']['results']['0']['f1_all']:.3f} so với {dev['k']['results']['8']['f1_all']:.3f}.\n")

    for ds, title in (("ptit_test", "PTIT test"), ("viquad_test", "ViQuAD test (1.000 câu, validation)")):
        w(f"## 3{'a' if ds == 'ptit_test' else 'b'}. Kết quả chính — {title}\n")
        w("F1_all tính theo kiểu SQuAD2: câu không có đáp án được tính đúng khi hệ thống từ chối. F1_ans chỉ tính trên câu có đáp án. "
          "RefU là tỷ lệ từ chối đúng trên câu không có đáp án. FalseRef là tỷ lệ từ chối sai trên câu có đáp án. Recall@8 là recall truy xuất.\n")
        w("| SLM | Hệ thống | F1_all | F1_ans | EM_ans | RefU | FalseRef | Recall@8 |")
        w("|---|---|---:|---:|---:|---:|---:|---:|")
        for slm in SLMS:
            for s in SYS:
                o = met[ds][s][slm]["overall"]
                w(f"| {slm} | {s} | {f3(o['f1_all'])} | {f3(o['f1_ans'])} | {f3(o['em_ans'])} | {f3(o['refusal_unans'])} | {f3(o['false_refusal_ans'])} | {f3(o.get('recall'))} |")
        w("")
        w(f"**Kiểm định S5 so với baseline** ({title}; ΔF1_all [KTC 95%], bootstrap ghép cặp 10.000 lần; McNemar trên tỷ lệ từ chối đúng):\n")
        w("| SLM | vs S0 | vs S1 | vs S2 | vs S3 | vs S4 |")
        w("|---|---|---|---|---|---|")
        for slm in SLMS:
            w(f"| {slm} | " + " | ".join(sig(st[ds][slm][b]["f1_all"]) for b in SYS[:-1]) + " |")
        w("")
        w("McNemar trên từ chối đúng (số câu chỉ S5 từ chối đúng / số câu chỉ baseline từ chối đúng, p_Holm):\n")
        w("| SLM | vs S0 | vs S1 | vs S2 | vs S3 | vs S4 |")
        w("|---|---|---|---|---|---|")
        for slm in SLMS:
            w(f"| {slm} | " + " | ".join(f"{st[ds][slm][b]['halluc_unans']['s5_only']}/{st[ds][slm][b]['halluc_unans']['baseline_only']} (p={st[ds][slm][b]['halluc_unans']['p_holm']:.2g})" for b in SYS[:-1]) + " |")
        w("")

    w("## 4. RQ1 — câu đa bước và liên sách (PTIT test, F1_ans)\n")
    w("| SLM | Hệ thống | single | multi_intra | multi_cross | Recall multi_cross |")
    w("|---|---|---:|---:|---:|---:|")
    for slm in ("qwen3.5-4b", "qwen3-8b", "gemma4-e4b"):
        for s in SYS:
            bt = met["ptit_test"][s][slm]["by_type"]
            w(f"| {slm} | {s} | {f3(bt['single']['f1_ans'])} | {f3(bt['multi_intra']['f1_ans'])} | {f3(bt['multi_cross']['f1_ans'])} | {f3(bt['multi_cross'].get('recall'))} |")
    w("")
    for name, title in (("multi_hop", "multi_intra + multi_cross"), ("multi_cross_book", "chỉ câu liên sách")):
        n = mh[name]["qwen3.5-4b"]["S0"]["n"]
        w(f"**ΔF1 S5 − baseline trên tập con {title}** (n = {n}):\n")
        w("| SLM | vs S0 | vs S1 | vs S2 | vs S3 | vs S4 |")
        w("|---|---|---|---|---|---|")
        for slm in SLMS:
            w(f"| {slm} | " + " | ".join(sig(mh[name][slm][b]) for b in SYS[:-1]) + " |")
        w("")

    w("## 5. Ablation của S5 (qwen3.5-4b)\n")
    w("| Biến thể | PTIT F1_all | Δ so với S5 [KTC 95%] | ViQuAD F1_all | Δ so với S5 [KTC 95%] |")
    w("|---|---:|---|---:|---|")
    w(f"| S5 (độ sâu {cfg['S5']['depth']}) | {f3(met['ptit_test']['S5']['qwen3.5-4b']['overall']['f1_all'])} | – | {f3(met['viquad_test']['S5']['qwen3.5-4b']['overall']['f1_all'])} | – |")
    for a_ in ABL:
        w(f"| {a_} | {f3(met['ptit_test'][a_]['qwen3.5-4b']['overall']['f1_all'])} | {sig(sta['ptit_test'][a_]['f1_all'])} | "
          f"{f3(met['viquad_test'][a_]['qwen3.5-4b']['overall']['f1_all'])} | {sig(sta['viquad_test'][a_]['f1_all'])} |")
    w("\nΔ ở đây là S5 − biến thể; Δ dương nghĩa là thành phần bị bỏ đi có ích.\n")

    w("## 6. RQ2 — ảo giác\n")
    w("### 6.1 Chấm tự động bằng judge (qwen3.6), mẫu 300 câu mỗi tập test\n")
    w("Faithfulness được tính trên các câu trả lời không phải từ chối (supported = 1, partial = 0,5, unsupported = 0).\n")
    for ds in ("ptit_test", "viquad_test"):
        w(f"| {ds} | " + " | ".join(SYS[1:]) + " |")
        w("|---|" + "---:|" * 5)
        for slm in SLMS:
            w(f"| {slm} | " + " | ".join(f3(jd.get(ds, {}).get(s, {}).get(slm, {}).get("faithfulness")) for s in SYS[1:]) + " |")
        w("")
    w("### 6.2 Gán nhãn thủ công (qwen3.5-4b, 150 câu PTIT test × 4 hệ thống)\n")
    ag = hl["agreement"]
    w(f"Độ đồng thuận A–B trên 120 dòng: κ 5 lớp = {ag['kappa_5class']:.2f}, κ nhị phân (ảo giác hay không) = {ag['kappa_binary_halluc']:.2f}.\n")
    w("| Hệ thống | faithful | partial | hallucinated | correct_refusal | wrong_refusal | Tỷ lệ ảo giác [KTC 95%] | Tỷ lệ ảo giác trên câu đã trả lời |")
    w("|---|---:|---:|---:|---:|---:|---|---:|")
    for s in ("S0", "S2", "S4", "S5"):
        r = hl["rates"][s]
        lab = r["labels"]
        w(f"| {s} | {lab['faithful']} | {lab['partially_hallucinated']} | {lab['hallucinated']} | {lab['correct_refusal']} | {lab['wrong_refusal']} | "
          f"{r['halluc_rate_all']:.3f} [{r['halluc_rate_all_ci95'][0]:.3f}, {r['halluc_rate_all_ci95'][1]:.3f}] | {f3(r['halluc_rate_answered'])} |")
    mc = hl["mcnemar_s5_vs"]
    w("\nMcNemar trên ảo giác thủ công: " + "; ".join(f"S5 vs {b}: {v['s5_better_only']}/{v['baseline_better_only']}, p_Holm {'< 0.001' if v['p_holm'] < 0.001 else '= ' + format(v['p_holm'], '.3g')}" for b, v in mc.items()) + ".")
    v = hl["validation"]
    w(f"\n**Kiểm chứng bộ chấm tự động:**")
    w(f"- Spearman giữa nhãn thủ công và judge: ρ = {v['spearman_manual_vs_judge']['rho']:.2f} (n = {v['spearman_manual_vs_judge']['n']}).")
    w(f"- Spearman giữa nhãn thủ công và F1: ρ = {v['spearman_manual_vs_f1']['rho']:.2f} (n = {v['spearman_manual_vs_f1']['n']}).")
    w(f"- Bộ phát hiện từ chối khớp nhãn thủ công {v['refusal_detector_vs_manual']['agreement']:.1%} (κ = {v['refusal_detector_vs_manual']['kappa']:.2f}).\n")

    w("## 7. RQ3 — chi phí trên phần cứng hạn chế\n")
    w("| Kho | Chunk | BM25 (s) | Dense (s) | Trích xuất S5 (s) | Đồ thị S5 (thực thể / quan hệ) | LightRAG dựng (s) | Đồ thị LightRAG (nút / cạnh) |")
    w("|---|---:|---:|---:|---:|---|---:|---|")
    for c, o in co["offline"].items():
        lr = o["lightrag_build_runs"][0]["wall_s"]
        w(f"| {c} | {o['chunks']} | {o['bm25_build_s']} | {o['dense_build_s']} | {o['s5_extraction_s']:.0f} | {o['s5_graph']['entities']} / {o['s5_graph']['relations']} | {lr:.0f} | {o['lightrag_graph']['nodes']} / {o['lightrag_graph']['edges']} |")
    w("\nĐộ trễ mỗi câu (qwen3.5-4b; truy xuất + sinh; các lượt test chạy tuần tự, một mình trên GPU GB10):\n")
    w("| Hệ thống | PTIT trung bình (s) | PTIT p95 (s) | ViQuAD trung bình (s) | ViQuAD p95 (s) | Prompt token PTIT |")
    w("|---|---:|---:|---:|---:|---:|")
    for s in SYS:
        p_, v_ = co["online"]["ptit_test"][s]["qwen3.5-4b"], co["online"]["viquad_test"][s]["qwen3.5-4b"]
        w(f"| {s} | {p_['total_mean_s']:.2f} | {p_['total_p95_s']:.2f} | {v_['total_mean_s']:.2f} | {v_['total_p95_s']:.2f} | {p_['prompt_tokens_mean']:.0f} |")
    mem = co["memory"]
    w("\n**Bộ nhớ:**")
    w("- SLM (Ollama, gồm KV cache ở num_ctx 8192), GiB: " + ", ".join(f"{k} {v:.2f}" for k, v in mem["slm_ollama_gib"].items()) + ".")
    w("- Đỉnh bộ nhớ GPU của tiến trình truy xuất, GiB: " + ", ".join(f"{k} {v['gpu_peak_gib']:.2f}" for k, v in mem["retrieval"].items() if k in SYS) + ".")
    w("- RSS của tiến trình truy xuất khoảng " + f"{max(v['max_rss_gib'] for v in mem['retrieval'].values()):.1f} GiB (Python, torch và chỉ mục).\n")
    L.append(open(os.path.join(data.WORK, "summary_narrative.md"), encoding="utf-8").read())
    open(os.path.join(R, "SUMMARY.md"), "w", encoding="utf-8").write("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
