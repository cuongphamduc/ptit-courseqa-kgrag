"""Figure 3 — (a) hallucination rate over all answers (abstentions count as not hallucinated), Qwen3.5-4B, PTIT test:
manual labels (partially_hallucinated or hallucinated; 150 questions x S0/S2/S4/S5) and the automatic judge
(partial or unsupported; 300-question judge sample x S1-S5), 95% bootstrap CIs over questions (10,000, seed 42);
(b) quality-latency trade-off on PTIT test: mean end-to-end latency vs F1_all, all SLMs (primary SLM highlighted).
Sources: results/halluc/manual_labels_decoded.jsonl, results/judge/ptit_test/*.jsonl, results/costs.json,
results/metrics/ptit_test.json
"""
import csv
import os

import os

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

import style
from audit_panel_alignment import require_matplotlib_panel_alignment  # noqa: E402  (path set by style)

SLM = "qwen3.5-4b"
REFUSE = {"correct_refusal", "wrong_refusal"}
HAL = {"partially_hallucinated", "hallucinated"}


def boot_ci(x, rng, b=10000):
    x = np.asarray(x, float)
    m = x[rng.integers(0, len(x), size=(b, len(x)))].mean(axis=1)
    return float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


def export(fig, name, multi_panel):
    """final width = one IEEEtran column (88.9 mm); vector PDF/SVG for the manuscript, 600-dpi PNG for preview"""
    out = os.path.join(style.FIG, name)
    if multi_panel:
        fig.canvas.draw()
        require_matplotlib_panel_alignment(fig, json_out=out + ".alignment.json", overlay_svg=out + ".alignment.svg",
                                           tolerance_pt=1.5, gutter_tolerance_pt=1.5, strict=True)
    fig.savefig(out + ".pdf", bbox_inches="tight", pad_inches=0.02)
    fig.savefig(out + ".svg", bbox_inches="tight", pad_inches=0.02)
    fig.savefig(out + ".png", dpi=600, bbox_inches="tight", pad_inches=0.02)


def main():
    mpl.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Liberation Sans", "Arial", "Helvetica", "DejaVu Sans"],
        "pdf.fonttype": 42,        # editable TrueType text in PDF
        "svg.fonttype": "none",    # editable text in SVG
        "font.size": 7, "axes.labelsize": 7.5, "axes.titlesize": 7, "xtick.labelsize": 6.5, "ytick.labelsize": 6.5,
        "legend.fontsize": 6.5, "axes.spines.right": False, "axes.spines.top": False, "axes.linewidth": 0.6,
        "xtick.major.width": 0.6, "ytick.major.width": 0.6, "xtick.major.size": 2.5, "ytick.major.size": 2.5,
        "legend.frameon": False,
    })
    rng = np.random.default_rng(42)
    man = style.jsonl("halluc", "manual_labels_decoded.jsonl")
    rows = []
    mh = style.load("halluc", "manual_halluc.json")["rates"]  # rate and CI exactly as reported in the text
    for s in ("S0", "S2", "S4", "S5"):
        v = [r["label"] in HAL for r in man if r["system"] == s]
        assert abs(float(np.mean(v)) - mh[s]["halluc_rate_all"]) < 1e-3
        lo, hi = mh[s]["halluc_rate_all_ci95"]
        rows.append({"source": "manual", "system": s, "rate": float(np.mean(v)), "ci_lo": lo, "ci_hi": hi, "n": len(v)})
    for s in ("S1", "S2", "S3", "S4", "S5"):
        j = [r for r in style.jsonl("judge", "ptit_test", f"{s}__{SLM}.jsonl") if r["refusal"] or r.get("label")]  # drop judge parse failures
        v = [(not r["refusal"]) and r["label"] != "supported" for r in j]
        lo, hi = boot_ci(v, rng)
        rows.append({"source": "judge", "system": s, "rate": float(np.mean(v)), "ci_lo": lo, "ci_hi": hi, "n": len(v)})
    with open(os.path.join(style.DATA, "fig3a_hallucination.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    met, co = style.load("metrics", "ptit_test.json"), style.load("costs.json")
    crow = [{"system": s, "slm": m, "latency_mean_s": co["online"]["ptit_test"][s][m]["total_mean_s"],
             "latency_p95_s": co["online"]["ptit_test"][s][m]["total_p95_s"], "f1_all": met[s][m]["overall"]["f1_all"]}
            for s in style.SYS for m in style.SLMS]
    with open(os.path.join(style.DATA, "fig3b_quality_latency.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(crow[0]))
        w.writeheader()
        w.writerows(crow)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(style.COL_W, 3.9), gridspec_kw={"height_ratios": [1.0, 1.15], "hspace": 0.55})
    order = style.SYS
    x = np.arange(len(order))
    for src, off, mk, face in (("manual", -0.17, "o", True), ("judge", 0.17, "D", False)):
        for i, s in enumerate(order):
            r = next((r for r in rows if r["source"] == src and r["system"] == s), None)
            if r is None:
                continue
            col = style.COLORS[s] if s != "S0" else "#8C8C8C"
            ax1.errorbar(x[i] + off, r["rate"], yerr=[[r["rate"] - r["ci_lo"]], [r["ci_hi"] - r["rate"]]], fmt=mk, ms=3.6,
                         mfc=col if face else "white", mec=col if s != "S5" else style.COLORS["S5"], ecolor=col, elinewidth=0.8, capsize=1.5,
                         color=col, zorder=3)
    ax1.plot([], [], "o", color="#555555", ms=3.6, label="manual labels (150 questions)")
    ax1.plot([], [], "D", mfc="white", mec="#555555", ms=3.4, label="LLM judge (300 questions)")
    ax1.set_xticks(x, [style.SYS_LABEL[s].replace(" ", "\n", 1) for s in order])
    ax1.set_ylim(0, 0.9)
    ax1.set_ylabel("Share of answers that\nare (partly) hallucinated")
    ax1.legend(loc="upper right", handletextpad=0.3)
    style.panel_label(ax1, "a", x=-0.16, y=1.02)

    markers = {"qwen3.5-4b": "o", "gemma4-e4b": "s", "llama3.2-3b": "^", "phogpt-4b": "v", "qwen3-8b": "D"}
    for m in style.SLMS:
        pts = [r for r in crow if r["slm"] == m]
        primary = m == SLM
        for r in pts:
            ax2.scatter(r["latency_mean_s"], r["f1_all"], s=16 if primary else 9, marker=markers[m], color=style.COLORS[r["system"]],
                        edgecolor="#333333" if primary else "none", linewidth=0.4, alpha=1.0 if primary else 0.75, zorder=3 if primary else 2)
    prim = {r["system"]: r for r in crow if r["slm"] == SLM}
    for sy, (tx, ty, ha) in {"S0": (0.95, 0.31, "left"), "S4": (3.62, 0.62, "left"), "S5": (2.75, 0.69, "left")}.items():
        r = prim[sy]
        ax2.annotate(sy, (r["latency_mean_s"], r["f1_all"]), xytext=(tx, ty), fontsize=6.5, ha=ha, va="center",
                     fontweight="bold" if sy == "S5" else "normal", arrowprops=dict(arrowstyle="-", lw=0.4, color="#666666", shrinkA=1, shrinkB=2.5))
    c = [prim[x] for x in ("S1", "S2", "S3")]
    ax2.annotate("S1, S2, S3", (np.mean([r["latency_mean_s"] for r in c]), np.mean([r["f1_all"] for r in c])), xytext=(0.95, 0.69), fontsize=6.5,
                 ha="left", va="center", arrowprops=dict(arrowstyle="-", lw=0.4, color="#666666", shrinkA=1, shrinkB=3))
    for m in style.SLMS:
        ax2.scatter([], [], marker=markers[m], s=12, color="#777777", label=style.SLM_LABEL[m])
    ax2.set_xlabel("Mean end-to-end latency per question on GB10 (s)")
    ax2.set_ylabel("F1$_\\mathrm{all}$ (PTIT-CourseQA)")
    ax2.set_ylim(0.0, 0.74)
    ax2.set_xlim(0.3, max(r["latency_mean_s"] for r in crow) + 0.3)
    ax2.legend(ncol=3, loc="lower right", handletextpad=0.2, columnspacing=0.7, borderaxespad=0.25, fontsize=6)
    style.panel_label(ax2, "b", x=-0.16, y=1.02)
    export(fig, "fig_halluc_cost", multi_panel=True)


if __name__ == "__main__":
    main()
