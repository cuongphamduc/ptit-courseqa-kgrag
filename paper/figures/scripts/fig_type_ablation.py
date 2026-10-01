"""Figure 2 — (a) F1_ans per question type and abstention on unanswerable questions, S0-S5, Qwen3.5-4B, PTIT test,
with 95% bootstrap CIs over questions (10,000 resamples, seed 42); (b) S5 ablations: F1_all(S5) - F1_all(variant)
with 95% paired-bootstrap CIs from results/stats/*_ablations.json.
Sources: results/metrics/per_question/ptit_test__<S>__qwen3.5-4b.jsonl, results/stats/{ptit,viquad}_test_ablations.json
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
GROUPS = [("single", "Single-hop"), ("multi_intra", "Multi-hop\nwithin a book"), ("multi_cross", "Cross-chapter/\ncross-book"),
          ("unanswerable", "Unanswerable\n(abstention)")]
ABL = [("S5-novector", "no text branch"), ("S5-norouting", "no routing"), ("S5-noframe", "no structural frame"),
       ("S5-depth2", "depth 2"), ("S5-depth3", "depth 3")]


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
    rows = []
    for s in style.SYS:
        pq = style.jsonl("metrics", "per_question", f"ptit_test__{s}__{SLM}.jsonl")
        for t, _ in GROUPS:
            vals = [r["refusal"] if t == "unanswerable" else r["f1"] for r in pq if r["type"] == t]
            lo, hi = boot_ci(vals, rng)
            rows.append({"system": s, "type": t, "value": float(np.mean(vals)), "ci_lo": lo, "ci_hi": hi, "n": len(vals)})
    with open(os.path.join(style.DATA, "fig2a_type_scores.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    abl = {ds: style.load("stats", f"{ds}_ablations.json")["S5"][SLM] for ds in ("ptit_test", "viquad_test")}
    arows = [{"variant": v, "dataset": ds, "diff": abl[ds][v]["f1_all"]["diff"], "ci_lo": abl[ds][v]["f1_all"]["ci95"][0],
              "ci_hi": abl[ds][v]["f1_all"]["ci95"][1], "p_holm": abl[ds][v]["f1_all"]["p_holm"]} for v, _ in ABL for ds in abl]
    with open(os.path.join(style.DATA, "fig2b_ablation.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(arows[0]))
        w.writeheader()
        w.writerows(arows)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(style.COL_W, 3.75), gridspec_kw={"height_ratios": [1.25, 1.0], "hspace": 0.62})
    n = len(style.SYS)
    width = 0.8 / n
    x0 = np.arange(len(GROUPS))
    for i, s in enumerate(style.SYS):
        sub = [r for r in rows if r["system"] == s]
        xs = x0 - 0.4 + width * (i + 0.5)
        vals = [r["value"] for r in sub]
        err = np.array([[r["value"] - r["ci_lo"] for r in sub], [r["ci_hi"] - r["value"] for r in sub]])
        ax1.bar(xs, vals, width * 0.92, color=style.COLORS[s], edgecolor="none", label=style.SYS_LABEL[s], zorder=2)
        ax1.errorbar(xs, vals, yerr=err, fmt="none", ecolor="#444444", elinewidth=0.5, capsize=0, zorder=3)
    ax1.set_xticks(x0, [g for _, g in GROUPS])
    ax1.set_ylim(0, 1.0)
    ax1.set_ylabel("F1$_\\mathrm{ans}$  /  abstention rate")
    ax1.axvline(2.5, color="#999999", lw=0.5, ls=(0, (2, 2)))
    ax1.yaxis.grid(True, color="#E6E6E6", lw=0.5, zorder=0)
    ax1.legend(ncol=3, loc="upper center", bbox_to_anchor=(0.5, 1.30), handlelength=1.0, columnspacing=0.9, handletextpad=0.4)
    style.panel_label(ax1, "a", x=-0.13, y=1.02)

    y = np.arange(len(ABL))[::-1]
    for k, (ds, lab, mk, off) in enumerate((("ptit_test", "PTIT-CourseQA", "o", 0.14), ("viquad_test", "ViQuAD2.0 subset", "s", -0.14))):
        d = [r for r in arows if r["dataset"] == ds]
        xs = [r["diff"] for r in d]
        err = np.array([[r["diff"] - r["ci_lo"] for r in d], [r["ci_hi"] - r["diff"] for r in d]])
        col = style.COLORS["S5"] if ds == "ptit_test" else "#8C8C8C"
        ax2.errorbar(xs, y + off, xerr=err, fmt=mk, ms=3.2, color=col, ecolor=col, elinewidth=0.8, capsize=1.5, label=lab)
    ax2.axvline(0, color="#555555", lw=0.6)
    ax2.set_yticks(y, [a for _, a in ABL])
    ax2.set_xlabel("F1$_\\mathrm{all}$(S5) $-$ F1$_\\mathrm{all}$(variant)   (> 0: removed part helps)")
    ax2.legend(loc="lower right", handletextpad=0.3, borderaxespad=0.2)
    style.panel_label(ax2, "b", x=-0.13, y=1.02)
    export(fig, "fig_type_ablation", multi_panel=True)


if __name__ == "__main__":
    main()
