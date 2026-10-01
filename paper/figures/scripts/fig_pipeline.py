"""Figure 1 — schematic of S5 (structure-framed KG-RAG). Configuration values are read from
results/configs/systems_config.json and results/corpus_stats.json so the drawing matches the frozen system."""
import os

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

import style

EDGE = "#4D4D4D"
FILL_OFF = "#F2F2F2"
FILL_ON = "#E8EEF5"
FILL_KEY = "#1F4E79"
FS = 5.5


def box(ax, x0, y0, x1, y1, text, fill, color="black", bold=False, fs=FS):
    ax.add_patch(FancyBboxPatch((x0, y0), x1 - x0, y1 - y0, boxstyle="round,pad=0,rounding_size=1.5", fc=fill, ec=EDGE, lw=0.6))
    ax.text((x0 + x1) / 2, (y0 + y1) / 2, text, ha="center", va="center", fontsize=fs, color=color,
            fontweight="bold" if bold else "normal", linespacing=1.2)


def arrow(ax, p, q, ls="-"):
    ax.add_patch(FancyArrowPatch(p, q, arrowstyle="-|>", mutation_scale=6, lw=0.7, color=EDGE, linestyle=ls, shrinkA=0, shrinkB=0))


def export(fig, name, multi_panel):
    """final width = one IEEEtran column (88.9 mm); vector PDF/SVG for the manuscript, 600-dpi PNG for preview"""
    out = os.path.join(style.FIG, name)
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
    cfg = style.load("configs", "systems_config.json")
    cs = style.load("corpus_stats.json")
    k, m, d = cfg["k"], cfg["S5"]["graph_slots"], cfg["S5"]["depth"]
    fig, ax = plt.subplots(figsize=(style.COL_W, 2.75))
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 76)
    ax.axis("off")
    # offline indexing
    ax.text(0, 75, "Offline indexing", fontsize=6.6, fontweight="bold", va="top")
    box(ax, 0, 46, 13, 60, f"{cs['books']}\ncourse\nPDFs", FILL_OFF)
    box(ax, 16, 46, 34, 60, f"{cs['chunks']:,}\nchunks of\n{cs['min_tok']}–{cs['max_tok']} tok.", FILL_OFF)
    box(ax, 38, 58, 76, 69, "Structural frame\n(book › chapter › section)", FILL_OFF)
    box(ax, 38, 45, 76, 56, "Entities and relations\nextracted by the SLM", FILL_OFF)
    box(ax, 38, 33, 76, 42, "Text index (BM25 + dense)", FILL_OFF)
    box(ax, 80, 45, 100, 69, "Knowledge\ngraph", FILL_KEY, color="white", bold=True, fs=6.0)
    arrow(ax, (13, 53), (16, 53))
    arrow(ax, (34, 57), (38, 63.5))
    arrow(ax, (34, 53), (38, 50.5))
    arrow(ax, (34, 49), (38, 37.5))
    arrow(ax, (76, 63.5), (80, 63.5))
    arrow(ax, (76, 50.5), (80, 50.5))
    # online answering
    ax.text(0, 29, "Online answering", fontsize=6.6, fontweight="bold", va="top")
    box(ax, 0, 5, 13, 20, "Question", FILL_ON)
    box(ax, 16, 5, 32, 20, "Router:\nsingle or\nmulti-hop", FILL_ON)
    box(ax, 36, 13.5, 68, 23.5, f"multi-hop: text top-{k - m} +\n{m} graph chunk ({d}-hop)", FILL_ON)
    box(ax, 36, 1, 68, 10.5, f"single-hop: text top-{k}", FILL_ON)
    box(ax, 72, 3, 100, 22, "SLM answers from\nchunks with source\nheaders, or abstains", FILL_ON)
    arrow(ax, (13, 12.5), (16, 12.5))
    arrow(ax, (32, 15), (36, 18.5))
    arrow(ax, (32, 10), (36, 6))
    arrow(ax, (68, 18.5), (72, 15))
    arrow(ax, (68, 6), (72, 9))
    arrow(ax, (57, 33), (52, 23.5), ls=(0, (2, 1.5)))
    arrow(ax, (88, 45), (64, 23.5), ls=(0, (2, 1.5)))
    export(fig, "fig_pipeline", multi_panel=False)


if __name__ == "__main__":
    main()
