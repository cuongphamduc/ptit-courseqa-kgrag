"""Shared figure style for the JSTIC manuscript (IEEEtran two-column: one column = 3.5 in).

Palette: neutral/light hues for baselines, one saturated accent for the proposed system (S5).
Every figure reads its numbers from ./results and writes its source data to paper/figures/data/.
"""
import json
import os
import sys

import matplotlib as mpl

HERE = os.path.dirname(os.path.abspath(__file__))
FIG = os.path.dirname(HERE)
ROOT = os.path.dirname(os.path.dirname(FIG))
RES = os.path.join(ROOT, "results")
DATA = os.path.join(FIG, "data")
os.makedirs(DATA, exist_ok=True)
sys.path.insert(0, os.path.join(ROOT, "nature-skills", "skills", "nature-figure", "scripts"))

COL_W = 3.5          # inches, IEEEtran column width
SYS = ["S0", "S1", "S2", "S3", "S4", "S5"]
SYS_LABEL = {"S0": "S0 no retrieval", "S1": "S1 BM25", "S2": "S2 dense", "S3": "S3 hybrid", "S4": "S4 LightRAG", "S5": "S5 (ours)"}
COLORS = {"S0": "#C9C9C9", "S1": "#A9BFDD", "S2": "#9ED3C6", "S3": "#EBCB93", "S4": "#D8A7A7", "S5": "#1F4E79"}
SLMS = ["qwen3.5-4b", "gemma4-e4b", "llama3.2-3b", "phogpt-4b", "qwen3-8b"]
SLM_LABEL = {"qwen3.5-4b": "Qwen3.5-4B", "gemma4-e4b": "Gemma-4-E4B", "llama3.2-3b": "Llama-3.2-3B", "phogpt-4b": "PhoGPT-4B",
             "qwen3-8b": "Qwen3-8B"}


def setup():  # kept for reference; figure scripts set rcParams inline
    mpl.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Liberation Sans", "Arial", "Helvetica", "DejaVu Sans"],
        "pdf.fonttype": 42, "svg.fonttype": "none",
        "font.size": 7, "axes.titlesize": 7, "axes.labelsize": 7, "xtick.labelsize": 6.5, "ytick.labelsize": 6.5,
        "legend.fontsize": 6.5, "axes.spines.right": False, "axes.spines.top": False, "axes.linewidth": 0.6,
        "xtick.major.width": 0.6, "ytick.major.width": 0.6, "xtick.major.size": 2.5, "ytick.major.size": 2.5,
        "legend.frameon": False, "lines.linewidth": 0.9,
    })


def load(*p):
    return json.load(open(os.path.join(RES, *p), encoding="utf-8"))


def jsonl(*p):
    return [json.loads(l) for l in open(os.path.join(RES, *p), encoding="utf-8")]


def panel_label(ax, s, x=-0.14, y=1.04):
    ax.text(x, y, s, transform=ax.transAxes, fontsize=8, fontweight="bold", va="bottom", ha="left")


def save(fig, name, multi_panel):
    out = os.path.join(FIG, name)
    if multi_panel:
        from audit_panel_alignment import require_matplotlib_panel_alignment
        fig.canvas.draw()
        require_matplotlib_panel_alignment(fig, json_out=out + ".alignment.json", overlay_svg=None, tolerance_pt=1.5,
                                           gutter_tolerance_pt=1.5, strict=True)
    fig.savefig(out + ".pdf", bbox_inches="tight", pad_inches=0.02)
    fig.savefig(out + ".svg", bbox_inches="tight", pad_inches=0.02)
