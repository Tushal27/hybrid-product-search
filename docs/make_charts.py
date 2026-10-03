"""
Draw the two README charts from the saved result files (nothing is typed in by hand here).

    python docs/make_charts.py        ->  docs/quality.png, docs/load.png

Inputs (all in product_search/results/):
    quality_test_497.json                         nDCG@10 per pipeline with 95% bootstrap CIs
    loadtest_baseline.json                        the first API under load (one lock, no overload protection)
    loadtest_v2_nocache.json, loadtest_v2_high.json   the hardened API under load (all-distinct queries)

Colors: the first two slots of a palette checked for colour-blind separation and contrast (blue #2a78d6 /
orange #eb6834 on the #fcfcfb surface). The charts carry their own light surface so they read the same on
GitHub's light and dark themes.
"""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

ROOT = Path(__file__).resolve().parent.parent
RES = ROOT / "product_search" / "results"
OUT = Path(__file__).resolve().parent

SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e6e4dd"
BLUE, ORANGE, GRAY = "#2a78d6", "#eb6834", "#a8a69d"

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 10, "axes.edgecolor": GRID, "axes.labelcolor": INK2,
    "xtick.color": INK2, "ytick.color": INK2, "text.color": INK, "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
})


def style(ax):
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.spines["left"].set_color(GRID)
    ax.spines["bottom"].set_color(GRID)
    ax.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


# --------------------------------------------------------------------------- 1. search quality
def quality_chart():
    data = json.loads((RES / "quality_test_497.json").read_text(encoding="utf-8"))
    rows = data["rows"]
    best = max(rows, key=lambda r: r["ndcg"])["key"]
    dense = next(r for r in rows if r["key"] == "dense")

    fig, ax = plt.subplots(figsize=(9, 3.9), dpi=200)
    style(ax)
    ax.grid(axis="y", visible=False)
    for i, r in enumerate(rows):
        y = len(rows) - 1 - i
        emphasized = r["key"] == best
        color = BLUE if emphasized else GRAY
        ax.plot(r["ci"], [y, y], color=color, linewidth=2, solid_capstyle="round", zorder=2)
        ax.plot(r["ndcg"], y, "o", markersize=9 if emphasized else 8, color=color, markeredgecolor=SURFACE, markeredgewidth=2, zorder=3)
        ax.text(r["ci"][1] + 0.003, y, f"{r['ndcg']:.3f}", va="center", ha="left", fontsize=10,
                color=INK if emphasized else INK2, fontweight="bold" if emphasized else "normal")
    ax.axvline(dense["ndcg"], color=GRAY, linewidth=1, linestyle=(0, (3, 3)), zorder=1)
    ax.text(dense["ndcg"], len(rows) - 0.35, "dense-only", ha="center", va="bottom", fontsize=8.5, color=INK2)

    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([r["label"] for r in reversed(rows)], fontsize=10)
    for lbl, r in zip(ax.get_yticklabels(), reversed(rows)):
        if r["key"] == best:
            lbl.set_fontweight("bold")
            lbl.set_color(INK)
    ax.set_xlim(0.10, 0.225)
    ax.set_ylim(-0.6, len(rows) - 0.1)
    ax.set_xlabel("nDCG@10  (higher is better;  dot = mean,  line = 95% confidence interval)")
    ax.set_title(f"Search quality on {data['n_queries']} unseen shopper queries", loc="left", fontsize=12, fontweight="bold", pad=22)
    fig.text(0.075, 0.015, "Intervals overlap heavily: only tuned hybrid + rerank vs dense-only is a reliable difference (+0.016, 95% CI +0.006 to +0.025).",
             fontsize=8.5, color=INK2, ha="left", va="bottom")
    fig.subplots_adjust(left=0.24, right=0.97, top=0.84, bottom=0.2)
    fig.savefig(OUT / "quality.png", facecolor=SURFACE)
    plt.close(fig)


# --------------------------------------------------------------------------- 2. behaviour under load
def load_chart():
    old = json.loads((RES / "loadtest_baseline.json").read_text(encoding="utf-8"))
    new = json.loads((RES / "loadtest_v2_nocache.json").read_text(encoding="utf-8")) + json.loads((RES / "loadtest_v2_high.json").read_text(encoding="utf-8"))

    def series(rows, key):
        return [r["rate"] for r in rows if key(r) is not None], [key(r) for r in rows if key(r) is not None]

    def served(r):
        return 100 * r["ok"] / r["sent"]

    fig, (a, b) = plt.subplots(1, 2, figsize=(9.6, 4.0), dpi=200)
    ticks = [1, 2, 3, 5, 8, 12, 16, 24, 32, 48]
    for ax in (a, b):
        style(ax)
        ax.set_xscale("log", base=2)
        ax.set_xticks(ticks)
        ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
        ax.minorticks_off()
        ax.set_xlim(0.85, 60)
        ax.set_xlabel("requests per second sent to the server")

    for rows, color, name in ((old, ORANGE, "first version"), (new, BLUE, "hardened version")):
        x, y = series(rows, served)
        a.plot(x, y, "-o", color=color, linewidth=2, markersize=6, markeredgecolor=SURFACE, markeredgewidth=1.5, label=name)
        x2, y2 = series(rows, lambda r: r["p50"])
        b.plot(x2, y2, "-o", color=color, linewidth=2, markersize=6, markeredgecolor=SURFACE, markeredgewidth=1.5, label=name)

    a.set_ylim(-3, 106)
    a.set_ylabel("% of requests answered successfully")
    a.set_title("Requests answered", loc="left", fontsize=11, fontweight="bold")
    a.annotate("first version: 70% fail at 3/s,\nall fail at 5/s (timeouts)", xy=(5, 0), xytext=(8.5, 10), fontsize=8.5, color=INK2,
               arrowprops=dict(arrowstyle="-", color=GRAY, linewidth=0.8))
    a.annotate("hardened: extra load is refused\ninstantly (HTTP 503) by design", xy=(48, 54), xytext=(7.5, 40), fontsize=8.5, color=INK2,
               arrowprops=dict(arrowstyle="-", color=GRAY, linewidth=0.8))
    b.set_ylim(0, 5600)
    b.set_ylabel("median latency of answered requests (ms)")
    b.set_title("Median latency", loc="left", fontsize=11, fontweight="bold")
    b.annotate("first version: 5 s at 3/s\n(no answers at all at 5/s)", xy=(3, 5038), xytext=(3.5, 4450), fontsize=8.5, color=INK2)
    b.annotate("hardened: stays under 1 s", xy=(24, 824), xytext=(4.2, 1750), fontsize=8.5, color=INK2,
               arrowprops=dict(arrowstyle="-", color=GRAY, linewidth=0.8))
    b.legend(loc="center right", bbox_to_anchor=(1.0, 0.5), frameon=False, fontsize=9.5)

    fig.suptitle("Behaviour under load: first API version vs hardened version (same laptop, same queries)", x=0.01, ha="left",
                 fontsize=12, fontweight="bold")
    fig.text(0.01, 0.01, "Measured with product_search/loadtest.py on one laptop; the load generator shares its CPU. Above ~16 req/s most answers skip the reranker "
             "(lower ranking quality).", fontsize=8, color=INK2, ha="left", va="bottom")
    fig.subplots_adjust(left=0.07, right=0.985, top=0.84, bottom=0.2, wspace=0.22)
    fig.savefig(OUT / "load.png", facecolor=SURFACE)
    plt.close(fig)


if __name__ == "__main__":
    quality_chart()
    load_chart()
    print("wrote", OUT / "quality.png", "and", OUT / "load.png")
