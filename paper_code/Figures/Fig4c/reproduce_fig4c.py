#!/usr/bin/env python3
"""
Fig. 4c — coordinate-replacement (disagreement-elimination) analysis.

How much of each dataset's residual error comes from the comparison annotator's
coordinate disagreement (Fig. 4b) rather than from interpolation? The RIPPLE
anchor coordinates are replaced with the exhaustive ground-truth coordinates at
the same frames, the flow-blend interpolation is re-run, and the APP is
recomputed (AFTER) and compared with the original RIPPLE APP (BEFORE).

Reads ``data/fig4c_results.json`` and writes the panel and bar charts to
``results/`` with the published styling.

    python reproduce_fig4c.py
"""
import json
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch

matplotlib.rcParams["font.family"] = "sans-serif"
matplotlib.rcParams["font.sans-serif"] = ["Arial", "Helvetica", "DejaVu Sans"]
matplotlib.rcParams["font.size"] = 11
matplotlib.rcParams["figure.facecolor"] = "white"
matplotlib.rcParams["axes.facecolor"] = "white"
matplotlib.rcParams["savefig.facecolor"] = "white"

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(HERE, "results")
os.makedirs(RESULTS, exist_ok=True)
RES = json.load(open(os.path.join(HERE, "data", "fig4c_results.json")))["datasets"]

DS_ORDER = ["Neural Activity", "Pinned Down", "Freely", "Sperm", "Homeostasis"]
DS_COLORS = {"Neural Activity": "#88CCEE", "Pinned Down": "#44AA99",
             "Sperm": "#999933", "Freely": "#AA4499", "Homeostasis": "#BBBBBB"}
SHORT = {"Neural Activity": "Neural", "Pinned Down": "Pinned", "Freely": "Freely",
         "Sperm": "Sperm", "Homeostasis": "Homeo"}


def plot_panel():
    _diamond_s, _tri_s = 20, 30
    _diamond_r, _tri_r = np.sqrt(_diamond_s) / 2, np.sqrt(_tri_s) / 2
    fig, ax = plt.subplots(figsize=(3.72, 2.28), layout="constrained")
    max_x = 0
    for name in DS_ORDER:
        r = RES[name]
        col = DS_COLORS[name]
        mx, my, rx, ry = r["gt_frames"], 100.0, r["anchor_count"], r["after"]
        max_x = max(max_x, mx)
        ax.add_patch(FancyArrowPatch(
            (mx, my), (rx, ry), arrowstyle="-|>, head_length=4, head_width=3",
            color=col, lw=0.8, zorder=2, connectionstyle="arc3,rad=-0.08",
            shrinkA=_diamond_r + 1, shrinkB=_tri_r + 1))
        ax.scatter(mx, my, s=_diamond_s, marker="D", color=col,
                   edgecolors="white", linewidths=0.3, zorder=3)
        ax.scatter(rx, ry, s=_tri_s, marker="^", color=col,
                   edgecolors="white", linewidths=0.4, zorder=5)
    ax.set_xlabel("Total Manual Annotations")
    ax.set_ylabel("Avg APP (%)")
    ax.set_xlim(-50, max_x * 1.12)
    ax.set_ylim(0, 108)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.savefig(os.path.join(RESULTS, "disagreement_elimination_panel.svg"),
                bbox_inches="tight")
    plt.close(fig)
    print("  wrote results/disagreement_elimination_panel.svg")


def plot_bars(key, stem):
    names = list(DS_ORDER)
    vals = [RES[n][key] for n in names]
    colors = [DS_COLORS[n] for n in names]
    x = np.arange(len(names))
    fig, ax = plt.subplots(figsize=(3.72, 2.28), layout="constrained")
    for i, (v, col) in enumerate(zip(vals, colors)):
        ax.bar(x[i], v, 0.55, color=col, edgecolor=col, linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels([SHORT[n] for n in names], fontsize=7)
    ax.set_ylabel("Avg APP (%)", fontsize=7)
    ax.tick_params(axis="both", labelsize=7)
    ax.set_ylim(0, 115)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.savefig(os.path.join(RESULTS, f"{stem}.svg"), bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote results/{stem}.svg")


plot_panel()
plot_bars("before", "disagreement_elimination_bars_before")
plot_bars("after", "disagreement_elimination_bars_after")

print()
for name in DS_ORDER:
    r = RES[name]
    print(f"  {name:16s} before={r['before']:6.2f}  after={r['after']:6.2f}  "
          f"delta={r['delta']:+6.2f}")
