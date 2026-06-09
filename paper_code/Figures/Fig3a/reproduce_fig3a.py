#!/usr/bin/env python3
"""
Fig. 3a — annotation effort vs. accuracy.

Per dataset, a diamond marks the exhaustive manual effort (total annotations at
100% APP by definition) and a square marks RIPPLE (its correction count vs. its
average point precision); an arrow links the two. The annotated variant adds the
APP value and the effort-reduction factor.

Reads ``data/fig3a_values.json`` (from ``generate_data.py``) and writes the
panels to ``results/`` with the published styling.

    python reproduce_fig3a.py
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
VALUES = json.load(open(os.path.join(HERE, "data", "fig3a_values.json")))["datasets"]

DS_ORDER = ["Neural Activity", "Pinned Down", "Freely", "Sperm", "Homeostasis"]
DS_COLORS = {"Neural Activity": "#88CCEE", "Pinned Down": "#44AA99",
             "Sperm": "#999933", "Freely": "#AA4499", "Homeostasis": "#BBBBBB"}
LABEL_OFFSETS = {"Neural Activity": (10, 5), "Pinned Down": (10, -10),
                 "Sperm": (8, 8), "Freely": (10, 5), "Homeostasis": (10, -8)}
_diamond_s, _square_s = 20, 30
_diamond_r = np.sqrt(_diamond_s) / 2
_square_r = np.sqrt(_square_s) / 2


def _save(fig, stem):
    fig.savefig(os.path.join(RESULTS, f"{stem}.svg"), bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote results/{stem}.svg")


def panel_b(minimal=False):
    fig, ax = plt.subplots(figsize=(3.72, 2.28), layout="constrained")
    max_x = 0
    for name in DS_ORDER:
        d = VALUES[name]
        col = DS_COLORS[name]
        mx, my = d["total_gt_frames"], 100.0
        rx, ry = d["total_anchors"], d["avg_app"]
        max_x = max(max_x, mx)
        if not minimal:
            ax.add_patch(FancyArrowPatch(
                (mx, my), (rx, ry),
                arrowstyle="-|>, head_length=4, head_width=3",
                color=col, lw=0.8, zorder=2,
                connectionstyle="arc3,rad=-0.08",
                shrinkA=_diamond_r + 1, shrinkB=_square_r + 1))
        ax.scatter(mx, my, s=_diamond_s, marker="D", color=col,
                   edgecolors="white", linewidths=0.3, zorder=3)
        ax.scatter(rx, ry, s=_square_s, marker="s", color=col,
                   edgecolors="white", linewidths=0.4, zorder=4)
    ax.set_xlabel("Total Manual Annotations")
    ax.set_ylabel("Avg APP (%)")
    ax.set_xlim(-50, max_x * 1.12)
    ax.set_ylim(0, 108)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    _save(fig, "panel_B_minimal" if minimal else "panel_B")


def panel_b_annotated():
    fig, ax = plt.subplots(figsize=(3.72, 2.28), layout="constrained")
    max_x = 0
    for name in DS_ORDER:
        d = VALUES[name]
        col = DS_COLORS[name]
        mx, my = d["total_gt_frames"], 100.0
        rx, ry = d["total_anchors"], d["avg_app"]
        max_x = max(max_x, mx)
        ax.add_patch(FancyArrowPatch(
            (mx, my), (rx, ry),
            arrowstyle="-|>, head_length=4, head_width=3",
            color=col, lw=0.8, zorder=2,
            connectionstyle="arc3,rad=-0.08",
            shrinkA=_diamond_r + 1, shrinkB=_square_r + 1))
        ax.scatter(mx, my, s=_diamond_s, marker="D", color=col,
                   edgecolors="white", linewidths=0.3, zorder=3)
        ax.scatter(rx, ry, s=_square_s, marker="s", color=col,
                   edgecolors="white", linewidths=0.4, zorder=4)
        lbl_dx, lbl_dy = LABEL_OFFSETS[name]
        va = "bottom" if lbl_dy >= 0 else "top"
        ax.annotate(f"{ry:.1f}%", (rx, ry), fontsize=9, color=col, ha="left",
                    xytext=(lbl_dx, lbl_dy), textcoords="offset points", va=va,
                    arrowprops=dict(arrowstyle="-", color=col, alpha=0.3, lw=0.4))
        ratio = mx / rx if rx > 0 else 0
        dy_off = -4 if name == "Pinned Down" else 4
        ax.text((mx + rx) / 2, (my + ry) / 2 + dy_off, f"{ratio:.1f}\u00d7",
                fontsize=9, color=col, alpha=0.8, ha="center", va="bottom",
                style="italic")
    ax.set_xlabel("Total Manual Annotations")
    ax.set_ylabel("Avg APP (%)")
    ax.set_xlim(-50, max_x * 1.12)
    ax.set_ylim(0, 108)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    _save(fig, "panel_B_annotated")


panel_b(minimal=False)
panel_b(minimal=True)
panel_b_annotated()
print("\nPanels written to results/ (panel_B_annotated.svg is the published "
      "Fig. 3a).")
