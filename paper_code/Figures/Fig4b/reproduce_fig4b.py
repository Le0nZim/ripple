#!/usr/bin/env python3
"""
Fig. 4b — annotator-disagreement distribution and CDF.

The disagreement at each corrected frame is the Euclidean distance (native px)
between the exhaustive manual ground-truth point and the comparison annotator's
RIPPLE correction. This draws the per-dataset box+strip plot and the cumulative
distribution, with the published styling.

Reads ``data/fig4b_inputs.json`` and writes ``results/disagreement_box.svg`` and
``results/disagreement_cdf.svg``.

    python reproduce_fig4b.py
"""
import json
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

matplotlib.rcParams["font.family"] = "sans-serif"
matplotlib.rcParams["font.sans-serif"] = ["Arial", "Helvetica", "DejaVu Sans"]
matplotlib.rcParams["font.size"] = 11
matplotlib.rcParams["figure.facecolor"] = "white"
matplotlib.rcParams["axes.facecolor"] = "white"
matplotlib.rcParams["savefig.facecolor"] = "white"

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(HERE, "results")
os.makedirs(RESULTS, exist_ok=True)
DATA = json.load(open(os.path.join(HERE, "data", "fig4b_inputs.json")))["datasets"]

DS_ORDER = ["Neural Activity", "Pinned Down", "Freely", "Sperm", "Homeostasis"]
DS_COLORS = {"Neural Activity": "#88CCEE", "Pinned Down": "#44AA99",
             "Freely": "#AA4499", "Sperm": "#999933", "Homeostasis": "#BBBBBB"}
SHORT = {"Neural Activity": "Neural", "Pinned Down": "Pinned", "Freely": "Freely",
         "Sperm": "Sperm", "Homeostasis": "Homeo"}

dists = {}
for name in DS_ORDER:
    pts = np.asarray(DATA[name]["anchor_points"], float)
    dists[name] = np.hypot(pts[:, 1] - pts[:, 3], pts[:, 2] - pts[:, 4])

# ─── box + strip ────────────────────────────────────────────────────────────
names = [n for n in DS_ORDER if len(dists[n])]
data = [dists[n] for n in names]
colors = [DS_COLORS[n] for n in names]
fig, ax = plt.subplots(figsize=(3.72, 2.28), layout="constrained")
bp = ax.boxplot(data, positions=range(len(names)), widths=0.5, patch_artist=True,
                showfliers=False, medianprops=dict(color="black", lw=1.2),
                whiskerprops=dict(color="gray"), capprops=dict(color="gray"))
for patch, c in zip(bp["boxes"], colors):
    patch.set_facecolor(c); patch.set_alpha(0.45)
rng = np.random.default_rng(42)
for i, (d, c) in enumerate(zip(data, colors)):
    jitter = rng.uniform(-0.15, 0.15, size=len(d))
    ax.scatter(np.full(len(d), i) + jitter, d, s=6, color=c, alpha=0.5,
               edgecolors="black", linewidths=0.3, zorder=3)
ax.set_xticks(range(len(names)))
ax.set_xticklabels([SHORT[n] for n in names], fontsize=9)
ax.set_ylabel("Disagreement (px)")
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
fig.savefig(os.path.join(RESULTS, "disagreement_box.svg"), bbox_inches="tight")
plt.close(fig)
print("  wrote results/disagreement_box.svg")

# ─── CDF ────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(3.72, 2.28), layout="constrained")
for name in DS_ORDER:
    d = np.sort(dists[name])
    if not len(d):
        continue
    cdf = np.arange(1, len(d) + 1) / len(d)
    ax.step(d, cdf, where="post", color=DS_COLORS[name], lw=1.5, label=name)
ax.set_xlabel("Euclidean Distance (px)")
ax.set_ylabel("Cumulative Fraction")
ax.set_xlim(left=0)
ax.set_ylim(0, 1.02)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.legend(fontsize=8, loc="lower right", frameon=True, edgecolor="gray")
fig.savefig(os.path.join(RESULTS, "disagreement_cdf.svg"), bbox_inches="tight")
plt.close(fig)
print("  wrote results/disagreement_cdf.svg")

# ─── box + strip, log y-axis (paper panel) ──────────────────────────────────
data_log = [np.maximum(dists[n], 0.1) for n in names]
fig, ax = plt.subplots(figsize=(3.72, 2.28), layout="constrained")
bp = ax.boxplot(data_log, positions=range(len(names)), widths=0.5, patch_artist=True,
                showfliers=False, medianprops=dict(color="black", lw=1.2),
                whiskerprops=dict(color="gray"), capprops=dict(color="gray"))
for patch, c in zip(bp["boxes"], colors):
    patch.set_facecolor(c); patch.set_alpha(0.45)
rng = np.random.default_rng(42)
for i, (d, c) in enumerate(zip(data_log, colors)):
    jitter = rng.uniform(-0.15, 0.15, size=len(d))
    ax.scatter(np.full(len(d), i) + jitter, d, s=6, color=c, alpha=0.5,
               edgecolors="black", linewidths=0.3, zorder=3)
ax.set_yscale("log")
ax.set_xticks(range(len(names)))
ax.set_xticklabels([SHORT[n] for n in names], fontsize=9)
ax.set_ylabel("Disagreement (px)")
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
fig.savefig(os.path.join(RESULTS, "disagreement_box_log.svg"), bbox_inches="tight")
plt.close(fig)
print("  wrote results/disagreement_box_log.svg")

# ─── CDF, log x-axis (paper panel) ──────────────────────────────────────────
fig, ax = plt.subplots(figsize=(3.72, 2.28), layout="constrained")
for name in DS_ORDER:
    d = np.sort(dists[name])
    if not len(d):
        continue
    d = np.where(d == 0, 0.05, d)  # shift exact zeros so log scale works
    cdf = np.arange(1, len(d) + 1) / len(d)
    ax.step(d, cdf, where="post", color=DS_COLORS[name], lw=1.5, label=name)
ax.set_xscale("log")
ax.set_xlabel("Disagreement (px)")
ax.set_ylabel("Cumulative Fraction")
ax.set_ylim(0, 1.02)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
fig.savefig(os.path.join(RESULTS, "disagreement_cdf_log.svg"), bbox_inches="tight")
plt.close(fig)
print("  wrote results/disagreement_cdf_log.svg")

print()
for name in DS_ORDER:
    d = dists[name]
    print(f"  {name:16s} n={len(d):4d}  median={np.median(d):6.2f}px  "
          f"mean={np.mean(d):6.2f}px")
