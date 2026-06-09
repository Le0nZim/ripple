#!/usr/bin/env python3
"""
Supplementary Fig. 2 — resolution dependence of APP and disagreement (256x256).

  (a) Dataset-level APP vs. total annotations, after rescaling every dataset to a
      common 256x256 grid (diamond = manual reference at 100%, square = RIPPLE).
  (c) Annotator disagreement after the 256x256 rescale (box + CDF).

Both panels are recomputed from ``data/suppfig2_inputs.json`` (matched
ground-truth/RIPPLE coordinates and per-anchor points). Coordinates are scaled by
(256/H, 256/W) before APP and distances are evaluated.

Panel (b) — RIPPLE/Linear APP-vs-corrections at 256 scale — is replayed through
the optical-flow volumes (15+ GB) and is shipped in ``results/`` rather than
recomputed here.

    python reproduce_suppfig2.py
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
DATA = json.load(open(os.path.join(HERE, "data", "suppfig2_inputs.json")))
THR = DATA["thresholds"]
D = DATA["datasets"]

DS_ORDER = ["Neural Activity", "Pinned Down", "Freely", "Sperm", "Homeostasis"]
DS_COLORS = {"Neural Activity": "#88CCEE", "Pinned Down": "#44AA99",
             "Sperm": "#999933", "Freely": "#AA4499", "Homeostasis": "#BBBBBB"}
SHORT = {"Neural Activity": "Neural", "Pinned Down": "Pinned", "Freely": "Freely",
         "Sperm": "Sperm", "Homeostasis": "Homeo"}
TARGET = 256


def _arr(x):
    return np.array([[np.nan, np.nan] if v is None else v for v in x], float)


def app_256(name):
    d = D[name]
    h, w = d["native_hw"]
    gt = _arr(d["gt"]) * [TARGET / h, TARGET / w]
    pred = _arr(d["pred"]) * [TARGET / h, TARGET / w]
    valid = ~(np.isnan(gt).any(1) | np.isnan(pred).any(1))
    dist = np.sqrt(((gt[valid] - pred[valid]) ** 2).sum(1))
    return float(np.mean([(dist <= t).mean() * 100 for t in THR]))


# ─── panel (a): APP-256 vs annotations scatter ──────────────────────────────
app256 = {n: app_256(n) for n in DS_ORDER}
_diamond_s, _square_s = 20, 30
_dr, _sr = np.sqrt(_diamond_s) / 2, np.sqrt(_square_s) / 2
fig, ax = plt.subplots(figsize=(3.72, 2.28), layout="constrained")
max_x = 0
for name in DS_ORDER:
    d = D[name]; col = DS_COLORS[name]
    mx, my = d["total_gt_frames"], 100.0
    rx, ry = d["total_anchors"], app256[name]
    max_x = max(max_x, mx)
    ax.add_patch(FancyArrowPatch((mx, my), (rx, ry),
                 arrowstyle="-|>, head_length=4, head_width=3", color=col, lw=0.8,
                 zorder=2, connectionstyle="arc3,rad=-0.08",
                 shrinkA=_dr + 1, shrinkB=_sr + 1))
    ax.scatter(mx, my, s=_diamond_s, marker="D", color=col, edgecolors="white",
               linewidths=0.3, zorder=3)
    ax.scatter(rx, ry, s=_square_s, marker="s", color=col, edgecolors="white",
               linewidths=0.4, zorder=4)
ax.set_xlabel("Total Manual Annotations")
ax.set_ylabel("Avg APP (%) [256]")
ax.set_xlim(-50, max_x * 1.12)
ax.set_ylim(0, 108)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
fig.savefig(os.path.join(RESULTS, "panel_a_app_vs_annotations_256.svg"), bbox_inches="tight")
plt.close(fig)
print("  wrote results/panel_a_app_vs_annotations_256.svg")

# ─── panel (c): scaled disagreement box + CDF ───────────────────────────────
dists = {}
for name in DS_ORDER:
    d = D[name]
    h, w = d["native_hw"]
    sx, sy = TARGET / h, TARGET / w
    pts = np.asarray(d["anchor_points"], float)
    dists[name] = np.hypot((pts[:, 1] - pts[:, 3]) * sx, (pts[:, 2] - pts[:, 4]) * sy)

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
for i, (dd, c) in enumerate(zip(data, colors)):
    jitter = rng.uniform(-0.15, 0.15, size=len(dd))
    ax.scatter(np.full(len(dd), i) + jitter, dd, s=6, color=c, alpha=0.5,
               edgecolors="black", linewidths=0.3, zorder=3)
ax.set_xticks(range(len(names)))
ax.set_xticklabels([SHORT[n] for n in names], fontsize=9)
ax.set_ylabel("Disagreement (256 px)")
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
fig.savefig(os.path.join(RESULTS, "disagreement_box.svg"), bbox_inches="tight")
plt.close(fig)
print("  wrote results/disagreement_box.svg")

fig, ax = plt.subplots(figsize=(3.72, 2.28), layout="constrained")
for name in DS_ORDER:
    dd = np.sort(dists[name])
    if not len(dd):
        continue
    cdf = np.arange(1, len(dd) + 1) / len(dd)
    ax.step(dd, cdf, where="post", color=DS_COLORS[name], lw=1.5, label=name)
ax.set_xlabel("Euclidean Distance (256 px)")
ax.set_ylabel("Cumulative Fraction [256]")
ax.set_xlim(left=0)
ax.set_ylim(0, 1.02)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.legend(fontsize=8, loc="lower right", frameon=True, edgecolor="gray")
fig.savefig(os.path.join(RESULTS, "disagreement_cdf.svg"), bbox_inches="tight")
plt.close(fig)
print("  wrote results/disagreement_cdf.svg")

print("\nBefore (native) -> after (256) APP:")
for name in DS_ORDER:
    d = D[name]
    h, w = d["native_hw"]
    gt = _arr(d["gt"]); pred = _arr(d["pred"])
    valid = ~(np.isnan(gt).any(1) | np.isnan(pred).any(1))
    dn = np.sqrt(((gt[valid] - pred[valid]) ** 2).sum(1))
    before = float(np.mean([(dn <= t).mean() * 100 for t in THR]))
    print(f"  {name:16s} {before:6.2f} -> {app256[name]:6.2f}")
print("\nPanel (b) scaling SVGs require the optical-flow volumes; see README.")
