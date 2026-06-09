#!/usr/bin/env python3
"""
Supplementary Fig. 1 — comparison of interpolation strategies (accuracy & cost).

  (a) APP vs. number of corrections for four interpolation strategies (Linear,
      RIPPLE flow-blend, Corridor-DP, TAP-Vid original) across the five datasets.
  (b) Runtime summary at k = 25: per-dataset build-time bars, time-vs-pixels
      scaling, and a speed-up heatmap relative to TAP-Vid original.

Everything is computed from the sparse-correction scaling outputs in ``data/``
(per track: the APP curve and the wall-clock rebuild-time curve vs. correction
count). Panels are written to ``results/`` with the published styling.

    python reproduce_suppfig1.py
"""
import json
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(HERE, "results")
os.makedirs(RESULTS, exist_ok=True)

DS = ["neural", "pinned", "freely", "sperm", "homeostasis"]
DS_COLORS = {"neural": "#88CCEE", "pinned": "#44AA99", "freely": "#AA4499",
             "sperm": "#999933", "homeostasis": "#BBBBBB"}

# (a) the four strategies and their scaling-panel y-axis labels
STRATEGIES = {
    "linear_interp":   "Linear Interpolation\nAvg APP (%)",
    "flow_blend":      "RIPPLE Interpolation\nAvg APP (%)",
    "corridor_dp":     "Corridor DP\nAvg APP (%)",
    "tapvid_original": "TAPVid Original\nAvg APP (%)",
}

# (b) display names, palette, and per-dataset pixel counts (image metadata)
B_LABEL = {"linear_interp": "Linear Interp.", "flow_blend": "Flow-Blend",
           "corridor_dp": "Corridor DP", "tapvid_original": "TAP-Vid Original"}
B_COLORS = {"Linear Interp.": "#2ca02c", "Flow-Blend": "#1f77b4",
            "Corridor DP": "#444444", "TAP-Vid Original": "#d62728"}
B_DS = ["Pinned", "Neural", "Sperm", "Freely", "Homeostasis"]
B_KEY = {"Pinned": "pinned", "Neural": "neural", "Sperm": "sperm",
         "Freely": "freely", "Homeostasis": "homeostasis"}
PIXELS = {"Pinned": 10_000, "Neural": 360_000, "Sperm": 1_253_376,
          "Freely": 1_310_720, "Homeostasis": 31_719_424}


def load(stem):
    return json.load(open(os.path.join(HERE, "data", f"optimal_corrections_{stem}.json")))


def t_at_k(tr, target=25, tol=5):
    tc = tr.get("timing", {}).get("time_curve", {})
    if not tc:
        return None
    ks = sorted(int(k) for k in tc)
    b = min(ks, key=lambda k: abs(k - target))
    return float(tc[str(b)]) if abs(b - target) <= tol else None


def curves(data):
    out = {}
    for ds in DS:
        if ds not in data:
            continue
        tcs = []
        for e in data[ds].values():
            kv = {int(k): v * 100 for k, v in e["app_curve"].items()}
            sks = sorted(kv)
            tcs.append((max(sks), sks, [kv[k] for k in sks]))
        aks = sorted({k for _, sks, _ in tcs for k in sks})
        vals = {k: [100.0 if k > mk else float(np.interp(k, sks, sa))
                    for mk, sks, sa in tcs] for k in aks}
        out[ds] = (aks, [np.mean(vals[k]) for k in aks],
                   [np.min(vals[k]) for k in aks], [np.max(vals[k]) for k in aks])
    return out


def mean_ms_per_ds(data):
    """Per-dataset mean rebuild time at k=25 (ms), in B_DS order."""
    out = {}
    for label in B_DS:
        ds = B_KEY[label]
        ts = [t * 1000 for t in (t_at_k(tr) for tr in data.get(ds, {}).values())
              if t is not None]
        out[label] = float(np.mean(ts)) if ts else float("nan")
    return out


# ════════════════════════════════════════════════════════════════════════════
# (a) APP-vs-correction-count curves, one panel per strategy
# ════════════════════════════════════════════════════════════════════════════
matplotlib.rcParams.update({
    "font.family": "sans-serif", "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size": 11, "figure.facecolor": "white", "axes.facecolor": "white",
    "savefig.facecolor": "white", "axes.grid": False,
})
all_timing = {}
for stem, ylabel in STRATEGIES.items():
    data = load(stem)
    all_timing[stem] = mean_ms_per_ds(data)
    c = curves(data)
    fig, ax = plt.subplots(figsize=(3.72, 2.28), layout="constrained")
    for ds in DS:
        if ds not in c:
            continue
        ks, mn, lo, hi = c[ds]
        ax.plot(ks, mn, color=DS_COLORS[ds], lw=1.5)
        ax.fill_between(ks, lo, hi, color=DS_COLORS[ds], alpha=0.12)
    ax.axvline(25, color="#888888", ls="--", lw=0.8, zorder=0)
    ax.set_xlabel("Number of corrections")
    ax.set_ylabel(ylabel, multialignment="center")
    ax.set_xlim(left=0)
    ax.set_ylim(0, 108)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.savefig(os.path.join(RESULTS, f"scaling_{stem}.svg"), bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote results/scaling_{stem}.svg")

# per-method per-dataset timing matrix (ms), B_DS order
METHODS_MS = {B_LABEL[s]: [all_timing[s][d] for d in B_DS] for s in STRATEGIES}


def fmt_ms(x, _=None):
    if x >= 1000:
        return f"{x/1000:.0f}s" if x >= 10000 else f"{x/1000:.1f}s"
    if x >= 1:
        return f"{x:.0f}ms"
    return f"{x:g}ms"


# ════════════════════════════════════════════════════════════════════════════
# (b1) grouped log bars
# ════════════════════════════════════════════════════════════════════════════
matplotlib.rcParams.update({"font.size": 11, "axes.spines.top": False,
                            "axes.spines.right": False, "axes.grid": True,
                            "grid.alpha": 0.25, "grid.linestyle": "--"})
fig, ax = plt.subplots(figsize=(10, 5.2))
x = np.arange(len(B_DS))
w = 0.2
offsets = np.linspace(-1.5 * w, 1.5 * w, len(METHODS_MS))
for off, (m, vals) in zip(offsets, METHODS_MS.items()):
    bars = ax.bar(x + off, vals, w, label=m, color=B_COLORS[m],
                  edgecolor="white", linewidth=0.5)
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v * 1.15, fmt_ms(v),
                ha="center", va="bottom", fontsize=7, color=B_COLORS[m])
ax.set_yscale("log")
ax.set_xticks(x); ax.set_xticklabels(B_DS)
ax.set_ylabel("wall-clock time per build (ms, log)")
ax.set_title("Per-track build time at k = 25 anchors")
ax.yaxis.set_major_formatter(FuncFormatter(fmt_ms))
ax.set_ylim(0.1, 1e6)
ax.legend(ncol=4, frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.08))
fig.tight_layout()
fig.savefig(os.path.join(RESULTS, "01_grouped_log_bars.svg"), bbox_inches="tight")
plt.close(fig)
print("  wrote results/01_grouped_log_bars.svg")

# ════════════════════════════════════════════════════════════════════════════
# (b2) time vs pixels
# ════════════════════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(7.5, 5.2))
px = np.array([PIXELS[d] for d in B_DS])
order = np.argsort(px)
px_sorted = px[order]
for m, vals in METHODS_MS.items():
    v = np.array(vals)[order]
    ax.plot(px_sorted, v, "-o", color=B_COLORS[m], label=m, markersize=7, linewidth=1.6)
top = np.array(METHODS_MS["TAP-Vid Original"])[order]
for p, tval, name in zip(px_sorted, top, np.array(B_DS)[order]):
    ax.annotate(name, (p, tval), textcoords="offset points", xytext=(0, 8),
                ha="center", fontsize=8, color="#555")
ax.set_xscale("log"); ax.set_yscale("log")
ax.set_xlabel("pixels per frame (W × H)")
ax.set_ylabel("time per build (ms)")
ax.set_title("Scaling with frame size")
ax.yaxis.set_major_formatter(FuncFormatter(fmt_ms))
ax.xaxis.set_major_formatter(FuncFormatter(
    lambda v, _: f"{v/1e6:g}M" if v >= 1e6 else (f"{v/1e3:g}K" if v >= 1e3 else f"{v:g}")))
ax.legend(frameon=False, loc="lower right")
fig.tight_layout()
fig.savefig(os.path.join(RESULTS, "02_time_vs_pixels.svg"), bbox_inches="tight")
plt.close(fig)
print("  wrote results/02_time_vs_pixels.svg")

# ════════════════════════════════════════════════════════════════════════════
# (b3) speed-up heatmap vs TAP-Vid original
# ════════════════════════════════════════════════════════════════════════════
baseline = np.array(METHODS_MS["TAP-Vid Original"], float)
methods = ["Linear Interp.", "Flow-Blend", "Corridor DP"]
M = np.array([np.array(METHODS_MS[m], float) for m in methods])
speedup = baseline[None, :] / M
fig, ax = plt.subplots(figsize=(8.5, 3.2))
im = ax.imshow(np.log10(speedup), cmap="viridis", aspect="auto")
ax.set_xticks(range(len(B_DS))); ax.set_xticklabels(B_DS)
ax.set_yticks(range(len(methods))); ax.set_yticklabels(methods)
for i in range(speedup.shape[0]):
    for j in range(speedup.shape[1]):
        v = speedup[i, j]
        label = (f"{v:,.0f}×" if v >= 100 else f"{v:.0f}×" if v >= 10 else f"{v:.1f}×")
        ax.text(j, i, label, ha="center", va="center",
                color="white" if np.log10(v) < 4 else "black",
                fontsize=10, fontweight="bold")
ax.set_title("Speedup over TAP-Vid Original (per dataset)")
cbar = fig.colorbar(im, ax=ax, shrink=0.85)
cbar.set_label("log10(speedup)")
ax.grid(False)
fig.tight_layout()
fig.savefig(os.path.join(RESULTS, "03_speedup_heatmap.svg"), bbox_inches="tight")
plt.close(fig)
print("  wrote results/03_speedup_heatmap.svg")

# ─── speed-up ladder (mean rebuild time over datasets) ──────────────────────
mean_ms = {s: float(np.nanmean(list(all_timing[s].values()))) for s in STRATEGIES}
print("\nMean rebuild time at k=25 (ms):")
for s in STRATEGIES:
    print(f"  {B_LABEL[s]:18s} {mean_ms[s]:10.2f}")
print("\nSpeed-up ladder:")
print(f"  TAP-Vid -> Corridor-DP        {mean_ms['tapvid_original']/mean_ms['corridor_dp']:8.1f}x")
print(f"  Corridor-DP -> Flow-Blend     {mean_ms['corridor_dp']/mean_ms['flow_blend']:8.1f}x")
print(f"  TAP-Vid -> Flow-Blend         {mean_ms['tapvid_original']/mean_ms['flow_blend']:8.1f}x")
print(f"  Flow-Blend -> Linear          {mean_ms['flow_blend']/mean_ms['linear_interp']:8.1f}x")
