#!/usr/bin/env python3
"""
Fig. 3d — Neural *Clytia* GCaMP6s downstream analysis.

From the per-neuron GCaMP brightness matrix (``data/neural_brightness_71.csv``,
71 neurons x 400 frames) this reproduces the two published panels:

  * ``neural_heatmap_zscore``                — neuron x frame z-scored activity,
  * ``gcamp_event_triggered_zscore_trace``   — population calcium response
    aligned to detected events.

Styling (7 pt Arial, Tol-sunset colormap, exact panel sizes) matches the paper.

    python reproduce_fig3d.py
"""
import csv
import os

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from scipy.signal import find_peaks

mpl.use("Agg")

# ─── colours & style (as published) ────────────────────────────────────────
TOL_SUNSET = LinearSegmentedColormap.from_list("tol_sunset", [
    "#364B9A", "#4A7BB7", "#6EA6CD", "#98CAE1", "#C2E4EF", "#EAECCC",
    "#FEDA8B", "#FDB366", "#F67E4B", "#DD3D2D", "#A50026"])
ZSCORE_COLOR = "#84E291"
EVENT_LINE_COLOR = "#A0A0A0"
mpl.rcParams["figure.constrained_layout.h_pad"] = 0.02
mpl.rcParams["figure.constrained_layout.w_pad"] = 0.02
mpl.rcParams.update({
    "font.family": "sans-serif", "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size": 7, "axes.titlesize": 7, "axes.labelsize": 7,
    "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 7,
    "figure.facecolor": "white", "axes.facecolor": "white", "savefig.facecolor": "white",
    "axes.linewidth": 0.5, "xtick.major.width": 0.5, "ytick.major.width": 0.5,
    "xtick.major.size": 2.5, "ytick.major.size": 2.5, "lines.linewidth": 0.7,
    "pdf.fonttype": 42, "svg.fonttype": "none",
})

HERE = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(HERE, "results")
os.makedirs(OUT_DIR, exist_ok=True)
BASELINE_WIN = 11
PRE, POST = 20, 60


def _clean(ax, keep_left=True, keep_bottom=True):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    if not keep_left:
        ax.spines["left"].set_visible(False); ax.set_yticks([])
    if not keep_bottom:
        ax.spines["bottom"].set_visible(False); ax.set_xticks([])


def _save(fig, name):
    for ext in (".svg", ".png"):
        fig.savefig(os.path.join(OUT_DIR, name.replace(".svg", ext)),
                    transparent=False, dpi=300)
    plt.close(fig)
    print(f"  wrote results/{name}")


# ─── load brightness matrix ─────────────────────────────────────────────────
rows = list(csv.reader(open(os.path.join(HERE, "data", "neural_brightness_71.csv"))))
order = rows[0][1:]
data = np.array([[float(x) if x else np.nan for x in r[1:]] for r in rows[1:]])
n_frames = data.shape[0]
n_neurons = len(order)

# z-score matrix: per neuron dF/F over baseline, then z-score
z_mat = np.zeros((n_neurons, n_frames))
for i, tid in enumerate(order):
    trace = data[:, i].copy()
    f0 = np.nanmean(trace[:BASELINE_WIN])
    dff = (trace - f0) / f0 if f0 > 0 else trace
    mu, sigma = np.nanmean(dff), np.nanstd(dff)
    z_mat[i] = (dff - mu) / sigma if sigma > 0 else dff
z_mat = np.nan_to_num(z_mat)

# ─── 1. event-triggered population trace ────────────────────────────────────
pop_mean = np.mean(z_mat, axis=0)
events, _ = find_peaks(pop_mean, height=0.8, distance=15, prominence=0.5)
snippets = np.array([z_mat[:, ef - PRE:ef + POST] for ef in events
                     if ef - PRE >= 0 and ef + POST <= n_frames])
mean_per_neuron = np.mean(snippets, axis=0)
pop_trig_mean = np.mean(mean_per_neuron, axis=0)
pop_trig_sem = np.std(mean_per_neuron, axis=0) / np.sqrt(n_neurons)
time_axis = np.arange(-PRE, POST)

fig, ax = plt.subplots(figsize=(1.48, 2.43), constrained_layout=True)
ax.fill_between(time_axis, pop_trig_mean - pop_trig_sem, pop_trig_mean + pop_trig_sem,
                color=ZSCORE_COLOR, alpha=0.25, linewidth=0)
ax.plot(time_axis, pop_trig_mean, color=ZSCORE_COLOR, linewidth=1.2)
ax.axvline(0, color=EVENT_LINE_COLOR, linewidth=0.6, linestyle="--", zorder=0)
ax.set_ylabel("Pop. mean z-score")
ax.set_xlabel("Frame (event = 0)")
ax.set_xlim(time_axis[0], time_axis[-1])
_clean(ax)
_save(fig, "gcamp_event_triggered_zscore_trace.svg")

# ─── 2. z-scored activity heatmap ───────────────────────────────────────────
vabs = np.percentile(np.abs(z_mat), 97)
fig, ax = plt.subplots(figsize=(1.48, 2.82), constrained_layout=True)
im = ax.imshow(z_mat, aspect="auto", cmap=TOL_SUNSET, vmin=-vabs, vmax=vabs,
               interpolation="nearest")
ax.set_xlabel("Frame")
ax.set_ylabel("Neuron (1 px each)")
ax.set_yticks([])
cbar = fig.colorbar(im, ax=ax, shrink=0.8, pad=0.02)
cbar.set_label("z-score")
cbar.ax.tick_params(labelsize=7)
_clean(ax, keep_left=True, keep_bottom=True)
ax.spines["left"].set_visible(False)
_save(fig, "neural_heatmap_zscore.svg")

print(f"\n{n_neurons} neurons, {len(events)} detected calcium events "
      f"(frames {list(events)}).")
