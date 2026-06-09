#!/usr/bin/env python3
"""
Fig. 3e — Sperm QPM downstream analysis.

For two RIPPLE-tracked sperm cells (Track1, Track4) this reproduces, per cell,
the quantitative-phase-microscopy (QPM) phase trace and its spectrogram (the
spectral band reveals the periodic flagellar/rolling beat). Each trace is
truncated at the occlusion cutoff where the cell rolls out of view.

Reads ``data/sperm_brightness.csv`` and ``data/sperm_meta.json`` and writes the
four panels to ``results/`` with the published styling.

    python reproduce_fig3e.py
"""
import csv
import json
import os

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from scipy.signal import spectrogram as _spectrogram

mpl.use("Agg")

TOL_SUNSET = LinearSegmentedColormap.from_list("tol_sunset", [
    "#364B9A", "#4A7BB7", "#6EA6CD", "#98CAE1", "#C2E4EF", "#EAECCC",
    "#FEDA8B", "#FDB366", "#F67E4B", "#DD3D2D", "#A50026"])
SPERM_COLORS = {"Track1": "#999933", "Track4": "#AA4499"}
GUIDE_LINE = "#A0A0A0"
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
META = json.load(open(os.path.join(HERE, "data", "sperm_meta.json")))
CUT = META["cutoffs"]


def _clean(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def _save(fig, name):
    for ext in (".svg", ".png"):
        fig.savefig(os.path.join(OUT_DIR, name.replace(".svg", ext)),
                    transparent=False, dpi=300)
    plt.close(fig)
    print(f"  wrote results/{name}")


rows = list(csv.reader(open(os.path.join(HERE, "data", "sperm_brightness.csv"))))
order = rows[0][1:]
data = np.array([[float(x) if x else np.nan for x in r[1:]] for r in rows[1:]])
n_frames = data.shape[0]
bright = {tid: data[:, i] for i, tid in enumerate(order)}
order = sorted(bright, key=lambda t: int(t.replace("Track", "")))
frames_sp = np.arange(n_frames)

# ─── phase traces ───────────────────────────────────────────────────────────
for tid in order:
    label = tid.replace("Track", "s").lower()
    vals = bright[tid]; mask = ~np.isnan(vals); c = SPERM_COLORS[tid]
    fig, ax = plt.subplots(figsize=(1.73, 0.85), constrained_layout=True)
    ax.plot(frames_sp[mask], vals[mask], color=c, linewidth=0.6)
    ax.fill_between(frames_sp[mask], np.nanmin(vals[mask]), vals[mask],
                    alpha=0.12, color=c, linewidth=0)
    ax.set_ylabel("Phase (rad)")
    ax.set_xlabel("Frame")
    _clean(ax)
    if tid in CUT:
        ax.axvline(CUT[tid], color=GUIDE_LINE, linewidth=0.6, linestyle="--", zorder=2)
        ax.axvspan(CUT[tid], frames_sp[-1], color="#E0E0E0", alpha=0.35, zorder=0)
    _save(fig, f"sperm_trace_{label}.svg")

# ─── spectrograms ───────────────────────────────────────────────────────────
for tid in order:
    label = tid.replace("Track", "s").lower()
    vals = bright[tid]; y = vals[~np.isnan(vals)]
    nperseg = min(64, len(y) // 2)
    f_sg, t_sg, Sxx = _spectrogram(y, fs=1.0, nperseg=nperseg, noverlap=nperseg - 2)
    fig, ax = plt.subplots(figsize=(1.72, 1.07), constrained_layout=True)
    im = ax.pcolormesh(t_sg, f_sg, 10 * np.log10(Sxx + 1e-12), cmap=TOL_SUNSET,
                       shading="gouraud", rasterized=True)
    ax.set_ylabel("Cycles/frame")
    ax.set_xlabel("Frame")
    if tid in CUT:
        ax.axvline(CUT[tid], color="white", linewidth=0.8, linestyle="--", zorder=3)
    cbar = fig.colorbar(im, ax=ax, shrink=0.6, pad=0.005, aspect=15)
    cbar.set_label("Log Power (dB)", fontsize=7)
    cbar.ax.tick_params(labelsize=6)
    _clean(ax)
    _save(fig, f"sperm_spectrogram_{label}.svg")

print(f"\ncells {order}, cutoffs {CUT}")
