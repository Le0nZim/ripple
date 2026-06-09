#!/usr/bin/env python3
"""
Fig. 3c — average point precision (APP) vs. correction count, RIPPLE flow-blend
interpolation vs. the TAP-Vid algorithm.

For each dataset the per-track APP curve is linearly interpolated onto the union
of sampled correction counts and forward-filled to 100% past the last sample;
the dataset curve is the mean over tracks, shaded by the per-track min/max
envelope. A dashed line marks k = 25 corrections.

Reads ``data/optimal_corrections_{flow_blend,tapvid_original}.json`` and writes
``results/scaling_flow_blend.svg`` and ``results/scaling_tapvid_original.svg``.

    python reproduce_fig3c.py
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

DS_INCLUDE = ["neural", "pinned", "freely", "sperm", "homeostasis"]
DS_PLOT_COLORS = {"neural": "#88CCEE", "pinned": "#44AA99", "freely": "#AA4499",
                  "sperm": "#999933", "homeostasis": "#BBBBBB"}


def extract_curves(data):
    """Per-dataset mean/min/max APP curves vs. correction count."""
    curves = {}
    for ds in DS_INCLUDE:
        if ds not in data:
            continue
        track_curves = []
        for entry in data[ds].values():
            kv = {int(k): v * 100 for k, v in entry["app_curve"].items()}
            sks = sorted(kv)
            track_curves.append((max(sks), sks, [kv[k] for k in sks]))
        all_ks = sorted(set(k for _, sks, _ in track_curves for k in sks))
        all_vals = {}
        for k in all_ks:
            all_vals[k] = [100.0 if k > mk else float(np.interp(k, sks, sa))
                           for mk, sks, sa in track_curves]
        curves[ds] = {
            "ks": all_ks,
            "means": [np.mean(all_vals[k]) for k in all_ks],
            "mins": [np.min(all_vals[k]) for k in all_ks],
            "maxs": [np.max(all_vals[k]) for k in all_ks],
        }
    return curves


def plot_scaling(curves, stem, ylabel, vline_k=25):
    fig, ax = plt.subplots(figsize=(3.72, 2.28), layout="constrained")
    for ds in DS_INCLUDE:
        if ds not in curves:
            continue
        c = curves[ds]
        color = DS_PLOT_COLORS[ds]
        ax.plot(c["ks"], c["means"], color=color, lw=1.5)
        ax.fill_between(c["ks"], c["mins"], c["maxs"], color=color, alpha=0.12)
    if vline_k is not None:
        ax.axvline(vline_k, color="#888888", ls="--", lw=0.8, zorder=0)
    ax.set_xlabel("Number of corrections")
    ax.set_ylabel(ylabel, multialignment="center")
    ax.set_xlim(left=0)
    ax.set_ylim(0, 108)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.savefig(os.path.join(RESULTS, f"{stem}.svg"), bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote results/{stem}.svg")


def mean_app_at_k(curves, k=25):
    return float(np.mean([np.interp(k, c["ks"], c["means"]) for c in curves.values()]))


PANELS = [
    ("flow_blend", "scaling_flow_blend", "RIPPLE Interpolation\nAvg APP (%)"),
    ("tapvid_original", "scaling_tapvid_original", "TAPVid Original\nAvg APP (%)"),
]
for key, stem, ylabel in PANELS:
    data = json.load(open(os.path.join(HERE, "data", f"optimal_corrections_{key}.json")))
    curves = extract_curves(data)
    plot_scaling(curves, stem, ylabel)
    print(f"    mean APP at k=25 = {mean_app_at_k(curves):.2f}%")
