#!/usr/bin/env python3
"""
Fig. 3b — RIPPLE vs. baseline trackers on four practical-cost axes.

For the Neural *Clytia* dataset, each method's average point precision (APP) is
plotted against four axes of practical cost: manual annotations, total elapsed
time, computation time, and the size of the hyper-parameter search space. The
exhaustive-manual reference sits at 100% APP on the time/parameter axes.

Everything is read from ``data/benchmark_results.json`` (the baseline-evaluation
output; see the README for provenance). The four panels are written to
``results/`` using the same styling as the published figure.

    python reproduce_fig3b.py
"""
import json
import os
from collections import defaultdict

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

FIG_SIZE = (1.86, 1.26)

METHOD_COLORS = {"LocoTrack": "#88CCEE", "Ripple": "#00B050", "TrackMate": "#88CCEE",
                 "SLEAP": "#88CCEE", "SLEAP-op": "#88CCEE", "Manual": "#CC6677"}
METHOD_MARKERS = {"LocoTrack": "o", "Ripple": "s", "TrackMate": "^",
                  "SLEAP": "P", "SLEAP-op": "X", "Manual": "D"}

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(HERE, "results")
os.makedirs(RESULTS, exist_ok=True)
data = json.load(open(os.path.join(HERE, "data", "benchmark_results.json")))
methods = data["methods"]
t = data["timing"]


def jitter_duplicates(points, x_frac=0.015, y_frac=0.015):
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    x_range = (max(xs) - min(xs)) if len(xs) > 1 else (max(abs(v) for v in xs) or 1.0)
    y_range = (max(ys) - min(ys)) if len(ys) > 1 else (max(abs(v) for v in ys) or 1.0)
    dx, dy = x_range * x_frac, y_range * y_frac
    groups = defaultdict(list)
    for i, p in enumerate(points):
        groups[(round(p[0], 6), round(p[1], 6))].append(i)
    out = list(points)
    for idxs in groups.values():
        n = len(idxs)
        if n == 1:
            continue
        for rank, i in enumerate(idxs):
            ang = 2 * np.pi * rank / n
            out[i] = (points[i][0] + dx * np.cos(ang), points[i][1] + dy * np.sin(ang))
    return out


def aggregate_plot(x_values, names, app_pct, xlabel, stem):
    fig, ax = plt.subplots(figsize=FIG_SIZE, layout="constrained")
    use_log = "Time" in xlabel
    pts = jitter_duplicates(list(zip(x_values, app_pct)))
    for (x, y), name in zip(pts, names):
        if use_log and x <= 0:
            continue
        ax.scatter(x, y, c=METHOD_COLORS[name], marker=METHOD_MARKERS[name],
                   s=60, zorder=5, edgecolors="black", linewidths=0.5, clip_on=False)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Avg APP (%)")
    if use_log:
        ax.set_xscale("log")
    ax.set_ylim(-5, 108)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.savefig(os.path.join(RESULTS, f"{stem}.svg"), bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote results/{stem}.svg")


app = [data["aggregate_app"][m]["average"] for m in methods]
manual_real_s = t["manual_real_min"] * 60.0
manual_comp_s = t.get("manual_comp_s", 0)

# Each panel: (key, xlabel, per-method x-values, append Manual reference?)
PANELS = {
    "annotations": ("Manual Annotations",
                    [data["annotations"][m] for m in methods], False),
    "real_time": ("Real Time (s)",
                  [t["locotrack_inference_s"], t["ripple_real_s"],
                   data["real_time_min"]["TrackMate"], t["sleap_real_s"],
                   t["sleap_op_real_s"]], True),
    "comp_time_no_cache": ("Comp. Time (s)",
                           [t["locotrack_inference_s"], t["ripple_comp_no_cache_s"],
                            t["trackmate_comp_s"], t["sleap_comp_s"],
                            t["sleap_op_comp_s"]], True),
    "params": ("Hyper-Parameters", [0, 0, 357, 0, 0], True),
}
MANUAL_X = {"real_time": manual_real_s, "comp_time_no_cache": manual_comp_s,
            "params": 0}

for key, (xlabel, xvals, add_manual) in PANELS.items():
    names, xs, ys = list(methods), list(xvals), list(app)
    if add_manual:
        names += ["Manual"]; xs += [MANUAL_X[key]]; ys += [100.0]
    aggregate_plot(xs, names, ys, xlabel, f"aggregate_{key}_aggregated")

print("\nRIPPLE APP = %.2f%% (158 annotations).  Panels written to results/."
      % data["aggregate_app"]["Ripple"]["average"])
