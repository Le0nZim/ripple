#!/usr/bin/env python3
"""
Table 1 — RIPPLE accuracy and annotation effort across the five datasets.

The average point precision (APP) column is computed here from the matched
ground-truth/RIPPLE coordinates in ``data/app_inputs.json`` (produced by
``generate_data.py``). APP@tau is the fraction of frames in which the RIPPLE
point lies within tau px of the manual point, averaged over tau in
{1, 2, 4, 8, 16} and pooled over all frames of all matched tracks.

The effort columns (track count, interaction time, click counts) are logged
measurements from the annotation sessions.

    python reproduce_table1.py        # prints the table and writes results/table1.md
"""
import json
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = json.load(open(os.path.join(HERE, "data", "app_inputs.json")))
THR = DATA["thresholds"]

# Display order and logged (non-computed) effort measurements.
ROWS = [
    # name, label, tracks, ripple_time, ripple_clicks, manual_time, manual_clicks
    ("Neural Activity", "Neural Clytia",          10, "15 min",     158, "~2 h",   "4,000"),
    ("Pinned Down",     "Pinned Clytia",           3, "10 min",     116, "27 min", "1,200"),
    ("Freely",          "Freely swimming Clytia",  3, "14 min",     352, ">1 h",   "1,008"),
    ("Homeostasis",     "Homeostatic Clytia",     10, "4 min",       47, "14 min", "480"),
    ("Sperm",           "Sperm QPM",               6, "1 h 16 min", 529, ">2 h",   "1,847"),
]


def _arr(x):
    return np.array([[np.nan, np.nan] if v is None else v for v in x], float)


def app(name):
    d = DATA["datasets"][name]
    gt, pred = _arr(d["gt"]), _arr(d["pred"])
    valid = ~(np.isnan(gt).any(1) | np.isnan(pred).any(1))
    dist = np.sqrt(((gt[valid] - pred[valid]) ** 2).sum(1))
    return float(np.mean([(dist <= t).mean() * 100 for t in THR]))


header = ("| Dataset | Tracks | APP (%) | RIPPLE time | RIPPLE clicks | "
          "Manual time | Manual clicks |")
sep = "|---|---:|---:|---|---:|---|---:|"
md = [header, sep]
print(f"{'Dataset':<24}{'Tracks':>7}{'APP (%)':>9}{'RIPPLE time':>13}"
      f"{'clicks':>8}{'Manual time':>13}{'clicks':>8}")
for name, label, tracks, rt, rc, mt, mc in ROWS:
    a = app(name)
    print(f"{label:<24}{tracks:>7}{a:>9.2f}{rt:>13}{rc:>8}{mt:>13}{mc:>8}")
    md.append(f"| {label} | {tracks} | {a:.2f} | {rt} | {rc} | {mt} | {mc} |")

os.makedirs(os.path.join(HERE, "results"), exist_ok=True)
out = os.path.join(HERE, "results", "table1.md")
with open(out, "w") as f:
    f.write("# Table 1 — RIPPLE accuracy and annotation effort\n\n"
            + "\n".join(md) + "\n")
print(f"\nwrote {os.path.relpath(out, HERE)}")
