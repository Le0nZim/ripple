#!/usr/bin/env python3
"""
Table 3 — track-rebuild time and average APP at k = 25 corrections, RIPPLE vs.
the TAP-Vid algorithm.

Both quantities are computed from the sparse-correction scaling-experiment
outputs (``data/optimal_corrections_*.json``) — the same artefacts that drive
Fig. 3c and Supplementary Fig. 1.

  * rebuild time (per dataset) = mean over that dataset's tracks of the rebuild
    time at k = 25, in ms.
  * Mean = the per-dataset rebuild times averaged across datasets (per the table
    caption). Rebuild times are wall-clock measurements and therefore depend on
    the host machine; the >10,000x RIPPLE/TAP-Vid ratio is the hardware-
    independent result.
  * Avg APP at k = 25 = mean over datasets of the per-dataset mean APP at k = 25
    (each track's APP curve interpolated to k = 25). This is deterministic.

    python reproduce_table3.py        # prints the table and writes results/table3.md
"""
import json
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
DS = ["pinned", "neural", "sperm", "freely", "homeostasis"]
DS_LABEL = ["Pinned", "Neural", "Sperm", "Freely", "Homeo"]
SRC = {
    "RIPPLE":  os.path.join(HERE, "data", "optimal_corrections_flow_blend.json"),
    "TAP-Vid": os.path.join(HERE, "data", "optimal_corrections_tapvid_original.json"),
}


def time_at_k(track, target=25, tol=5):
    tc = track.get("timing", {}).get("time_curve", {})
    if not tc:
        return None
    keys = sorted(int(k) for k in tc)
    best = min(keys, key=lambda k: abs(k - target))
    return float(tc[str(best)]) if abs(best - target) <= tol else None


def app_at_k(track, target=25):
    kv = {int(k): v * 100 for k, v in track["app_curve"].items()}
    ks = sorted(kv)
    return 100.0 if target > max(ks) else float(np.interp(target, ks, [kv[k] for k in ks]))


def rebuild_times(data):
    """Per-dataset mean rebuild time (ms) and the across-dataset mean."""
    per_ds = {}
    for ds in DS:
        ts = [t * 1000 for t in (time_at_k(tr) for tr in data.get(ds, {}).values())
              if t is not None]
        per_ds[ds] = float(np.mean(ts)) if ts else float("nan")
    across = float(np.nanmean([per_ds[ds] for ds in DS]))
    return per_ds, across


def avg_app(data):
    per_ds = [float(np.mean([app_at_k(tr) for tr in data[ds].values()]))
              for ds in DS if ds in data]
    return float(np.mean(per_ds))


header = "| Method | " + " | ".join(DS_LABEL) + " | Mean | Avg APP at k=25 (%) |"
sep = "|---|" + "".join("---:|" for _ in DS_LABEL) + "---:|---:|"
md = [header, sep]
print(f"{'Method':<9}" + "".join(f"{d:>9}" for d in DS_LABEL) +
      f"{'Mean':>10}{'AvgAPP':>9}")
for method, path in SRC.items():
    data = json.load(open(path))
    per_ds, mean = rebuild_times(data)
    app = avg_app(data)
    print(f"{method:<9}" + "".join(f"{per_ds[d]:>9.1f}" for d in DS) +
          f"{mean:>10.1f}{app:>9.2f}")
    cells = " | ".join(f"{per_ds[d]:,.1f}" for d in DS)
    md.append(f"| {method} | {cells} | {mean:,.1f} | {app:.2f} |")

os.makedirs(os.path.join(HERE, "results"), exist_ok=True)
out = os.path.join(HERE, "results", "table3.md")
with open(out, "w") as f:
    f.write("# Table 3 — rebuild time and average APP at k = 25\n\n"
            + "\n".join(md) + "\n")
print(f"\nwrote {os.path.relpath(out, HERE)}")
