#!/usr/bin/env python3
"""
Generate the Fig. 4b input from the raw annotations in ``source_data/``.

For every matched track, at each RIPPLE-corrected (anchor) frame, records the
ground-truth point and the comparison annotator's RIPPLE point. The Euclidean
distance between them is the annotator disagreement. Writes
``data/fig4b_inputs.json``.

    python generate_data.py
"""
import json
import os
import sys

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, ROOT)
from common import ripple_io as rio

SOURCE = os.path.join(ROOT, "source_data")
OUT = os.path.join(os.path.dirname(__file__), "data", "fig4b_inputs.json")


def main():
    cfgs = rio.dataset_config(SOURCE)
    out = {"datasets": {}}
    for name in rio.DS_ORDER:
        cfg = cfgs[name]
        gt = rio.load_gt(cfg)
        rip = rio.load_ripple_tracks(cfg["ripple"])
        anchors = rio.load_ripple_anchors(cfg["ripple"])
        pairs = rio.match_tracks(gt, rip, tol=rio.MATCH_TOL[name])
        pts = []
        for seg_id, rid in pairs:
            g = gt[seg_id]
            for frame, ax, ay in anchors[rid]:
                if frame >= g.shape[0]:
                    continue
                gx, gy = g[frame]
                if np.isnan(gx) or np.isnan(gy):
                    continue
                pts.append([int(frame), float(gx), float(gy), float(ax), float(ay)])
        out["datasets"][name] = {"anchor_points": pts}
        d = np.array([np.hypot(p[1] - p[3], p[2] - p[4]) for p in pts])
        print(f"{name:16s} n={len(pts):4d}  median={np.median(d):6.2f}  mean={np.mean(d):6.2f}")

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    json.dump(out, open(OUT, "w"))
    print(f"\nwrote {os.path.relpath(OUT, ROOT)}")


if __name__ == "__main__":
    main()
