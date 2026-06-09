#!/usr/bin/env python3
"""
Generate the plotted values for Fig. 3a from the raw annotations in
``source_data/``.

For each dataset this records the three quantities the panel plots:
  * total_gt_frames — number of exhaustively annotated (valid) ground-truth
    frames across the matched tracks (the manual-effort x-position),
  * total_anchors   — number of RIPPLE corrections (the RIPPLE x-position),
  * avg_app         — RIPPLE average point precision (the RIPPLE y-position).

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
OUT = os.path.join(os.path.dirname(__file__), "data", "fig3a_values.json")


def main():
    cfgs = rio.dataset_config(SOURCE)
    out = {"datasets": {}}
    for name in rio.DS_ORDER:
        cfg = cfgs[name]
        gt = rio.load_gt(cfg)
        rip = rio.load_ripple_tracks(cfg["ripple"])
        anchors = rio.load_ripple_anchors(cfg["ripple"])
        pairs = rio.match_tracks(gt, rip, tol=rio.MATCH_TOL[name])

        gts, preds = [], []
        total_gt_frames = 0
        total_anchors = 0
        for seg_id, rid in pairs:
            g, p = gt[seg_id], rip[rid]
            n = min(g.shape[0], p.shape[0])
            gts.append(g[:n]); preds.append(p[:n])
            total_gt_frames += int(np.sum(~np.isnan(g[:n]).any(axis=1)))
            total_anchors += len(anchors.get(rid, []))
        app = rio.compute_app(np.concatenate(gts), np.concatenate(preds))

        out["datasets"][name] = {
            "total_gt_frames": total_gt_frames,
            "total_anchors": total_anchors,
            "avg_app": app["average"],
        }
        print(f"{name:16s} gt_frames={total_gt_frames:5d} anchors={total_anchors:4d} "
              f"avg_app={app['average']:.2f}")

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w") as f:
        json.dump(out, f, indent=1)
    print(f"\nwrote {os.path.relpath(OUT, ROOT)}")


if __name__ == "__main__":
    main()
