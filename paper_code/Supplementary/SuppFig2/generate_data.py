#!/usr/bin/env python3
"""
Generate the Supplementary Fig. 2 input from the raw annotations in
``source_data/``.

Records, per dataset, the matched (ground-truth, RIPPLE) coordinate arrays, the
native image size, the manual-frame and RIPPLE-correction counts, and the
per-anchor points — everything needed to recompute APP and annotator
disagreement both at native resolution and after rescaling to 256x256.

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
OUT = os.path.join(os.path.dirname(__file__), "data", "suppfig2_inputs.json")


def main():
    cfgs = rio.dataset_config(SOURCE)
    out = {"thresholds": rio.THRESHOLDS, "datasets": {}}
    for name in rio.DS_ORDER:
        cfg = cfgs[name]
        gt = rio.load_gt(cfg)
        rip = rio.load_ripple_tracks(cfg["ripple"])
        anchors = rio.load_ripple_anchors(cfg["ripple"])
        pairs = rio.match_tracks(gt, rip, tol=rio.MATCH_TOL[name])
        gts, preds, pts = [], [], []
        total_gt_frames = total_anchors = 0
        for seg_id, rid in pairs:
            g, p = gt[seg_id], rip[rid]
            n = min(g.shape[0], p.shape[0])
            gts.append(g[:n]); preds.append(p[:n])
            total_gt_frames += int(np.sum(~np.isnan(g[:n]).any(axis=1)))
            total_anchors += len(anchors.get(rid, []))
            for frame, ax, ay in anchors[rid]:
                if frame < g.shape[0] and not np.isnan(g[frame]).any():
                    gx, gy = g[frame]
                    pts.append([int(frame), float(gx), float(gy), float(ax), float(ay)])
        gt_arr, pred_arr = np.concatenate(gts), np.concatenate(preds)
        out["datasets"][name] = {
            "native_hw": list(rio.NATIVE_SIZE[name]),
            "total_gt_frames": total_gt_frames,
            "total_anchors": total_anchors,
            "gt": np.where(np.isnan(gt_arr), None, gt_arr).tolist(),
            "pred": np.where(np.isnan(pred_arr), None, pred_arr).tolist(),
            "anchor_points": pts,
        }
        print(f"{name:16s} frames={gt_arr.shape[0]:5d} anchors={total_anchors:4d}")

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    json.dump(out, open(OUT, "w"))
    print(f"\nwrote {os.path.relpath(OUT, ROOT)}")


if __name__ == "__main__":
    main()
