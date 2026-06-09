#!/usr/bin/env python3
"""
Generate the lightweight input for Table 1 from the raw annotations in
``source_data/``.

For each dataset this matches every manual ground-truth track to its RIPPLE
track and stores the two aligned per-frame coordinate arrays. ``reproduce_
table1.py`` then computes the average point precision (APP) from these arrays
without needing the multi-gigabyte NRRD volumes.

Run from this folder (with ``source_data/`` present at the repository root):

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
OUT = os.path.join(os.path.dirname(__file__), "data", "app_inputs.json")


def main():
    cfgs = rio.dataset_config(SOURCE)
    out = {"thresholds": rio.THRESHOLDS, "datasets": {}}
    for name in rio.DS_ORDER:
        cfg = cfgs[name]
        gt = rio.load_gt(cfg)
        rip = rio.load_ripple_tracks(cfg["ripple"])
        pairs = rio.match_tracks(gt, rip, tol=rio.MATCH_TOL[name])
        gts, preds = [], []
        for seg_id, rid in pairs:
            g, p = gt[seg_id], rip[rid]
            n = min(g.shape[0], p.shape[0])
            gts.append(g[:n])
            preds.append(p[:n])
        gt_arr = np.concatenate(gts)
        pred_arr = np.concatenate(preds)
        out["datasets"][name] = {
            "n_tracks": len(pairs),
            "native_hw": list(rio.NATIVE_SIZE[name]),
            "gt": np.where(np.isnan(gt_arr), None, gt_arr).tolist(),
            "pred": np.where(np.isnan(pred_arr), None, pred_arr).tolist(),
        }
        print(f"{name:16s} tracks={len(pairs):2d}  frames={gt_arr.shape[0]:5d}")

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w") as f:
        json.dump(out, f)
    print(f"\nwrote {os.path.relpath(OUT, ROOT)}  "
          f"({os.path.getsize(OUT) / 1e6:.2f} MB)")


if __name__ == "__main__":
    main()
