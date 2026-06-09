#!/usr/bin/env python3
"""
Generate the Fig. 4c input from the raw annotations in ``source_data/``.

Computes, per dataset, the BEFORE average point precision (RIPPLE tracks vs.
ground truth) along with the manual-frame and RIPPLE-correction counts. These
are combined with the pre-computed AFTER values (``data/fig4c_after.json``) into
``data/fig4c_results.json``.

The AFTER value (GT-anchor flow-blend) is produced by substituting the manual
ground-truth coordinates at each RIPPLE anchor frame and re-running the flow-blend
interpolation; that step needs the per-dataset optical-flow volumes (15+ GB) and
is therefore shipped as a checkpoint rather than recomputed here.

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
HERE = os.path.dirname(__file__)
AFTER = json.load(open(os.path.join(HERE, "data", "fig4c_after.json")))["after"]
OUT = os.path.join(HERE, "data", "fig4c_results.json")


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
        gt_frames = anchor_count = 0
        for seg_id, rid in pairs:
            g, p = gt[seg_id], rip[rid]
            n = min(g.shape[0], p.shape[0])
            gts.append(g[:n]); preds.append(p[:n])
            gt_frames += int(np.sum(~np.isnan(g[:n]).any(axis=1)))
            anchor_count += len(anchors.get(rid, []))
        before = rio.compute_app(np.concatenate(gts), np.concatenate(preds))["average"]
        after = AFTER[name]
        out["datasets"][name] = {
            "gt_frames": gt_frames, "anchor_count": anchor_count,
            "before": round(before, 2), "after": after,
            "delta": round(after - before, 2),
        }
        print(f"{name:16s} before={before:6.2f}  after={after:6.2f}  "
              f"delta={after - before:+6.2f}")

    json.dump(out, open(OUT, "w"), indent=1)
    print(f"\nwrote {os.path.relpath(OUT, ROOT)}")


if __name__ == "__main__":
    main()
