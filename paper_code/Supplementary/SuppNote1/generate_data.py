#!/usr/bin/env python3
"""
Generate the Supplementary Note 1 input from ``source_data/``.

Parses each dataset's TrackMate XML export and the matched ground-truth tracks,
and writes the compact arrays the cost analysis needs to
``data/intervention_inputs.json``: per matched GT track its visible coordinates,
and the TrackMate detections pruned to those within ``tol`` of a GT point at each
frame (the cost analysis only ever selects the nearest detection within ``tol``,
so this pruning is exact).

Reads ``source_data/trackmate/<dataset>.xml`` and the GT annotations under
``source_data/<dataset>/``. Requires ``numpy`` (and ``scipy``/``pynrrd`` for the
NRRD ground truth).

    python generate_data.py
"""
import json
import os
import sys
import xml.etree.ElementTree as ET

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, ROOT)
from common import ripple_io as rio

SOURCE = os.path.join(ROOT, "source_data")
OUT = os.path.join(os.path.dirname(__file__), "data", "intervention_inputs.json")

# TrackMate XML file and coordinate-scaling mode per dataset.
# "pixel" = XML already in pixels; "auto" = XML in microns, recover an isotropic
# pixel size from the most extreme coordinate vs. the native image size.
TM = {
    "Neural Activity": ("neural.xml", "auto"),
    "Pinned Down":     ("pinned.xml", "pixel"),
    "Sperm":           ("sperm.xml",  "pixel"),
    "Freely":          ("freely.xml", "pixel"),
    "Homeostasis":     ("homeo.xml",  "auto"),
}


def load_trackmate_xml(path):
    root = ET.parse(path).getroot()
    tracks = []
    for part in root.findall("particle"):
        ts, xs, ys = [], [], []
        for d in part.findall("detection"):
            ts.append(int(d.get("t"))); xs.append(float(d.get("x"))); ys.append(float(d.get("y")))
        if ts:
            order = np.argsort(ts)
            tracks.append({"frames": np.array(ts)[order],
                           "xy": np.stack([xs, ys], 1)[order]})
    return tracks


def compute_scale(tracks, h, w, mode):
    if mode == "pixel":
        return 1.0, 1.0
    max_x = max((t["xy"][:, 0].max() for t in tracks), default=1.0)
    max_y = max((t["xy"][:, 1].max() for t in tracks), default=1.0)
    s = max(w, h) / max(max_x, max_y)
    return s, s


def main():
    cfgs = rio.dataset_config(SOURCE)
    payload = {}
    for name in rio.DS_ORDER:
        cfg = cfgs[name]
        gt = rio.load_gt(cfg)
        rip = rio.load_ripple_tracks(cfg["ripple"])
        anchors = rio.load_ripple_anchors(cfg["ripple"])
        pairs = rio.match_tracks(gt, rip, tol=rio.MATCH_TOL[name])

        xml_name, mode = TM[name]
        tracks = load_trackmate_xml(os.path.join(SOURCE, "trackmate", xml_name))
        h, w = rio.NATIVE_SIZE[name]
        sx, sy = compute_scale(tracks, h, w, mode)
        n_tm_frames = max((int(t["frames"].max()) + 1 for t in tracks), default=0)
        per_frame = [[] for _ in range(n_tm_frames)]
        for ti, tr in enumerate(tracks):
            for f, (x, y) in zip(tr["frames"], tr["xy"]):
                per_frame[int(f)].append((ti, float(x * sx), float(y * sy)))

        tol = rio.MATCH_TOL[name]
        tracks_out, gt_by_frame = [], {}
        for seg_id, rid in pairs:
            g = gt[seg_id]
            vis = []
            for f in range(g.shape[0]):
                if not np.isnan(g[f]).any():
                    vis.append([int(f), float(g[f, 0]), float(g[f, 1])])
                    gt_by_frame.setdefault(f, []).append((g[f, 0], g[f, 1]))
            tracks_out.append({"seg_id": str(seg_id), "n_total_frames": int(g.shape[0]),
                               "visible": vis, "ripple_anchors": len(anchors.get(rid, []))})

        det_out = []
        for f in range(min(n_tm_frames, len(per_frame))):
            gts = gt_by_frame.get(f)
            if not gts:
                continue
            for tid, tx, ty in per_frame[f]:
                if any((tx - gx) ** 2 + (ty - gy) ** 2 <= tol * tol for gx, gy in gts):
                    det_out.append([int(f), int(tid), float(tx), float(ty)])

        payload[name] = {"tol": tol, "n_tm_frames": int(n_tm_frames),
                         "tracks": tracks_out, "detections": det_out}
        print(f"{name:16s} tracks={len(tracks_out):2d}  kept_dets={len(det_out):6d}  "
              f"tm_frames={n_tm_frames}")

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    json.dump(payload, open(OUT, "w"))
    print(f"\nwrote {os.path.relpath(OUT, ROOT)}  ({os.path.getsize(OUT) / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
