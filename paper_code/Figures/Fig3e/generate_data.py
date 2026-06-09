#!/usr/bin/env python3
"""
Generate the Fig. 3e input from the raw QPM volume in ``source_data/``.

Samples the quantitative-phase-microscopy (QPM) phase of two RIPPLE-tracked sperm
cells (Track1, Track4) at their tracked positions in every frame, and records the
occlusion cutoff (the frame at which each cell rolls out of view). Writes
``data/sperm_brightness.csv`` and ``data/sperm_meta.json``.

Requires ``source_data/sperm/qpm_volume.tif`` (~1.9 GB) and ``tifffile``.

    python generate_data.py
"""
import json
import os

import numpy as np
import tifffile

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
SRC = os.path.join(ROOT, "source_data", "sperm")
VOLUME = os.path.join(SRC, "qpm_volume.tif")
TRACKS = os.path.join(SRC, "ripple_annotations.json")
DATA = os.path.join(os.path.dirname(__file__), "data")
KEEP = {"Track1", "Track4"}


def main():
    data = json.load(open(TRACKS))
    total = data["metadata"]["total_frames"]
    tracks, meta = {}, {}
    for td in data["tracks"]:
        if td["track_id"] not in KEEP:
            continue
        coords = np.full((total, 2), np.nan)
        for a in td["annotations"]:
            coords[a["frame"]] = [a["x"], a["y"]]
        tracks[td["track_id"]] = coords
        meta[td["track_id"]] = td

    # occlusion cutoff = start of the terminal occlusion segment
    cutoffs = {}
    for tid, tm in meta.items():
        terminal = [s for s in tm.get("occlusion_segments", []) if s["end"] >= total - 10]
        if terminal:
            cutoffs[tid] = terminal[0]["start"]

    print(f"loading {os.path.relpath(VOLUME, ROOT)} (memmap) ...")
    vol = tifffile.imread(VOLUME, out="memmap")
    _, H, W = vol.shape
    order = sorted(tracks, key=lambda t: int(t.replace("Track", "")))
    bright = {}
    for tid in order:
        coords = tracks[tid]
        end = cutoffs.get(tid, total)
        vals = np.full(total, np.nan)
        for f in range(end):
            x, y = coords[f]
            if np.isnan(x):
                continue
            ix = int(np.clip(round(x), 0, W - 1))
            iy = int(np.clip(round(y), 0, H - 1))
            vals[f] = float(vol[f, iy, ix])
        bright[tid] = vals

    os.makedirs(DATA, exist_ok=True)
    with open(os.path.join(DATA, "sperm_brightness.csv"), "w") as fh:
        fh.write("frame," + ",".join(order) + "\n")
        for f in range(total):
            row = [str(f)] + ["" if np.isnan(bright[t][f]) else f"{bright[t][f]:.6g}"
                              for t in order]
            fh.write(",".join(row) + "\n")
    json.dump({"cutoffs": cutoffs, "total_frames": total},
              open(os.path.join(DATA, "sperm_meta.json"), "w"), indent=1)
    print(f"wrote data/sperm_brightness.csv ({order}, {total} frames), cutoffs={cutoffs}")


if __name__ == "__main__":
    main()
