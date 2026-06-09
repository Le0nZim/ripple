#!/usr/bin/env python3
"""
Generate the Fig. 3d input from the raw GCaMP volume in ``source_data/``.

Samples the GCaMP6s brightness of every RIPPLE-tracked neuron at its tracked
position in every frame, writing a neuron x frame matrix to
``data/neural_brightness_71.csv``. Track23 is excluded (as in the paper),
leaving 71 neurons.

Requires the raw volume ``source_data/neural_activity/gcamp_volume.tif`` and the
neuron tracks ``source_data/neural_activity/gcamp_annotations.json``, plus
``tifffile``.

    python generate_data.py
"""
import json
import os

import numpy as np
import tifffile

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
SRC = os.path.join(ROOT, "source_data", "neural_activity")
VOLUME = os.path.join(SRC, "gcamp_volume.tif")
TRACKS = os.path.join(SRC, "gcamp_annotations.json")
OUT = os.path.join(os.path.dirname(__file__), "data", "neural_brightness_71.csv")
DROP = {"Track23"}


def load_tracks(path, drop):
    data = json.load(open(path))
    total = data["metadata"]["total_frames"]
    tracks = {}
    for td in data["tracks"]:
        tid = td["track_id"]
        if tid in drop:
            continue
        coords = np.full((total, 2), np.nan)
        for a in td["annotations"]:
            coords[a["frame"]] = [a["x"], a["y"]]
        tracks[tid] = coords
    return tracks, total


def main():
    print(f"loading {os.path.relpath(VOLUME, ROOT)} ...")
    vol = tifffile.imread(VOLUME)
    _, H, W = vol.shape
    tracks, total = load_tracks(TRACKS, DROP)
    order = sorted(tracks, key=lambda t: int(t.replace("Track", "")))

    bright = {}
    for tid in order:
        coords = tracks[tid]
        vals = np.full(total, np.nan)
        for f in range(total):
            x, y = coords[f]
            if np.isnan(x):
                continue
            ix = int(np.clip(round(x), 0, W - 1))
            iy = int(np.clip(round(y), 0, H - 1))
            vals[f] = float(vol[f, iy, ix])
        bright[tid] = vals

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w") as fh:
        fh.write("frame," + ",".join(order) + "\n")
        for f in range(total):
            row = [str(f)] + ["" if np.isnan(bright[t][f]) else f"{bright[t][f]:.6g}"
                              for t in order]
            fh.write(",".join(row) + "\n")
    print(f"wrote {os.path.relpath(OUT, ROOT)}  ({len(order)} neurons x {total} frames)")


if __name__ == "__main__":
    main()
