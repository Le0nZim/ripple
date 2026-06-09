#!/usr/bin/env python3
"""
Generate the Fig. 1c trajectory input from ``source_data/``.

Slims the full 3D-visualization annotation file
(``source_data/neural_activity/annotations_3d_viz.json``) down to the fields
needed to render the trajectories-alone panel (per track: id, colour, and the
per-frame x/y), writing ``data/tracks_3d.json``.

    python generate_data.py
"""
import json
import os

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
SRC = os.path.join(ROOT, "source_data", "neural_activity", "annotations_3d_viz.json")
OUT = os.path.join(os.path.dirname(__file__), "data", "tracks_3d.json")


def main():
    src = json.load(open(SRC))
    out = {"metadata": {"total_frames": src["metadata"]["total_frames"]}, "tracks": []}
    for t in src["tracks"]:
        out["tracks"].append({
            "track_id": t["track_id"],
            "color": t["color"],
            "annotations": [{"frame": a["frame"], "x": a["x"], "y": a["y"]}
                            for a in t["annotations"]],
        })
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    json.dump(out, open(OUT, "w"))
    print(f"wrote {os.path.relpath(OUT, ROOT)}  "
          f"({len(out['tracks'])} tracks, {out['metadata']['total_frames']} frames)")


if __name__ == "__main__":
    main()
