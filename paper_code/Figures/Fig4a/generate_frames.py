#!/usr/bin/env python3
"""
Generate the Fig. 4a representative crops from the raw videos in ``source_data/``.

For every matched track, the anchor frame with the largest ground-truth-vs-RIPPLE
disagreement is located, a square region is cropped from the raw video centred on
the two markers, and the crop is rendered with the ground-truth (diamond) and
RIPPLE (square) markers. The selected frames are also written to
``data/fig4a_worst_frames.json``.

This is both the data-generation and figure-reproduction step for Fig. 4a: the
crops are images, so they require the raw videos. Place each dataset's video at
``source_data/<dataset>/video.tif`` (see ``source_data/README.md``).

    python generate_frames.py
"""
import json
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import tifffile

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, ROOT)
from common import ripple_io as rio

SOURCE = os.path.join(ROOT, "source_data")
HERE = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(HERE, "results", "frames")

matplotlib.rcParams["font.family"] = "sans-serif"
matplotlib.rcParams["font.sans-serif"] = ["Arial", "Helvetica", "DejaVu Sans"]

# dataset -> source_data folder name (where video.tif and annotations live)
SRC_DIR = {"Neural Activity": "neural_activity", "Pinned Down": "pinned_jelly",
           "Freely": "freely", "Sperm": "sperm", "Homeostasis": "homeostasis"}
SHORT = {"Neural Activity": "neural", "Pinned Down": "pinned", "Freely": "freely",
         "Sperm": "sperm", "Homeostasis": "homeostasis"}

GT_MARKER, GT_COLOR = "D", "#CCBB44"
RIPPLE_MARKER, RIPPLE_COLOR = "s", "#EE3377"
EDGE_COLOR = "white"
DATASET_RENDER = {
    "Neural Activity": dict(crop_size=100, marker_size=9),
    "Pinned Down":     dict(crop_size=50,  marker_size=9),
    "Freely":          dict(crop_size=50,  marker_size=9),
    "Sperm":           dict(crop_size=100, marker_size=30),
    "Homeostasis":     dict(crop_size=100, marker_size=30),
}
BC_NEURAL = (275, 814)
EXCLUDE_FRAMES = {("freely", 3): {256}, ("sperm", 3): {48}}
TRACK_BC_OVERRIDE = {("neural", 3): (275, 2000), ("sperm", 1): (0.0, 4.0),
                     ("sperm", 4): (0.0, 4.0)}


def find_worst_anchor(gt, anchors, exclude):
    worst = None
    for frame, ax, ay in anchors:
        if frame in exclude or frame >= gt.shape[0]:
            continue
        gx, gy = gt[frame]
        if np.isnan(gx) or np.isnan(gy):
            continue
        d = np.hypot(gx - ax, gy - ay)
        if worst is None or d > worst[3]:
            worst = (frame, (gx, gy), (ax, ay), d)
    return worst


def render_crop(tiff_path, frame_idx, gt_xy, rip_xy, out_path, crop_size,
                marker_size, bc_min, bc_max, mag_label):
    with tifffile.TiffFile(tiff_path) as tif:
        if len(tif.pages) == 1 and tif.series[0].shape[0] > 1:
            img = tif.series[0].asarray()[frame_idx]
        else:
            img = tif.pages[frame_idx].asarray()
    cx, cy = (gt_xy[0] + rip_xy[0]) / 2, (gt_xy[1] + rip_xy[1]) / 2
    half = crop_size // 2
    h, w = img.shape[:2]
    x0 = int(np.clip(cx - half, 0, w - crop_size))
    y0 = int(np.clip(cy - half, 0, h - crop_size))
    crop = img[y0:y0 + crop_size, x0:x0 + crop_size]
    gt_rel = (gt_xy[0] - x0, gt_xy[1] - y0)
    rip_rel = (rip_xy[0] - x0, rip_xy[1] - y0)
    edge_lw = 0.6 if marker_size >= 20 else 0.4

    fig, ax = plt.subplots(figsize=(2.5, 2.5))
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
    if crop.ndim == 2:
        kw = dict(cmap="gray", interpolation="nearest")
        if bc_min is not None:
            kw["vmin"], kw["vmax"] = bc_min, bc_max
        ax.imshow(crop, **kw)
    else:
        ax.imshow(crop, interpolation="nearest")
    ax.scatter(*gt_rel, marker=GT_MARKER, s=marker_size, c=GT_COLOR,
               edgecolors=EDGE_COLOR, linewidths=edge_lw, zorder=5)
    ax.scatter(*rip_rel, marker=RIPPLE_MARKER, s=marker_size, c=RIPPLE_COLOR,
               edgecolors=EDGE_COLOR, linewidths=edge_lw, zorder=5)
    if mag_label:
        ax.text(0.04, 0.04, mag_label, transform=ax.transAxes, fontsize=20,
                fontfamily="Arial", fontweight="bold", color="white",
                ha="left", va="bottom",
                bbox=dict(facecolor="black", alpha=0.7, edgecolor="none", pad=1.5))
    ax.axis("off")
    fig.savefig(out_path, bbox_inches="tight", pad_inches=0)
    plt.close(fig)


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    cfgs = rio.dataset_config(SOURCE)
    selection = {"datasets": {}}
    for name in rio.DS_ORDER:
        cfg = cfgs[name]
        short = SHORT[name]
        tiff_path = os.path.join(SOURCE, SRC_DIR[name], "video.tif")
        if not os.path.exists(tiff_path):
            print(f"  [skip] {name}: {os.path.relpath(tiff_path, ROOT)} not found")
            continue
        rcfg = DATASET_RENDER[name]
        with tifffile.TiffFile(tiff_path) as tif:
            page = tif.pages[0].asarray()
        mag_label = f"{min(page.shape[0], page.shape[1]) / rcfg['crop_size']:.1f}\u00d7"
        if name == "Neural Activity":
            bc = BC_NEURAL
        elif name == "Sperm":
            raw = tifffile.imread(tiff_path)
            bc = (float(np.percentile(raw, 1)), float(np.percentile(raw, 99)))
        else:
            bc = (None, None)

        gt = rio.load_gt(cfg)
        rip = rio.load_ripple_tracks(cfg["ripple"])
        anchors = rio.load_ripple_anchors(cfg["ripple"])
        pairs = rio.match_tracks(gt, rip, tol=rio.MATCH_TOL[name])
        print(f"=== {name} ({len(pairs)} tracks) ===")
        sel = {}
        for seg_id, rid in pairs:
            excl = EXCLUDE_FRAMES.get((short, seg_id), set())
            worst = find_worst_anchor(gt[seg_id], anchors[rid], excl)
            if worst is None:
                continue
            frame, gt_xy, rip_xy, dist = worst
            tb = TRACK_BC_OVERRIDE.get((short, seg_id))
            bmin, bmax = (tb if tb else bc)
            fname = f"{short}_seg{seg_id}_track{rid}_f{frame}.svg"
            render_crop(tiff_path, frame, gt_xy, rip_xy,
                        os.path.join(OUT_DIR, fname), rcfg["crop_size"],
                        rcfg["marker_size"], bmin, bmax, mag_label)
            sel[str(rid)] = {"seg_id": seg_id, "worst_frame": frame,
                             "dist": round(float(dist), 2)}
            print(f"  seg={seg_id} track={rid} frame={frame} dist={dist:.1f}px")
        selection["datasets"][name] = sel

    os.makedirs(os.path.join(HERE, "data"), exist_ok=True)
    json.dump(selection, open(os.path.join(HERE, "data", "fig4a_worst_frames.json"), "w"),
              indent=1)
    print("\nwrote data/fig4a_worst_frames.json and crops in results/frames/")


if __name__ == "__main__":
    main()
