#!/usr/bin/env python3
"""
Fig. 1c — 3D visualization of tracked data.

Renders two views: (1) the trajectories alone, from ``data/tracks_3d.json``, and
(2) the trajectories overlaid on a volume rendering (grayscale slices at the
first, middle and last frame), from the raw volume
``source_data/neural_activity/raw_volume.tif``. Axes: X = frame (time),
Y = x (px), Z = y (px); each track is drawn in its stored colour.

The volume-overlay view is only produced if the raw volume is present; the
trajectories-alone view needs only the JSON.

    python reproduce_fig1c.py
"""
import json
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
RESULTS = os.path.join(HERE, "results")
os.makedirs(RESULTS, exist_ok=True)
VOLUME = os.path.join(ROOT, "source_data", "neural_activity", "raw_volume.tif")
BC_MIN, BC_MAX = 275, 814
SLICE_ALPHA = 1.0
VIEW = dict(elev=25, azim=-60)

data = json.load(open(os.path.join(HERE, "data", "tracks_3d.json")))
tracks = data["tracks"]
total_frames = data["metadata"]["total_frames"]

track_data = []
for t in tracks:
    anns = t["annotations"]
    if len(anns) < 2:
        continue
    frames = np.array([a["frame"] for a in anns])
    xs = np.array([a["x"] for a in anns])
    ys = np.array([a["y"] for a in anns])
    c = t["color"]
    color = (c["r"] / 255.0, c["g"] / 255.0, c["b"] / 255.0,
             min(c["a"] / 255.0, 1.0))
    track_data.append((frames, xs, ys, color))


# ─── 1. trajectories alone ──────────────────────────────────────────────────
fig = plt.figure(figsize=(16, 10))
ax = fig.add_subplot(111, projection="3d")
for frames, xs, ys, color in track_data:
    ax.plot(frames, xs, ys, color=color, linewidth=2.0)
ax.set_xlabel("Frame (time)")
ax.set_ylabel("X (px)")
ax.set_zlabel("Y (px)")
ax.invert_zaxis()
for pane in (ax.xaxis, ax.yaxis, ax.zaxis):
    pane.pane.fill = False
    pane.pane.set_edgecolor((1, 1, 1, 0))
ax.view_init(**VIEW)
plt.tight_layout()
fig.savefig(os.path.join(RESULTS, "tracks_3d_trajectories_only.png"), dpi=200)
fig.savefig(os.path.join(RESULTS, "tracks_3d_trajectories_only.svg"))
plt.close(fig)
print(f"  wrote results/tracks_3d_trajectories_only.{{png,svg}} ({len(track_data)} tracks)")


# ─── 2. trajectories on volume rendering ────────────────────────────────────
if os.path.exists(VOLUME):
    import tifffile
    print("  loading raw volume ...")
    vol = tifffile.imread(VOLUME)
    n_frames, h, w = vol.shape
    vol_f = np.clip((vol.astype(np.float64) - BC_MIN) / max(1, BC_MAX - BC_MIN), 0.0, 1.0)
    slice_indices = [0, n_frames // 2, n_frames - 1]
    xs_grid, ys_grid = np.arange(w), np.arange(h)
    X_grid, Y_grid = np.meshgrid(xs_grid, ys_grid)

    fig = plt.figure(figsize=(16, 10))
    ax = fig.add_subplot(111, projection="3d")
    for frames, xs, ys, color in track_data:
        ax.plot(frames, xs, ys, color=color, linewidth=2.0)
    for fi in slice_indices:
        img = vol_f[fi]
        rgba = np.zeros((*img.shape, 4))
        rgba[..., 0] = rgba[..., 1] = rgba[..., 2] = img
        rgba[..., 3] = SLICE_ALPHA
        F_grid = np.full_like(X_grid, fi, dtype=float)
        ax.plot_surface(F_grid, X_grid, Y_grid, facecolors=rgba,
                        rstride=4, cstride=4, shade=False, antialiased=False)
    ax.set_xlabel("Frame")
    ax.set_ylabel("X (px)")
    ax.set_zlabel("Y (px)")
    ax.invert_zaxis()
    for pane in (ax.xaxis, ax.yaxis, ax.zaxis):
        pane.pane.fill = False
        pane.pane.set_edgecolor((1, 1, 1, 0))
    ax.view_init(**VIEW)
    plt.tight_layout()
    fig.savefig(os.path.join(RESULTS, "tracks_3d_volume_overlay.png"), dpi=200)
    plt.close(fig)
    print("  wrote results/tracks_3d_volume_overlay.png")
else:
    print(f"  [skip] volume overlay: {os.path.relpath(VOLUME, ROOT)} not found")

print(f"\n{len(tracks)} tracks, {total_frames} frames.")
