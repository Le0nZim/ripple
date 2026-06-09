"""
Shared I/O and metric helpers for the RIPPLE paper analysis.

Used by the ``generate_data.py`` scripts in each figure/table folder to turn the
raw microscopy annotations under ``source_data/`` into the lightweight
``data/`` files that the ``reproduce_*.py`` scripts consume. Keeping these
loaders in one place avoids duplicating the (non-trivial) NRRD centroid
extraction and track-matching logic across folders.

Ground-truth point positions come from two annotation formats:
  * NRRD segmentation volumes (one labelled mask per frame) — neural, pinned,
    sperm and freely-swimming datasets. The point is the centre of mass of the
    mask in each frame.
  * RIPPLE-style JSON (a list of tracks, each with per-frame x/y) — used for the
    homeostasis manual annotation and for every dataset's RIPPLE annotations.
"""
import json
import os

import numpy as np
import nrrd
from scipy.ndimage import center_of_mass

# Native image size (height, width) of each dataset, used to rescale
# coordinates to a common grid when comparing APP across datasets.
NATIVE_SIZE = {
    "Neural Activity": (600, 600),
    "Pinned Down":     (100, 100),
    "Sperm":           (1024, 1224),
    "Freely":          (1024, 1280),
    "Homeostasis":     (5632, 5632),
}

# Dataset display order and the track-matching tolerance (px in native space).
DS_ORDER = ["Neural Activity", "Pinned Down", "Freely", "Sperm", "Homeostasis"]
MATCH_TOL = {
    "Neural Activity": 20.0, "Pinned Down": 20.0, "Freely": 20.0,
    "Sperm": 20.0, "Homeostasis": 200.0,
}

# Tol-muted dataset colours used throughout the paper.
DS_COLORS = {
    "Neural Activity": "#88CCEE",
    "Pinned Down":     "#44AA99",
    "Freely":          "#AA4499",
    "Sperm":           "#999933",
    "Homeostasis":     "#BBBBBB",
}
DS_SHORT = {"Neural Activity": "Neural", "Pinned Down": "Pinned",
            "Freely": "Freely", "Sperm": "Sperm", "Homeostasis": "Homeo"}

THRESHOLDS = [1, 2, 4, 8, 16]


def dataset_config(source_data_root):
    """Return per-dataset file paths under a ``source_data/`` directory.

    ``gt`` is either a dict of {segmentation id: NRRD path} (``gt_type='nrrd'``)
    or a single RIPPLE-style JSON path (``gt_type='json'``).
    """
    sd = source_data_root

    def seg(ds, ids, width=1):
        base = os.path.join(sd, ds, "manual_segmentation")
        return {i: os.path.join(base, f"segmentation_id{i:0{width}d}.nrrd")
                for i in ids}

    return {
        "Neural Activity": dict(
            gt_type="nrrd", gt=seg("neural_activity", range(1, 11), width=1),
            ripple=os.path.join(sd, "neural_activity", "ripple_annotations.json")),
        "Pinned Down": dict(
            gt_type="nrrd", gt=seg("pinned_jelly", range(1, 4), width=2),
            ripple=os.path.join(sd, "pinned_jelly", "ripple_annotations.json")),
        "Sperm": dict(
            gt_type="nrrd", gt=seg("sperm", range(1, 7), width=2),
            ripple=os.path.join(sd, "sperm", "ripple_annotations.json")),
        "Freely": dict(
            gt_type="nrrd", gt=seg("freely", range(1, 4), width=2),
            ripple=os.path.join(sd, "freely", "ripple_annotations.json")),
        "Homeostasis": dict(
            gt_type="json",
            gt=os.path.join(sd, "homeostasis", "manual_segmentation.json"),
            ripple=os.path.join(sd, "homeostasis", "ripple_annotations.json")),
    }


def load_nrrd_centroids(nrrd_files):
    """Centre of mass of each frame's mask. Returns {seg_id: (n_frames, 2)}."""
    out = {}
    for seg_id, path in nrrd_files.items():
        data, _ = nrrd.read(path)
        n_frames = data.shape[2]
        cen = np.full((n_frames, 2), np.nan)
        for f in range(n_frames):
            mask = data[:, :, f]
            if mask.any():
                cen[f] = center_of_mass(mask)
        out[seg_id] = cen
    return out


def load_ripple_tracks(json_path):
    """Dense per-frame coordinates for every track. {track_id: (n_frames, 2)}."""
    with open(json_path) as f:
        data = json.load(f)
    n_frames = data["metadata"]["total_frames"]
    tracks = {}
    for td in data["tracks"]:
        coords = np.full((n_frames, 2), np.nan)
        for ann in td["annotations"]:
            coords[ann["frame"]] = [ann["x"], ann["y"]]
        tracks[td["track_id"]] = coords
    return tracks


def load_ripple_anchors(json_path):
    """Sparse user corrections (anchors). {track_id: [(frame, x, y), ...]}."""
    with open(json_path) as f:
        data = json.load(f)
    anchors = {}
    for td in data["tracks"]:
        a = [(int(an["frame"]), float(an["x"]), float(an["y"]))
             for an in td.get("anchors", [])]
        a.sort(key=lambda p: p[0])
        anchors[td["track_id"]] = a
    return anchors


def load_gt(cfg):
    """Load ground-truth point tracks from a dataset config (NRRD or JSON)."""
    if cfg["gt_type"] == "nrrd":
        return load_nrrd_centroids(cfg["gt"])
    return load_ripple_tracks(cfg["gt"])


def match_tracks(gt_centroids, ripple_tracks, tol=20.0):
    """Greedy nearest-neighbour match on the first valid frame. -> [(seg, rid)]."""
    used, pairs = set(), []
    for seg_id, gt in gt_centroids.items():
        ref = next((gt[f] for f in range(gt.shape[0])
                    if not np.isnan(gt[f]).any()), None)
        if ref is None:
            continue
        best_rid, best_d = None, float("inf")
        for rid, rc in ripple_tracks.items():
            if rid in used or np.isnan(rc[0]).any():
                continue
            d = np.hypot(ref[0] - rc[0][0], ref[1] - rc[0][1])
            if d < best_d:
                best_d, best_rid = d, rid
        if best_rid is not None and best_d < tol:
            used.add(best_rid)
            pairs.append((seg_id, best_rid))
    return pairs


def compute_app(gt, pred, thresholds=THRESHOLDS):
    """Average point precision: fraction within tau px, pooled over valid frames."""
    valid = ~(np.isnan(gt).any(axis=1) | np.isnan(pred).any(axis=1))
    n = int(valid.sum())
    if n == 0:
        return {str(t): 0.0 for t in thresholds} | {"average": 0.0}
    d = np.sqrt(((gt[valid] - pred[valid]) ** 2).sum(axis=1))
    res = {str(t): float(np.sum(d <= t) / n * 100) for t in thresholds}
    res["average"] = float(np.mean([res[str(t)] for t in thresholds]))
    return res


def scale_coords(coords, native_hw, target=256):
    """Rescale an (N, 2) coordinate array from native resolution to target grid."""
    h, w = native_hw
    out = coords.copy()
    out[:, 0] = coords[:, 0] * (target / h)
    out[:, 1] = coords[:, 1] * (target / w)
    return out
