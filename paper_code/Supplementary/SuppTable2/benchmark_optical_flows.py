#!/usr/bin/env python3
"""
Benchmark multiple optical flow algorithms across 5 datasets.

For each algorithm × dataset × track, runs 100 random annotation-ordering trials
and finds the one crossing each APP threshold (>=90%, >=95%, >=97%, 100%) with
the fewest corrections.  Flows are computed in memory and immediately deleted.

Usage:
    python benchmark_optical_flows.py [--dataset NAME] [--algo NAME] [--readme-only]

Output:
    results/{algo_name}.json   — per-algorithm results across all datasets
    errors.json                — log of failed algorithm/dataset combos
    README.md                  — auto-generated comparison tables
"""

import argparse
import gc
import json
import os
import signal
import sys
import time
import traceback
from contextlib import contextmanager

import cv2
import numpy as np
import nrrd
import tifffile
from scipy.ndimage import center_of_mass

# ══════════════════════════════════════════════════════════════════════════════
# Paths & config
# ══════════════════════════════════════════════════════════════════════════════
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
DATA_DIR = os.path.join(ROOT_DIR, "figures_all", "data")

DATASETS = {
    "freely": {
        "tif_path": os.path.join(DATA_DIR, "freely.tif"),
        "ann_dir": os.path.join(DATA_DIR, "annotations", "freely"),
        "ann_format": "nrrd",
        "flow_ds": 1,
    },
    "neural": {
        "tif_path": os.path.join(DATA_DIR, "neural.tif"),
        "ann_dir": os.path.join(ROOT_DIR, "experiments_2_annotation_scaling",
                                "gt_tracks"),
        "ann_format": "nrrd",
        "flow_ds": 1,
    },
    "pinned": {
        "tif_path": os.path.join(DATA_DIR, "pinned.tif"),
        "ann_dir": os.path.join(DATA_DIR, "annotations", "pinned"),
        "ann_format": "nrrd",
        "flow_ds": 1,
    },
    "regen": {
        "tif_path": os.path.join(DATA_DIR, "regen.tif"),
        "ann_dir": os.path.join(DATA_DIR, "annotations", "regen"),
        "ann_format": "json",
        "ann_file": "video1_full_merge_motion_corr_unadj_annotations.json",
        "flow_ds": 4,
    },
    "sperm": {
        "tif_path": os.path.join(DATA_DIR, "sperm.tif"),
        "ann_dir": os.path.join(DATA_DIR, "annotations", "sperm"),
        "ann_format": "nrrd",
        "flow_ds": 1,
    },
}

# Process datasets in order of increasing size for quick feedback
DATASET_ORDER = ["pinned", "neural", "sperm", "freely", "regen"]

# ══════════════════════════════════════════════════════════════════════════════
# Experiment constants
# ══════════════════════════════════════════════════════════════════════════════
THRESHOLDS = [1, 2, 4, 8, 16]
N_TRIALS = 100
APP_TARGETS = [0.90, 0.95, 0.97, 1.00]
FLOW_TIMEOUT_PER_PAIR = 120  # seconds per frame pair before we skip


# ══════════════════════════════════════════════════════════════════════════════
# Timeout helper
# ══════════════════════════════════════════════════════════════════════════════
class FlowTimeoutError(Exception):
    pass


@contextmanager
def time_limit(seconds):
    """Raise FlowTimeoutError if block takes longer than `seconds`."""
    def handler(signum, frame):
        raise FlowTimeoutError(f"Exceeded {seconds}s timeout")
    old = signal.signal(signal.SIGALRM, handler)
    signal.alarm(seconds)
    try:
        yield
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old)


# ══════════════════════════════════════════════════════════════════════════════
# Core: segment_flow_blend — from figures_all/find_optimal_corrections.py
# ══════════════════════════════════════════════════════════════════════════════
def segment_flow_blend(flows, anchors, flow_ds=1):
    """Build track by blending forward/backward flow-based propagation."""
    Tm1, flow_H, flow_W, _ = flows.shape
    T = Tm1 + 1
    if not anchors:
        return np.zeros((T, 2), dtype=np.float32)
    anchors = sorted(
        set((int(t), float(x), float(y)) for (t, x, y) in anchors),
        key=lambda a: a[0],
    )
    all_pos = np.zeros((T, 2), dtype=np.float32)

    def get_flow(t, x, y):
        fx = int(np.clip(round(x / flow_ds), 0, flow_W - 1))
        fy = int(np.clip(round(y / flow_ds), 0, flow_H - 1))
        dx, dy = flows[t, fy, fx]
        return float(dx) * flow_ds, float(dy) * flow_ds

    for (t0, x0, y0), (t1, x1, y1) in zip(anchors[:-1], anchors[1:]):
        if t1 <= t0:
            continue
        seg_len = t1 - t0 + 1
        forward = np.zeros((seg_len, 2), dtype=np.float32)
        pos = np.array([x0, y0], dtype=np.float32)
        for i, t in enumerate(range(t0, t1 + 1)):
            forward[i] = pos.copy()
            if t < t1:
                dx, dy = get_flow(t, pos[0], pos[1])
                pos[0] += dx
                pos[1] += dy
        backward = np.zeros((seg_len, 2), dtype=np.float32)
        pos = np.array([x1, y1], dtype=np.float32)
        for i, t in enumerate(range(t1, t0 - 1, -1)):
            backward[seg_len - 1 - (t1 - t)] = pos.copy()
            if t > t0:
                dx, dy = get_flow(t - 1, pos[0], pos[1])
                pos[0] -= dx
                pos[1] -= dy
        alpha = np.linspace(0, 1, seg_len)
        blended = (1 - alpha[:, None]) * forward + alpha[:, None] * backward
        all_pos[t0 : t1 + 1] = blended

    t_first, x_first, y_first = anchors[0]
    if t_first > 0:
        pos = np.array([x_first, y_first], dtype=np.float32)
        for t in range(t_first - 1, -1, -1):
            dx, dy = get_flow(t, pos[0], pos[1])
            pos[0] -= dx
            pos[1] -= dy
            all_pos[t] = pos.copy()

    t_last, x_last, y_last = anchors[-1]
    pos = np.array([x_last, y_last], dtype=np.float32)
    for t in range(t_last, T):
        all_pos[t] = pos.copy()
        if t < T - 1:
            dx, dy = get_flow(t, pos[0], pos[1])
            pos[0] += dx
            pos[1] += dy
    return all_pos


# ══════════════════════════════════════════════════════════════════════════════
# Video loading
# ══════════════════════════════════════════════════════════════════════════════
def load_video_gray(name, cfg):
    """Load video and return (gray_frames, (T, H, W))."""
    tif_path = cfg["tif_path"]
    flow_ds = cfg["flow_ds"]
    print(f"  Loading {os.path.basename(tif_path)}...")
    raw = tifffile.imread(tif_path)
    T = raw.shape[0]
    H, W = raw.shape[1], raw.shape[2]
    fH, fW = H // flow_ds, W // flow_ds

    if name in ("freely", "pinned"):
        gray = np.zeros((T, fH, fW), dtype=np.uint8)
        for t in range(T):
            g = cv2.cvtColor(raw[t], cv2.COLOR_RGB2GRAY)
            if flow_ds > 1:
                g = cv2.resize(g, (fW, fH), interpolation=cv2.INTER_AREA)
            gray[t] = g

    elif name == "neural":
        bc_min = float(np.percentile(raw, 1))
        bc_max = float(np.percentile(raw, 99))
        gray = np.zeros((T, fH, fW), dtype=np.uint8)
        for t in range(T):
            frame = raw[t].astype(np.float32)
            g = np.clip((frame - bc_min) / max(1, bc_max - bc_min) * 255,
                        0, 255).astype(np.uint8)
            if flow_ds > 1:
                g = cv2.resize(g, (fW, fH), interpolation=cv2.INTER_AREA)
            gray[t] = g

    elif name == "regen":
        gray = np.zeros((T, fH, fW), dtype=np.uint8)
        for t in range(T):
            if flow_ds > 1:
                gray[t] = cv2.resize(raw[t], (fW, fH),
                                     interpolation=cv2.INTER_AREA)
            else:
                gray[t] = raw[t]

    elif name == "sperm":
        p1 = float(np.percentile(raw, 1))
        p99 = float(np.percentile(raw, 99))
        gray = np.zeros((T, fH, fW), dtype=np.uint8)
        for t in range(T):
            g = np.clip((raw[t] - p1) / max(1e-8, p99 - p1) * 255,
                        0, 255).astype(np.uint8)
            if flow_ds > 1:
                g = cv2.resize(g, (fW, fH), interpolation=cv2.INTER_AREA)
            gray[t] = g
    else:
        raise ValueError(f"Unknown dataset: {name}")

    del raw
    gc.collect()
    print(f"    Loaded: {T} frames, {H}x{W} (flow gray: {gray.shape})")
    return gray, (T, H, W)


# ══════════════════════════════════════════════════════════════════════════════
# Ground-truth loading
# ══════════════════════════════════════════════════════════════════════════════
def load_gt_nrrd(ann_dir, n_frames):
    gt_tracks = {}
    nrrd_files = sorted(
        [f for f in os.listdir(ann_dir) if f.endswith(".nrrd")],
        key=lambda f: int("".join(ch for ch in f if ch.isdigit())),
    )
    for fname in nrrd_files:
        track_id = int(
            "".join(ch for ch in fname.replace(".nrrd", "") if ch.isdigit())
        )
        data, _ = nrrd.read(os.path.join(ann_dir, fname))
        n_fr = data.shape[2] if data.ndim == 3 else data.shape[0]
        centroids = np.full((n_frames, 2), np.nan, dtype=np.float64)
        for f in range(min(n_fr, n_frames)):
            mask = data[:, :, f]
            if mask.any():
                cx, cy = center_of_mass(mask)
                centroids[f] = [cx, cy]
        gt_tracks[track_id] = centroids
    return gt_tracks


def load_gt_json(ann_dir, ann_file, n_frames):
    json_path = os.path.join(ann_dir, ann_file)
    with open(json_path) as f:
        data = json.load(f)
    gt_tracks = {}
    for track in data["tracks"]:
        tid_str = track["track_id"]
        tid = int("".join(ch for ch in tid_str if ch.isdigit()))
        centroids = np.full((n_frames, 2), np.nan, dtype=np.float64)
        for ann in track["annotations"]:
            fi = int(ann["frame"])
            if 0 <= fi < n_frames:
                centroids[fi] = [ann["x"], ann["y"]]
        gt_tracks[tid] = centroids
    return gt_tracks


def load_gt(name, cfg, n_frames):
    if cfg["ann_format"] == "nrrd":
        return load_gt_nrrd(cfg["ann_dir"], n_frames)
    elif cfg["ann_format"] == "json":
        return load_gt_json(cfg["ann_dir"], cfg["ann_file"], n_frames)
    else:
        raise ValueError(f"Unknown annotation format: {cfg['ann_format']}")


# ══════════════════════════════════════════════════════════════════════════════
# APP metric
# ══════════════════════════════════════════════════════════════════════════════
def compute_app_single_track(gt, pred):
    valid = ~np.isnan(gt[:, 0])
    if not valid.any():
        return {"average": 0.0}
    gt_v = gt[valid]
    pred_v = pred[valid]
    d = np.sqrt((gt_v[:, 0] - pred_v[:, 0])**2 +
                (gt_v[:, 1] - pred_v[:, 1])**2)
    n = len(d)
    results = {}
    for tau in THRESHOLDS:
        results[tau] = float(np.sum(d <= tau) / n)
    results["average"] = float(np.mean([results[tau] for tau in THRESHOLDS]))
    return results


# ══════════════════════════════════════════════════════════════════════════════
# Eval count selection
# ══════════════════════════════════════════════════════════════════════════════
def pick_eval_counts(max_k):
    counts = set(range(1, min(11, max_k + 1)))
    counts |= set(range(10, min(51, max_k + 1), 5))
    counts |= set(range(50, min(101, max_k + 1), 10))
    counts |= set(range(100, max_k + 1, 25))
    counts.add(max_k)
    return sorted(counts)


# ══════════════════════════════════════════════════════════════════════════════
# Core trial loop — find optimal corrections for one track
# ══════════════════════════════════════════════════════════════════════════════
def find_optimal_for_track(track_id, gt_track, flows, flow_ds, n_total_frames,
                           rng):
    valid_mask = ~np.isnan(gt_track[:, 0])
    valid_frames_all = np.where(valid_mask)[0]
    n_visible = int(len(valid_frames_all))
    if n_visible == 0:
        return None

    first_valid = int(valid_frames_all[0])
    seed_x, seed_y = gt_track[first_valid]

    valid_corrections = [
        int(f) for f in valid_frames_all if f != first_valid
    ]
    if len(valid_corrections) == 0:
        return None

    max_k = len(valid_corrections)
    eval_counts = pick_eval_counts(max_k)

    # Baseline
    rough = segment_flow_blend(
        flows, [(first_valid, seed_x, seed_y)], flow_ds=flow_ds,
    )
    baseline = compute_app_single_track(gt_track, rough)
    print(f"    Track {track_id:>2d}  baseline APP = {baseline['average']*100:.1f}%"
          f"  ({max_k} corrections, {len(eval_counts)} k-values)")

    best = {}
    for tgt in APP_TARGETS:
        best[tgt] = {
            "crossing_k": max_k + 1,
            "app": 0.0,
            "trial": -1,
            "ordering": [],
            "curve": {},
        }

    best_auc = -1.0
    best_auc_trial = -1
    best_auc_ordering = []
    best_auc_curve = {}

    for trial in range(N_TRIALS):
        perm = rng.permutation(valid_corrections).tolist()
        anchors = [(first_valid, float(seed_x), float(seed_y))]
        perm_idx = 0
        trial_curve = {}
        crossings = {tgt: None for tgt in APP_TARGETS}

        for k in eval_counts:
            while perm_idx < k:
                f = perm[perm_idx]
                gx, gy = gt_track[f]
                anchors.append((f, float(gx), float(gy)))
                perm_idx += 1

            blended = segment_flow_blend(flows, anchors, flow_ds=flow_ds)
            app = compute_app_single_track(gt_track, blended)
            trial_curve[k] = app["average"]

            for tgt in APP_TARGETS:
                if crossings[tgt] is None and app["average"] >= tgt:
                    crossings[tgt] = k

        # AUC
        ks_sorted = sorted(trial_curve.keys())
        xs = np.array([0] + ks_sorted, dtype=np.float64)
        ys = np.array([baseline["average"]] +
                      [trial_curve[k] for k in ks_sorted])
        auc = float(np.trapezoid(ys, xs))

        if auc > best_auc:
            best_auc = auc
            best_auc_trial = trial
            best_auc_ordering = perm
            best_auc_curve = trial_curve

        for tgt in APP_TARGETS:
            ck = crossings[tgt]
            if ck is not None:
                b = best[tgt]
                if (ck < b["crossing_k"] or
                    (ck == b["crossing_k"] and
                     trial_curve[ck] > b["app"])):
                    b["crossing_k"] = ck
                    b["app"] = trial_curve[ck]
                    b["trial"] = trial
                    b["ordering"] = perm
                    b["curve"] = trial_curve

        if (trial + 1) % 25 == 0:
            print(f"             trial {trial + 1}/{N_TRIALS}")

    # Build results
    results = {}
    for tgt in APP_TARGETS:
        b = best[tgt]
        if b["trial"] < 0:
            pct = int(tgt * 100)
            print(f"    WARNING: no trial crossed {pct}% APP, using best-AUC")
            best_k_app = max(best_auc_curve.items(), key=lambda x: x[1])
            results[tgt] = {
                "track_id": track_id,
                "first_valid_frame": first_valid,
                "n_total_frames": n_total_frames,
                "n_visible_frames": n_visible,
                "n_valid_corrections": max_k,
                "baseline_app": baseline["average"],
                "optimal_k": best_k_app[0],
                "optimal_app": best_k_app[1],
                "optimal_trial": best_auc_trial,
                "correction_frames": best_auc_ordering[:best_k_app[0]],
                "app_curve": {str(k): v for k, v in best_auc_curve.items()},
                "best_auc_trial": best_auc_trial,
                "best_auc": best_auc,
            }
        else:
            results[tgt] = {
                "track_id": track_id,
                "first_valid_frame": first_valid,
                "n_total_frames": n_total_frames,
                "n_visible_frames": n_visible,
                "n_valid_corrections": max_k,
                "baseline_app": baseline["average"],
                "optimal_k": b["crossing_k"],
                "optimal_app": b["app"],
                "optimal_trial": b["trial"],
                "correction_frames": b["ordering"][:b["crossing_k"]],
                "app_curve": {str(k): v for k, v in b["curve"].items()},
                "best_auc_trial": best_auc_trial,
                "best_auc": best_auc,
            }

    for tgt in APP_TARGETS:
        r = results[tgt]
        pct = int(tgt * 100)
        print(f"    Track {track_id:>2d}  @{pct}%: k={r['optimal_k']} "
              f"(APP={r['optimal_app']*100:.1f}%, trial {r['optimal_trial']})")

    return results


# ══════════════════════════════════════════════════════════════════════════════
# ═══  OPTICAL FLOW ALGORITHMS  ═══
# ══════════════════════════════════════════════════════════════════════════════

def _compute_dense_flow_cv2(gray_frames, create_fn):
    """Generic helper for OpenCV dense optical flow algorithms.

    create_fn: callable that returns a cv2.DenseOpticalFlow object
    """
    T, H, W = gray_frames.shape
    algo = create_fn()
    flows = np.zeros((T - 1, H, W, 2), dtype=np.float32)
    prev = gray_frames[0]
    for t in range(T - 1):
        curr = gray_frames[t + 1]
        flows[t] = algo.calc(prev, curr, None)
        prev = curr
        if (t + 1) % 50 == 0:
            print(f"      frame {t + 1}/{T - 1}")
    return flows


# ---------- OpenCV base ----------

def flow_dis_ultrafast(gray_frames):
    return _compute_dense_flow_cv2(
        gray_frames,
        lambda: cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_ULTRAFAST),
    )

def flow_dis_fast(gray_frames):
    return _compute_dense_flow_cv2(
        gray_frames,
        lambda: cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_FAST),
    )

def flow_dis_medium(gray_frames):
    return _compute_dense_flow_cv2(
        gray_frames,
        lambda: cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_MEDIUM),
    )

def flow_farneback(gray_frames):
    T, H, W = gray_frames.shape
    flows = np.zeros((T - 1, H, W, 2), dtype=np.float32)
    prev = gray_frames[0]
    for t in range(T - 1):
        curr = gray_frames[t + 1]
        flows[t] = cv2.calcOpticalFlowFarneback(
            prev, curr, None,
            pyr_scale=0.5, levels=3, winsize=15,
            iterations=3, poly_n=5, poly_sigma=1.2, flags=0,
        )
        prev = curr
        if (t + 1) % 50 == 0:
            print(f"      frame {t + 1}/{T - 1}")
    return flows


# ---------- OpenCV contrib optflow ----------

def flow_dual_tvl1(gray_frames):
    return _compute_dense_flow_cv2(
        gray_frames,
        lambda: cv2.optflow.createOptFlow_DualTVL1(),
    )

def flow_dense_rlof(gray_frames):
    """Dense RLOF needs 3-channel (BGR) input."""
    T, H, W = gray_frames.shape
    flows = np.zeros((T - 1, H, W, 2), dtype=np.float32)
    prev = cv2.cvtColor(gray_frames[0], cv2.COLOR_GRAY2BGR)
    for t in range(T - 1):
        curr = cv2.cvtColor(gray_frames[t + 1], cv2.COLOR_GRAY2BGR)
        flows[t] = cv2.optflow.calcOpticalFlowDenseRLOF(prev, curr, None)
        prev = curr
        if (t + 1) % 50 == 0:
            print(f"      frame {t + 1}/{T - 1}")
    return flows

def flow_deepflow(gray_frames):
    return _compute_dense_flow_cv2(
        gray_frames,
        lambda: cv2.optflow.createOptFlow_DeepFlow(),
    )

def flow_simpleflow(gray_frames):
    """SimpleFlow works on 3-channel images and uses a different API."""
    T, H, W = gray_frames.shape
    flows = np.zeros((T - 1, H, W, 2), dtype=np.float32)
    prev = cv2.cvtColor(gray_frames[0], cv2.COLOR_GRAY2BGR)
    for t in range(T - 1):
        curr = cv2.cvtColor(gray_frames[t + 1], cv2.COLOR_GRAY2BGR)
        try:
            with time_limit(FLOW_TIMEOUT_PER_PAIR):
                flow = cv2.optflow.calcOpticalFlowSF(
                    prev, curr, layers=3, averaging_block_size=2,
                    max_flow=4,
                )
        except FlowTimeoutError:
            raise FlowTimeoutError(
                f"SimpleFlow: frame {t} exceeded {FLOW_TIMEOUT_PER_PAIR}s"
            )
        flows[t] = flow
        prev = curr
        if (t + 1) % 50 == 0:
            print(f"      frame {t + 1}/{T - 1}")
    return flows

def flow_pcaflow(gray_frames):
    return _compute_dense_flow_cv2(
        gray_frames,
        lambda: cv2.optflow.createOptFlow_PCAFlow(),
    )

def flow_sparse_to_dense(gray_frames):
    return _compute_dense_flow_cv2(
        gray_frames,
        lambda: cv2.optflow.createOptFlow_SparseToDense(),
    )


# ---------- scikit-image ----------

def flow_skimage_ilk(gray_frames):
    """scikit-image ILK: returns (2, H, W) in (row, col) -> convert to (H, W, 2) in (x=col, y=row)."""
    from skimage.registration import optical_flow_ilk
    T, H, W = gray_frames.shape
    flows = np.zeros((T - 1, H, W, 2), dtype=np.float32)
    for t in range(T - 1):
        sk_flow = optical_flow_ilk(gray_frames[t], gray_frames[t + 1])
        # sk_flow[0] = row displacement (dy), sk_flow[1] = col displacement (dx)
        # OpenCV convention: flow[:,:,0] = dx, flow[:,:,1] = dy
        flows[t, :, :, 0] = sk_flow[1]  # dx = col
        flows[t, :, :, 1] = sk_flow[0]  # dy = row
        if (t + 1) % 50 == 0:
            print(f"      frame {t + 1}/{T - 1}")
    return flows

def flow_skimage_tvl1(gray_frames):
    """scikit-image TV-L1: same convention as ILK."""
    from skimage.registration import optical_flow_tvl1
    T, H, W = gray_frames.shape
    flows = np.zeros((T - 1, H, W, 2), dtype=np.float32)
    for t in range(T - 1):
        sk_flow = optical_flow_tvl1(gray_frames[t], gray_frames[t + 1])
        flows[t, :, :, 0] = sk_flow[1]
        flows[t, :, :, 1] = sk_flow[0]
        if (t + 1) % 50 == 0:
            print(f"      frame {t + 1}/{T - 1}")
    return flows


# ---------- pyoptflow (Horn-Schunck) ----------

def flow_horn_schunck(gray_frames):
    """pyoptflow Horn-Schunck: returns (U, V) where U=x, V=y displacement."""
    import pyoptflow
    T, H, W = gray_frames.shape
    flows = np.zeros((T - 1, H, W, 2), dtype=np.float32)
    for t in range(T - 1):
        im1 = gray_frames[t].astype(np.float32) / 255.0
        im2 = gray_frames[t + 1].astype(np.float32) / 255.0
        U, V = pyoptflow.HornSchunck(im1, im2, alpha=1.0, Niter=100)
        flows[t, :, :, 0] = U  # x displacement
        flows[t, :, :, 1] = V  # y displacement
        if (t + 1) % 50 == 0:
            print(f"      frame {t + 1}/{T - 1}")
    return flows


# ---------- torchvision RAFT ----------

def _torchvision_raft(gray_frames, model_fn, weights):
    """Generic RAFT wrapper via torchvision."""
    import torch
    from torchvision.transforms.functional import normalize

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = model_fn(weights=weights).to(device).eval()

    T, H, W = gray_frames.shape
    # Pad to multiple of 8
    pad_h = (8 - H % 8) % 8
    pad_w = (8 - W % 8) % 8

    flows = np.zeros((T - 1, H, W, 2), dtype=np.float32)

    with torch.no_grad():
        for t in range(T - 1):
            # Convert to 3-channel float tensor [0, 1] -> normalize for RAFT
            im1 = gray_frames[t].astype(np.float32)
            im2 = gray_frames[t + 1].astype(np.float32)
            # RAFT expects [B, 3, H, W] in [0, 255]
            t1 = torch.from_numpy(im1).unsqueeze(0).unsqueeze(0).repeat(1, 3, 1, 1).to(device)
            t2 = torch.from_numpy(im2).unsqueeze(0).unsqueeze(0).repeat(1, 3, 1, 1).to(device)

            # Normalize as expected by torchvision RAFT
            t1 = normalize(t1 / 255.0, mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
            t2 = normalize(t2 / 255.0, mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])

            # Pad
            if pad_h > 0 or pad_w > 0:
                t1 = torch.nn.functional.pad(t1, (0, pad_w, 0, pad_h), mode='replicate')
                t2 = torch.nn.functional.pad(t2, (0, pad_w, 0, pad_h), mode='replicate')

            pred = model(t1, t2)
            flow_t = pred[-1]  # last iteration
            flow_np = flow_t[0].cpu().numpy()  # (2, H+pad, W+pad)
            # flow_np[0] = dx, flow_np[1] = dy (matches OpenCV convention)
            flows[t, :, :, 0] = flow_np[0, :H, :W]
            flows[t, :, :, 1] = flow_np[1, :H, :W]

            if (t + 1) % 50 == 0:
                print(f"      frame {t + 1}/{T - 1}")

    del model
    torch.cuda.empty_cache()
    return flows


def flow_tv_raft_large(gray_frames):
    from torchvision.models.optical_flow import raft_large, Raft_Large_Weights
    return _torchvision_raft(gray_frames, raft_large, Raft_Large_Weights.DEFAULT)

def flow_tv_raft_small(gray_frames):
    from torchvision.models.optical_flow import raft_small, Raft_Small_Weights
    return _torchvision_raft(gray_frames, raft_small, Raft_Small_Weights.DEFAULT)


# ---------- ptlflow models ----------

def _ptlflow_compute(gray_frames, model_name, ckpt_name):
    """Generic ptlflow model wrapper."""
    import torch
    import ptlflow

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = ptlflow.get_model(model_name, ckpt_path=ckpt_name)
    model = model.to(device).eval()

    T, H, W = gray_frames.shape
    # Pad to multiple of 8 (some models need 32 or 64 — ptlflow handles this
    # internally in many cases, but we'll pad to 64 to be safe)
    pad_h = (64 - H % 64) % 64
    pad_w = (64 - W % 64) % 64

    flows = np.zeros((T - 1, H, W, 2), dtype=np.float32)

    with torch.no_grad():
        for t in range(T - 1):
            im1 = gray_frames[t].astype(np.float32)
            im2 = gray_frames[t + 1].astype(np.float32)
            # ptlflow expects dict with 'images': tensor of shape [B, 2, C, H, W]
            t1 = torch.from_numpy(im1).unsqueeze(0).unsqueeze(0).repeat(1, 3, 1, 1)  # [1, 3, H, W]
            t2 = torch.from_numpy(im2).unsqueeze(0).unsqueeze(0).repeat(1, 3, 1, 1)
            images = torch.stack([t1, t2], dim=1).to(device)  # [1, 2, 3, H, W]

            # Pad
            if pad_h > 0 or pad_w > 0:
                images = torch.nn.functional.pad(
                    images.view(1, 2 * 3, H, W),
                    (0, pad_w, 0, pad_h),
                    mode='replicate',
                ).view(1, 2, 3, H + pad_h, W + pad_w)

            inputs = {"images": images}
            preds = model(inputs)
            flow_t = preds["flows"]  # typically [B, 2, H, W]
            if flow_t.dim() == 5:
                flow_t = flow_t[:, 0]  # take first flow if multi-flow
            flow_np = flow_t[0].cpu().numpy()  # (2, H+pad, W+pad)
            flows[t, :, :, 0] = flow_np[0, :H, :W]
            flows[t, :, :, 1] = flow_np[1, :H, :W]

            if (t + 1) % 50 == 0:
                print(f"      frame {t + 1}/{T - 1}")

    del model
    torch.cuda.empty_cache()
    return flows


def _make_ptlflow_fn(model_name, ckpt_name):
    """Factory for ptlflow algorithm functions."""
    def fn(gray_frames):
        return _ptlflow_compute(gray_frames, model_name, ckpt_name)
    fn.__name__ = f"ptl_{model_name}"
    fn.__doc__ = f"ptlflow: {model_name} ({ckpt_name})"
    return fn


# ══════════════════════════════════════════════════════════════════════════════
# Algorithm registry
# ══════════════════════════════════════════════════════════════════════════════

# Each entry: (display_name, package, function)
# Use 'things' checkpoint for ptlflow by default (general-purpose)
# If 'things' not available, use first available checkpoint

PTLFLOW_MODELS = [
    ("raft", "things"),
    ("raft_small", "things"),
    ("gma", "things"),
    ("flowformer", "things"),
    ("flowformer_pp", "things"),
    ("gmflow", "things"),
    ("unimatch", "things"),          # GMFlow+ / UniMatch
    ("sea_raft_l", "things"),
    ("ms_raft_p", "mixed"),
    ("neuflow2", "things"),
    ("ccmr", "sintel"),              # no 'things'
    ("craft", "things"),
    ("memflow", "things"),
    ("skflow", "things"),
    ("rpknet", "things"),
    ("rapidflow", "things"),
    ("videoflow_bof", "things_288960"),
    ("pwcnet", "things"),
    ("liteflownet", "things"),
    ("maskflownet_s", "things"),
    ("flownets", "things"),
    ("flownet2", "things"),
    ("irr_pwc", "things"),
    ("separableflow", "things"),
    ("starflow", "things"),
    ("dip", "things"),
    ("matchflow", "things"),
    ("scopeflow", "things"),
    ("streamflow", "things"),
    ("dpflow", "things"),
    ("flow1d", "things"),
    ("fastflownet", "things"),
    ("csflow", "things"),
    ("llaflow", "things"),
    ("gmflownet", "things"),
    ("dicl", "things"),
]

ALGORITHMS = []

# OpenCV base
ALGORITHMS.append(("dis_ultrafast", "opencv", flow_dis_ultrafast))
ALGORITHMS.append(("dis_fast", "opencv", flow_dis_fast))
ALGORITHMS.append(("dis_medium", "opencv", flow_dis_medium))
ALGORITHMS.append(("farneback", "opencv", flow_farneback))

# OpenCV contrib
ALGORITHMS.append(("dual_tvl1", "opencv_contrib", flow_dual_tvl1))
ALGORITHMS.append(("dense_rlof", "opencv_contrib", flow_dense_rlof))
ALGORITHMS.append(("deepflow", "opencv_contrib", flow_deepflow))
ALGORITHMS.append(("simpleflow", "opencv_contrib", flow_simpleflow))
ALGORITHMS.append(("pcaflow", "opencv_contrib", flow_pcaflow))
ALGORITHMS.append(("sparse_to_dense", "opencv_contrib", flow_sparse_to_dense))

# scikit-image
ALGORITHMS.append(("skimage_ilk", "scikit-image", flow_skimage_ilk))
ALGORITHMS.append(("skimage_tvl1", "scikit-image", flow_skimage_tvl1))

# pyoptflow
ALGORITHMS.append(("horn_schunck", "pyoptflow", flow_horn_schunck))

# torchvision
ALGORITHMS.append(("tv_raft_large", "torchvision", flow_tv_raft_large))
ALGORITHMS.append(("tv_raft_small", "torchvision", flow_tv_raft_small))

# ptlflow
for pmodel, pckpt in PTLFLOW_MODELS:
    ALGORITHMS.append((
        f"ptl_{pmodel}",
        "ptlflow",
        _make_ptlflow_fn(pmodel, pckpt),
    ))


# ══════════════════════════════════════════════════════════════════════════════
# Results I/O
# ══════════════════════════════════════════════════════════════════════════════
RESULTS_DIR = os.path.join(BASE_DIR, "results")
ERRORS_PATH = os.path.join(BASE_DIR, "errors.json")


def load_algo_results(algo_name):
    path = os.path.join(RESULTS_DIR, f"{algo_name}.json")
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return None


def save_algo_results(algo_name, data):
    os.makedirs(RESULTS_DIR, exist_ok=True)
    path = os.path.join(RESULTS_DIR, f"{algo_name}.json")
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


def load_errors():
    if os.path.exists(ERRORS_PATH):
        with open(ERRORS_PATH) as f:
            return json.load(f)
    return []


def save_error(errors, algo_name, dataset_name, error_msg):
    errors.append({
        "algorithm": algo_name,
        "dataset": dataset_name,
        "error": error_msg,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    })
    with open(ERRORS_PATH, "w") as f:
        json.dump(errors, f, indent=2)


# ══════════════════════════════════════════════════════════════════════════════
# Process one algorithm × one dataset
# ══════════════════════════════════════════════════════════════════════════════
def process_algo_dataset(algo_name, pkg, compute_fn, gray_frames, gt_tracks,
                         flow_ds, n_total_frames, rng_state):
    """Compute flows, run trials on all tracks, return results dict.

    rng_state: the rng state BEFORE processing this dataset — we restore it
               so all algorithms get the exact same random orderings.
    """
    print(f"\n    --- {algo_name} ({pkg}) ---")
    t0 = time.time()

    # Compute flows
    print(f"    Computing optical flow...")
    flows = compute_fn(gray_frames)
    elapsed = time.time() - t0
    print(f"    Flow computed in {elapsed:.1f}s  shape={flows.shape}")

    # If flow_ds > 1, the gray frames were already downsampled, so flows match
    # But segment_flow_blend needs flow_ds to scale back to image coords
    # The flows are in downsampled pixel space, matching gray_frames shape

    # Run trials for each track
    results_by_target = {tgt: {} for tgt in APP_TARGETS}
    for tid, gt in sorted(gt_tracks.items()):
        # Restore RNG for each track so orderings match across algorithms
        rng = np.random.default_rng(seed=42)
        # Advance RNG state to match the right position
        # We need a deterministic RNG per (dataset, track) — use track-specific seed
        rng = np.random.default_rng(seed=42 + tid * 1000)

        multi = find_optimal_for_track(
            tid, gt, flows, flow_ds, n_total_frames, rng,
        )
        if multi is not None:
            for tgt in APP_TARGETS:
                results_by_target[tgt][str(tid)] = multi[tgt]

    # Immediately delete flows
    del flows
    gc.collect()

    total_elapsed = time.time() - t0
    print(f"    {algo_name} on dataset done in {total_elapsed:.1f}s")

    return results_by_target


# ══════════════════════════════════════════════════════════════════════════════
# README generation
# ══════════════════════════════════════════════════════════════════════════════
def generate_readme():
    """Generate README.md with comparison tables from all result JSONs."""
    print("\nGenerating README.md...")

    # Load all results
    all_results = {}  # algo_name -> data
    for fname in sorted(os.listdir(RESULTS_DIR)):
        if not fname.endswith(".json"):
            continue
        algo_name = fname.replace(".json", "")
        with open(os.path.join(RESULTS_DIR, fname)) as f:
            all_results[algo_name] = json.load(f)

    if not all_results:
        print("  No results found!")
        return

    # Find which algorithms have which package
    algo_pkg = {}
    for aname, pkg, _ in ALGORITHMS:
        algo_pkg[aname] = pkg

    lines = []
    lines.append("# Optical Flow Algorithm Benchmark\n")
    lines.append(f"**{len(all_results)} algorithms** tested across "
                 f"**{len(DATASET_ORDER)} datasets** "
                 f"(100 random trials, seed=42+tid*1000)\n")
    lines.append("Each cell shows **k** = number of correction annotations needed "
                 "to reach the APP threshold.\n")
    lines.append("Lower k = better algorithm (fewer corrections needed).\n")

    # For each threshold, produce a summary table
    for tgt in APP_TARGETS:
        pct = int(tgt * 100)
        lines.append(f"\n## APP >= {pct}%\n")

        # Collect: algo -> dataset -> median_k across tracks
        table_data = {}  # algo -> {dataset -> median_k_str}
        for algo_name, data in all_results.items():
            table_data[algo_name] = {}
            datasets_in = data.get("datasets", {})
            for dname in DATASET_ORDER:
                if dname not in datasets_in:
                    table_data[algo_name][dname] = "—"
                    continue
                tracks = datasets_in[dname]
                tgt_key = str(tgt)
                ks = []
                for tid, tdata in tracks.items():
                    if tgt_key in tdata:
                        ks.append(tdata[tgt_key]["optimal_k"])
                if ks:
                    median_k = int(np.median(ks))
                    table_data[algo_name][dname] = str(median_k)
                else:
                    table_data[algo_name][dname] = "—"

        # Compute average (mean of medians, excluding missing)
        for algo_name in table_data:
            vals = []
            for dname in DATASET_ORDER:
                v = table_data[algo_name][dname]
                if v != "—":
                    vals.append(int(v))
            if vals:
                table_data[algo_name]["avg"] = f"{np.mean(vals):.0f}"
            else:
                table_data[algo_name]["avg"] = "—"

        # Sort by average k (ascending)
        def sort_key(algo_name):
            v = table_data[algo_name]["avg"]
            return float(v) if v != "—" else 9999
        sorted_algos = sorted(table_data.keys(), key=sort_key)

        # Build table
        header = "| # | Algorithm | Package |"
        for dname in DATASET_ORDER:
            header += f" {dname} |"
        header += " **Avg** |"
        lines.append(header)

        sep = "|---|-----------|---------|"
        for _ in DATASET_ORDER:
            sep += "---:|"
        sep += "---:|"
        lines.append(sep)

        for rank, algo_name in enumerate(sorted_algos, 1):
            pkg = algo_pkg.get(algo_name, "?")
            row = f"| {rank} | {algo_name} | {pkg} |"
            for dname in DATASET_ORDER:
                v = table_data[algo_name][dname]
                row += f" {v} |"
            row += f" **{table_data[algo_name]['avg']}** |"
            lines.append(row)

        lines.append("")

    # ---- Per-dataset best algorithm ----
    lines.append("\n## Best Algorithm Per Dataset\n")
    for tgt in APP_TARGETS:
        pct = int(tgt * 100)
        lines.append(f"\n### APP >= {pct}%\n")
        lines.append("| Dataset | Best Algorithm | Median k | Package |")
        lines.append("|---------|---------------|----------|---------|")
        for dname in DATASET_ORDER:
            best_algo = None
            best_k = 9999
            for algo_name, data in all_results.items():
                datasets_in = data.get("datasets", {})
                if dname not in datasets_in:
                    continue
                tracks = datasets_in[dname]
                tgt_key = str(tgt)
                ks = []
                for tid, tdata in tracks.items():
                    if tgt_key in tdata:
                        ks.append(tdata[tgt_key]["optimal_k"])
                if ks:
                    mk = int(np.median(ks))
                    if mk < best_k:
                        best_k = mk
                        best_algo = algo_name
            if best_algo:
                pkg = algo_pkg.get(best_algo, "?")
                lines.append(f"| {dname} | {best_algo} | {best_k} | {pkg} |")
            else:
                lines.append(f"| {dname} | — | — | — |")
        lines.append("")

    # ---- Per-track detail ----
    lines.append("\n## Per-Track Detail\n")
    for tgt in APP_TARGETS:
        pct = int(tgt * 100)
        lines.append(f"\n### APP >= {pct}%\n")
        for dname in DATASET_ORDER:
            lines.append(f"\n#### {dname}\n")
            # Find all tracks in this dataset
            all_tids = set()
            for algo_name, data in all_results.items():
                datasets_in = data.get("datasets", {})
                if dname in datasets_in:
                    for tid in datasets_in[dname]:
                        all_tids.add(tid)
            if not all_tids:
                lines.append("No results.\n")
                continue
            sorted_tids = sorted(all_tids, key=lambda x: int(x))

            header = "| Algorithm |"
            for tid in sorted_tids:
                header += f" T{tid} |"
            lines.append(header)
            sep = "|-----------|"
            for _ in sorted_tids:
                sep += "---:|"
            lines.append(sep)

            # Sort algos by mean k for this dataset
            algo_mean_k = {}
            for algo_name, data in all_results.items():
                datasets_in = data.get("datasets", {})
                if dname not in datasets_in:
                    continue
                tracks = datasets_in[dname]
                tgt_key = str(tgt)
                ks = []
                for tid in sorted_tids:
                    if tid in tracks and tgt_key in tracks[tid]:
                        ks.append(tracks[tid][tgt_key]["optimal_k"])
                algo_mean_k[algo_name] = np.mean(ks) if ks else 9999

            for algo_name in sorted(algo_mean_k, key=algo_mean_k.get):
                data = all_results[algo_name]
                tracks = data.get("datasets", {}).get(dname, {})
                tgt_key = str(tgt)
                row = f"| {algo_name} |"
                for tid in sorted_tids:
                    if tid in tracks and tgt_key in tracks[tid]:
                        k = tracks[tid][tgt_key]["optimal_k"]
                        row += f" {k} |"
                    else:
                        row += " — |"
                lines.append(row)
            lines.append("")

    # ---- Baseline APP comparison ----
    lines.append("\n## Baseline APP (single-anchor, no corrections)\n")
    lines.append("Shows how well each algorithm propagates from just one annotation.\n")
    for dname in DATASET_ORDER:
        lines.append(f"\n### {dname}\n")
        all_tids = set()
        for algo_name, data in all_results.items():
            datasets_in = data.get("datasets", {})
            if dname in datasets_in:
                for tid in datasets_in[dname]:
                    all_tids.add(tid)
        if not all_tids:
            lines.append("No results.\n")
            continue
        sorted_tids = sorted(all_tids, key=lambda x: int(x))

        header = "| Algorithm |"
        for tid in sorted_tids:
            header += f" T{tid} |"
        header += " Avg |"
        lines.append(header)
        sep = "|-----------|"
        for _ in sorted_tids:
            sep += "---:|"
        sep += "---:|"
        lines.append(sep)

        # Collect baselines
        algo_baselines = {}
        for algo_name, data in all_results.items():
            tracks = data.get("datasets", {}).get(dname, {})
            if not tracks:
                continue
            baselines = {}
            # Use first available target key to get baseline_app
            for tid in sorted_tids:
                if tid in tracks:
                    for tgt_key in [str(t) for t in APP_TARGETS]:
                        if tgt_key in tracks[tid]:
                            baselines[tid] = tracks[tid][tgt_key]["baseline_app"]
                            break
            algo_baselines[algo_name] = baselines

        # Sort by avg baseline descending
        def avg_baseline(algo):
            vals = list(algo_baselines[algo].values())
            return np.mean(vals) if vals else 0
        for algo_name in sorted(algo_baselines, key=avg_baseline, reverse=True):
            baselines = algo_baselines[algo_name]
            row = f"| {algo_name} |"
            vals = []
            for tid in sorted_tids:
                if tid in baselines:
                    v = baselines[tid] * 100
                    row += f" {v:.1f}% |"
                    vals.append(baselines[tid])
                else:
                    row += " — |"
            avg = np.mean(vals) * 100 if vals else 0
            row += f" {avg:.1f}% |"
            lines.append(row)
        lines.append("")

    # ---- Flow computation time ----
    lines.append("\n## Flow Computation Times\n")
    lines.append("| Algorithm | Package |")
    for dname in DATASET_ORDER:
        lines.append(f" {dname} |")
    # This info is in the results JSON if we store it
    lines.append("")
    lines.append("(Times recorded during benchmark execution — check results JSONs for details)\n")

    # ---- Errors ----
    if os.path.exists(ERRORS_PATH):
        with open(ERRORS_PATH) as f:
            errors = json.load(f)
        if errors:
            lines.append("\n## Errors\n")
            lines.append("| Algorithm | Dataset | Error |")
            lines.append("|-----------|---------|-------|")
            for e in errors:
                # Truncate long error messages
                err_msg = e["error"][:100].replace("|", "/")
                lines.append(f"| {e['algorithm']} | {e['dataset']} | {err_msg} |")
            lines.append("")

    # ---- Method descriptions ----
    lines.append("\n## Algorithm Descriptions\n")
    lines.append("| Package | Algorithm | Description |")
    lines.append("|---------|-----------|-------------|")
    descs = {
        "dis_ultrafast": "DIS (Dense Inverse Search) — ultrafast preset",
        "dis_fast": "DIS — fast preset",
        "dis_medium": "DIS — medium preset (default in Ripple)",
        "farneback": "Gunnar Farnebäck's polynomial expansion method",
        "dual_tvl1": "Dual TV-L1 variational method",
        "dense_rlof": "Dense Robust Local Optical Flow",
        "deepflow": "DeepFlow — large displacement matching + variational",
        "simpleflow": "SimpleFlow — fast dense flow (Tao et al. 2012)",
        "pcaflow": "PCA-Flow — PCA-based prior on flow fields",
        "sparse_to_dense": "Sparse-to-Dense interpolation flow",
        "skimage_ilk": "Iterative Lucas-Kanade (scikit-image)",
        "skimage_tvl1": "TV-L1 variational (scikit-image)",
        "horn_schunck": "Horn-Schunck global variational method",
        "tv_raft_large": "RAFT large (torchvision, Teed & Deng 2020)",
        "tv_raft_small": "RAFT small (torchvision, lighter variant)",
    }
    ptl_descs = {
        "raft": "RAFT (ECCV 2020)",
        "raft_small": "RAFT small (ECCV 2020)",
        "gma": "GMA — Global Motion Aggregation (ICCV 2021)",
        "flowformer": "FlowFormer — transformer-based (ECCV 2022)",
        "flowformer_pp": "FlowFormer++ (TPAMI 2023)",
        "gmflow": "GMFlow — Global Matching Flow (CVPR 2022)",
        "unimatch": "UniMatch / GMFlow+ (TPAMI 2023)",
        "sea_raft_l": "SEA-RAFT large (ECCV 2024)",
        "ms_raft_p": "MS-RAFT+ multi-scale (CVPR 2022)",
        "neuflow2": "NeuFlow v2 — efficient neural flow",
        "ccmr": "CCMR — cost volume cross-attn (CVPR 2024)",
        "craft": "CRAFT — cross-attentional flow (ICCV 2023)",
        "memflow": "MemFlow — memory-based flow (ECCV 2024)",
        "skflow": "SKFlow — selective kernel flow",
        "rpknet": "RPKNet — recurrent prediction kernel",
        "rapidflow": "RAPIDFlow — efficient recurrent flow",
        "videoflow_bof": "VideoFlow (bi-directional, ICCV 2023)",
        "pwcnet": "PWC-Net (CVPR 2018)",
        "liteflownet": "LiteFlowNet (CVPR 2018)",
        "maskflownet_s": "MaskFlownet-S (CVPR 2020)",
        "flownets": "FlowNetS (ICCV 2015)",
        "flownet2": "FlowNet 2.0 (CVPR 2017)",
        "irr_pwc": "IRR-PWC (CVPR 2019)",
        "separableflow": "SeparableFlow (ICCV 2021)",
        "starflow": "STaRFlow (ECCV 2020)",
        "dip": "DIP — deep inverse patchmatch",
        "matchflow": "MatchFlow (CVPR 2023)",
        "scopeflow": "ScopeFlow (ECCV 2020)",
        "streamflow": "StreamFlow — streaming flow",
        "dpflow": "DPFlow — dynamic partition flow",
        "flow1d": "Flow1D — 1D attention flow",
        "fastflownet": "FastFlowNet — lightweight fast flow",
        "csflow": "CSFlow — cross-strip flow",
        "llaflow": "LLA-Flow — local-level attention",
        "gmflownet": "GMFlowNet — global matching + FlowNet",
        "dicl": "DICL — displacement-invariant cost learning",
    }
    for aname, pkg, _ in ALGORITHMS:
        desc = descs.get(aname, "")
        if aname.startswith("ptl_"):
            mname = aname[4:]
            desc = ptl_descs.get(mname, f"ptlflow: {mname}")
        lines.append(f"| {pkg} | {aname} | {desc} |")
    lines.append("")

    readme_path = os.path.join(BASE_DIR, "README.md")
    with open(readme_path, "w") as f:
        f.write("\n".join(lines))
    print(f"README saved to {readme_path}")


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════
def main():
    parser = argparse.ArgumentParser(
        description="Benchmark optical flow algorithms across datasets."
    )
    parser.add_argument(
        "--dataset", "-d", type=str, default=None,
        help=f"Process only this dataset. Choices: {', '.join(DATASET_ORDER)}",
    )
    parser.add_argument(
        "--algo", "-a", type=str, default=None,
        help="Process only this algorithm (by name).",
    )
    parser.add_argument(
        "--readme-only", action="store_true",
        help="Only regenerate README from existing results.",
    )
    parser.add_argument(
        "--list-algos", action="store_true",
        help="List all registered algorithms and exit.",
    )
    args = parser.parse_args()

    if args.list_algos:
        print(f"Registered algorithms ({len(ALGORITHMS)}):")
        for aname, pkg, _ in ALGORITHMS:
            print(f"  {aname:30s}  ({pkg})")
        return

    if args.readme_only:
        generate_readme()
        return

    # Filter datasets
    if args.dataset:
        if args.dataset not in DATASETS:
            print(f"Unknown dataset: {args.dataset}. "
                  f"Available: {list(DATASETS.keys())}")
            sys.exit(1)
        dataset_names = [args.dataset]
    else:
        dataset_names = DATASET_ORDER

    # Filter algorithms
    if args.algo:
        algo_list = [(n, p, f) for n, p, f in ALGORITHMS if n == args.algo]
        if not algo_list:
            print(f"Unknown algorithm: {args.algo}")
            print("Available:", [n for n, _, _ in ALGORITHMS])
            sys.exit(1)
    else:
        algo_list = ALGORITHMS

    errors = load_errors()

    for dname in dataset_names:
        cfg = DATASETS[dname]
        print(f"\n{'=' * 70}")
        print(f"DATASET: {dname}")
        print(f"{'=' * 70}")

        gray, (T, H, W) = load_video_gray(dname, cfg)
        gt_tracks = load_gt(dname, cfg, T)
        print(f"  {len(gt_tracks)} tracks: {sorted(gt_tracks.keys())}")

        for algo_name, pkg, compute_fn in algo_list:
            # Check if already computed
            existing = load_algo_results(algo_name)
            if existing and dname in existing.get("datasets", {}):
                print(f"\n    --- {algo_name}: {dname} already done, skipping ---")
                continue

            # Check if previously errored
            prev_errors = [e for e in errors
                           if e["algorithm"] == algo_name
                           and e["dataset"] == dname]
            if prev_errors:
                print(f"\n    --- {algo_name}: {dname} previously errored, skipping ---")
                continue

            try:
                results_by_target = process_algo_dataset(
                    algo_name, pkg, compute_fn,
                    gray, gt_tracks, cfg["flow_ds"], T,
                    rng_state=None,
                )

                # Build / update result file
                if existing is None:
                    existing = {
                        "algorithm": algo_name,
                        "package": pkg,
                        "datasets": {},
                    }
                ds_data = {}
                for tid_str, _ in sorted(gt_tracks.items()):
                    tid_s = str(tid_str)
                    track_entry = {}
                    for tgt in APP_TARGETS:
                        tgt_key = str(tgt)
                        if tid_s in results_by_target[tgt]:
                            track_entry[tgt_key] = results_by_target[tgt][tid_s]
                    if track_entry:
                        ds_data[tid_s] = track_entry
                existing["datasets"][dname] = ds_data
                save_algo_results(algo_name, existing)
                print(f"    Saved results/{algo_name}.json")

            except Exception as e:
                tb = traceback.format_exc()
                err_msg = f"{type(e).__name__}: {e}"
                print(f"\n    !!! ERROR: {algo_name} on {dname}: {err_msg}")
                print(tb)
                save_error(errors, algo_name, dname, err_msg)

            gc.collect()
            # Also clear CUDA cache if torch is available
            try:
                import torch
                torch.cuda.empty_cache()
            except ImportError:
                pass

        del gray
        gc.collect()

    # Generate README
    generate_readme()

    # Print summary
    n_results = len([f for f in os.listdir(RESULTS_DIR) if f.endswith(".json")])
    n_errors = len(errors)
    print(f"\n{'=' * 70}")
    print(f"BENCHMARK COMPLETE")
    print(f"  {n_results} algorithm result files")
    print(f"  {n_errors} errors logged")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
