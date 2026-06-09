#!/usr/bin/env python3
"""
Benchmark tracking methods against ground truth using Average Point Precision (APP).

Methods compared:
  1. LocoTrack   — deep learning point tracker (fully automatic)
  2. Ripple      — semi-automated annotation (anchor-based interpolation)
  3. TrackMate   — classical detection + linking (fully automatic)
  4. Manual Annotation — frame-by-frame labeling (= ground truth, 100% by definition)

Ground truth: Centroids of binary segmentation volumes stored as NRRD files.

Metric:
  APP at thresholds τ ∈ {1, 2, 4, 8, 16} pixels:

    d(p, p̂) = sqrt( (x − x̂)² + (y − ŷ)² )
    APP@τ   = (1/N) Σ 𝟙[ d(pᵢ, p̂ᵢ) ≤ τ ]

Plots produced (5 scatter plots):
  1. APP vs. Number of Parameters to Tune
  2. APP vs. Number of Manual Annotations
  3. APP vs. Computational Time (optical flow NOT cached)
  4. APP vs. Computational Time (optical flow cached)
  5. APP vs. Real Time
"""

import csv
import glob
import json
import os
import re
import xml.etree.ElementTree as ET

import matplotlib.pyplot as plt
import numpy as np
import nrrd
from matplotlib.lines import Line2D
from scipy.ndimage import center_of_mass

# ═══════════════════════════════════════════════════════
# Constants
# ═══════════════════════════════════════════════════════

W, H = 100, 100                   # Video resolution (pixels)
N_FRAMES = 400                     # Total frames in video
THRESHOLDS = [1, 2, 4, 8, 16]     # APP thresholds (pixels)

BASE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'pinned_down_jelly')

GT_FILES = {
    1: os.path.join(BASE_DIR, 'ground_truth_tracks', 'pinned_down_jelly_1.nrrd'),
    2: os.path.join(BASE_DIR, 'ground_truth_tracks', 'pinned_down_jelly_2.nrrd'),
    3: os.path.join(BASE_DIR, 'ground_truth_tracks', 'pinned_down_jelly_3.nrrd.nrrd'),
}

# Query points shared across all methods: track_id → (x, y) at frame 0
QUERY_POINTS = {1: (68, 64), 2: (63, 59), 3: (18, 44)}


# ═══════════════════════════════════════════════════════
# Data Loaders
# ═══════════════════════════════════════════════════════

def load_ground_truth() -> dict[int, np.ndarray]:
    """Load ground-truth centroids from NRRD binary segmentation volumes.

    Each NRRD file has shape (100, 100, 400) — axes (x, y, frame).
    The centroid (center of mass) of the foreground region is computed
    per frame to yield the ground-truth track.

    Returns
    -------
    dict[int, np.ndarray]
        Mapping track_id → (N_FRAMES, 2) array of (x, y) centroids.
    """
    gt_tracks = {}
    for track_id, filepath in GT_FILES.items():
        data, _ = nrrd.read(filepath)
        centroids = np.zeros((N_FRAMES, 2))
        for f in range(N_FRAMES):
            frame_mask = data[:, :, f]
            if frame_mask.any():
                cx, cy = center_of_mass(frame_mask)
                centroids[f] = [cx, cy]
            else:
                centroids[f] = [np.nan, np.nan]
        gt_tracks[track_id] = centroids
    return gt_tracks


def load_locotrack() -> tuple[dict[int, np.ndarray], float]:
    """Load LocoTrack predictions from CSV.

    Returns
    -------
    tuple[dict[int, np.ndarray], float]
        (tracks dict, total_inference_time_seconds).
    """
    csv_path = os.path.join(BASE_DIR, 'locotrack_tracks', 'pinned_down_jelly_tracks.csv')
    tracks: dict[int, np.ndarray] = {}
    inference_time = None

    with open(csv_path) as f:
        for line in f:
            line = line.strip()
            if line.startswith('# total_inference_time_s:'):
                inference_time = float(line.split(':')[1].strip())
                continue
            if line.startswith('#') or line.startswith('video'):
                continue
            parts = line.split(',')
            track_id = int(parts[1])
            frame = int(parts[2])
            x, y = float(parts[3]), float(parts[4])
            if track_id not in tracks:
                tracks[track_id] = np.zeros((N_FRAMES, 2))
            tracks[track_id][frame] = [x, y]

    return tracks, inference_time


def load_ripple() -> tuple[dict[int, np.ndarray], int, dict]:
    """Load Ripple dense annotations and anchor counts from JSON.

    The JSON contains both sparse ``anchors`` (user-placed keyframes) and
    dense ``annotations`` (400 per track, after optimization/interpolation).
    We evaluate accuracy on the dense annotations, but report the anchor
    count as the number of manual annotations.

    Returns
    -------
    tuple[dict[int, np.ndarray], int, dict]
        (tracks dict, total_anchor_count, metadata dict).
    """
    json_path = os.path.join(BASE_DIR, 'ripple_tracks', 'pinned_down_jelly.json')
    with open(json_path) as f:
        data = json.load(f)

    metadata = data['metadata']
    tracks: dict[int, np.ndarray] = {}
    total_anchors = 0

    for td in data['tracks']:
        total_anchors += len(td['anchors'])

        # Match track to a query point by starting coordinates
        start_x = td['annotations'][0]['x']
        start_y = td['annotations'][0]['y']
        matched_id = None
        for qid, (qx, qy) in QUERY_POINTS.items():
            if abs(start_x - qx) < 2 and abs(start_y - qy) < 2:
                matched_id = qid
                break

        if matched_id is None:
            raise ValueError(
                f"Could not match Ripple track '{td['track_id']}' "
                f"starting at ({start_x}, {start_y}) to any query point"
            )

        coords = np.zeros((N_FRAMES, 2))
        for ann in td['annotations']:
            coords[ann['frame']] = [ann['x'], ann['y']]
        tracks[matched_id] = coords

    return tracks, total_anchors, metadata


def load_trackmate() -> tuple[dict[int, np.ndarray], dict[int, int]]:
    """Load TrackMate tracks from XML, auto-matching to query points.

    Among the 187 particles in the XML, the three closest to the query
    points at frame 0 are selected.  Missing frames are filled with NaN
    (and will score 0 in the APP metric).

    Returns
    -------
    tuple[dict[int, np.ndarray], dict[int, int]]
        (tracks dict with NaN for missing frames, track_lengths dict).
    """
    xml_path = os.path.join(BASE_DIR, 'trackmate_tracks', 'pinned_down_jelly.xml')
    tree = ET.parse(xml_path)
    root = tree.getroot()

    # Parse all particles
    particles: list[list[tuple[int, float, float]]] = []
    for particle in root.findall('particle'):
        detections = []
        for det in particle.findall('detection'):
            t = int(det.get('t'))
            x = float(det.get('x'))
            y = float(det.get('y'))
            detections.append((t, x, y))
        particles.append(detections)

    # Match particles to query points by proximity at frame 0
    matched: dict[int, tuple[int, list]] = {}
    used_indices: set[int] = set()
    for qid, (qx, qy) in QUERY_POINTS.items():
        best_dist = float('inf')
        best_idx = -1
        for i, dets in enumerate(particles):
            if i in used_indices:
                continue
            for t, x, y in dets:
                if t == 0:
                    d = np.sqrt((x - qx) ** 2 + (y - qy) ** 2)
                    if d < best_dist:
                        best_dist = d
                        best_idx = i
                    break
        used_indices.add(best_idx)
        matched[qid] = (best_idx, particles[best_idx])

    # Convert to (N_FRAMES, 2) arrays — NaN where track is absent
    tracks: dict[int, np.ndarray] = {}
    track_lengths: dict[int, int] = {}
    for qid, (pidx, dets) in matched.items():
        coords = np.full((N_FRAMES, 2), np.nan)
        for t, x, y in dets:
            coords[t] = [x, y]
        tracks[qid] = coords
        track_lengths[qid] = len(dets)
        print(
            f"  Track {qid} → particle {pidx} | "
            f"{len(dets)} spots | frames {dets[0][0]}–{dets[-1][0]}"
        )

    return tracks, track_lengths


def load_sleap_training_time() -> tuple[float, int]:
    """Sum train/time + val/time from SLEAP training log CSV.

    Searches for a training log under the external SLEAP model directory
    for the pinned-down jellyfish experiment.  Also extracts the
    labeled-frame count (``n=<K>``) from the model directory name.

    Returns
    -------
    tuple[float, int]
        (total_training_time_seconds, n_labeled_frames).
    """
    sleap_model_root = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        '..', '..', 'RIPPLE_experiments', 'SLEAP',
        'pinned_down', 'SLEAP_pinned_down', 'models',
    )
    log_files = glob.glob(
        os.path.join(sleap_model_root, '*', 'training_log.csv'),
    )
    if not log_files:
        raise FileNotFoundError(
            f'No SLEAP training logs found under {sleap_model_root}'
        )

    total = 0.0
    n_labels = 0
    for path in sorted(log_files):
        model_dir = os.path.basename(os.path.dirname(path))
        m = re.search(r'n=(\d+)', model_dir)
        if m:
            n_labels += int(m.group(1))
        with open(path) as f:
            reader = csv.DictReader(f)
            for row in reader:
                t = row.get('train/time', '').strip()
                v = row.get('val/time', '').strip()
                if t:
                    total += float(t)
                if v:
                    total += float(v)

    return total, n_labels


def load_sleap() -> dict[int, np.ndarray]:
    """Load SLEAP tracks from analysis CSV.

    The CSV has columns: track, frame_idx, instance.score,
    track1.x, track1.y, track1.score, track2.x, ..., track3.x, track3.y, track3.score.
    Matches track1→(68,64), track2→(63,59), track3→(18,44).
    Missing values (empty strings) are stored as NaN.

    Returns
    -------
    dict[int, np.ndarray]
        Mapping track_id → (N_FRAMES, 2) array of (x, y).
    """
    csv_path = os.path.join(
        BASE_DIR, 'sleap_tracks',
        'labels.v001.000_pinned_down_jelly.analysis.csv',
    )
    tracks: dict[int, np.ndarray] = {
        1: np.full((N_FRAMES, 2), np.nan),
        2: np.full((N_FRAMES, 2), np.nan),
        3: np.full((N_FRAMES, 2), np.nan),
    }
    # Column mapping: track1 → GT track 1 (68,64), etc.
    col_map = {1: (3, 4), 2: (6, 7), 3: (9, 10)}

    with open(csv_path) as f:
        reader = csv.reader(f)
        next(reader)  # skip header
        for row in reader:
            frame = int(row[1])
            for tid, (cx, cy) in col_map.items():
                xs, ys = row[cx].strip(), row[cy].strip()
                if xs and ys:
                    tracks[tid][frame] = [float(xs), float(ys)]

    return tracks


# ═══════════════════════════════════════════════════════
# APP Metric
# ═══════════════════════════════════════════════════════

def compute_app(
    gt_tracks: dict[int, np.ndarray],
    pred_tracks: dict[int, np.ndarray],
) -> dict[int | str, float]:
    """Compute Average Point Precision at each threshold.

    Uses Euclidean distance in pixel space:

        d(p, p̂) = sqrt( (x − x̂)² + (y − ŷ)² )

    Points where the prediction is NaN (missing) automatically fail
    all thresholds (NaN comparisons evaluate to False).

    Returns
    -------
    dict[int | str, float]
        Mapping threshold → APP value in [0, 1], plus 'average' key.
    """
    results = {}
    for tau in THRESHOLDS:
        total_correct = 0
        total_points = 0
        for track_id in gt_tracks:
            gt = gt_tracks[track_id]      # (400, 2)
            pred = pred_tracks[track_id]  # (400, 2)

            dx = gt[:, 0] - pred[:, 0]
            dy = gt[:, 1] - pred[:, 1]
            d = np.sqrt(dx ** 2 + dy ** 2)

            # NaN distances → comparison yields False → counted as incorrect
            within = d <= tau
            total_correct += np.nansum(within)
            total_points += N_FRAMES

        results[tau] = total_correct / total_points

    results['average'] = np.mean([results[tau] for tau in THRESHOLDS])
    return results


def compute_app_per_track(
    gt_tracks: dict[int, np.ndarray],
    pred_tracks: dict[int, np.ndarray],
) -> dict[int, dict[int | str, float]]:
    """Compute APP separately for each track.

    Returns
    -------
    dict[int, dict[int | str, float]]
        Mapping track_id → {threshold → APP value}.
    """
    per_track = {}
    for track_id in gt_tracks:
        results = {}
        gt = gt_tracks[track_id]
        pred = pred_tracks[track_id]
        dx = gt[:, 0] - pred[:, 0]
        dy = gt[:, 1] - pred[:, 1]
        d = np.sqrt(dx ** 2 + dy ** 2)
        for tau in THRESHOLDS:
            within = d <= tau
            results[tau] = np.nansum(within) / N_FRAMES
        results['average'] = np.mean([results[tau] for tau in THRESHOLDS])
        per_track[track_id] = results
    return per_track


# ═══════════════════════════════════════════════════════
# Plotting
# ═══════════════════════════════════════════════════════
# Global font settings: Arial, bold
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial'],
    'font.weight': 'bold',
    'axes.labelweight': 'bold',
    'axes.titleweight': 'bold',
    'figure.titleweight': 'bold',
})
# Visual encoding
METHOD_MARKERS = {
    'LocoTrack': 'o',
    'Ripple': 's',
    'TrackMate': '^',
    'SLEAP': 'P',
    'Manual Annotation': 'D',
}


METHOD_COLORS = {
    'LocoTrack': '#377eb8',
    'Ripple': '#e41a1c',
    'TrackMate': '#4daf4a',
    'SLEAP': '#ff7f00',
    'Manual Annotation': '#984ea3',
}


def _save_with_legend(
    fig,
    ax,
    plotted_methods: set[str],
    outpath: str,
) -> None:
    """Save a copy of the plot with a legend to a parallel folder."""
    legend_path = outpath.replace('/plots/', '/plots_with_legend/')
    legend_dir = os.path.dirname(legend_path)
    os.makedirs(legend_dir, exist_ok=True)

    handles = [
        Line2D([0], [0],
               marker=METHOD_MARKERS[m], color='w',
               markerfacecolor=METHOD_COLORS[m],
               markeredgecolor='black', markeredgewidth=0.7,
               markersize=10, label=m)
        for m in plotted_methods if m in METHOD_MARKERS
    ]
    ax.legend(handles=handles, fontsize=13, loc='best',
              framealpha=0.9, edgecolor='gray')
    fig.savefig(legend_path, bbox_inches='tight')
    print(f"  Saved → {legend_path}")


def _make_plot(
    x_values: list[float],
    method_names: list[str],
    app_dicts: list[dict[int | str, float]],
    xlabel: str,
    title: str,
    tau: int | str,
    outpath: str,
) -> None:
    """Create and save a single scatter plot for one threshold (or average)."""
    fig, ax = plt.subplots(figsize=(6, 4.5))

    use_log = 'Time' in xlabel
    plotted_methods = set()

    for x, name, app_d in zip(x_values, method_names, app_dicts):
        y = app_d[tau] * 100
        if use_log and x <= 0:
            continue
        plotted_methods.add(name)
        ax.scatter(
            x, y,
            c=METHOD_COLORS[name],
            marker=METHOD_MARKERS[name],
            s=260,
            zorder=5,
            edgecolors='black',
            linewidths=0.7,
        )

    if isinstance(tau, int):
        ylabel = f'APP @ δ = {tau} px (%)'
    else:
        ylabel = 'Average APP (%)'
    ax.set_xlabel(xlabel, fontsize=22, fontweight='bold')
    ax.set_ylabel(ylabel, fontsize=22, fontweight='bold')
    ax.tick_params(axis='both', labelsize=18)
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontweight('bold')
    if use_log:
        ax.set_xscale('log')
    ax.set_ylim(-5, 108)
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.tight_layout()
    fig.savefig(outpath, bbox_inches='tight')
    print(f"  Saved → {outpath}")
    _save_with_legend(fig, ax, plotted_methods, outpath)
    plt.close(fig)


def _make_plot_per_track(
    x_values: list[float],
    method_names: list[str],
    per_track_dicts: list[dict[int, dict[int | str, float]]],
    xlabel: str,
    title: str,
    tau: int | str,
    outpath: str,
) -> None:
    """Scatter plot with one point per (method, track) pair."""
    fig, ax = plt.subplots(figsize=(6, 4.5))

    use_log = 'Time' in xlabel
    plotted_methods = set()

    for x, name, pt_d in zip(x_values, method_names, per_track_dicts):
        if use_log and x <= 0:
            continue
        plotted_methods.add(name)
        for track_id in sorted(pt_d.keys()):
            y = pt_d[track_id][tau] * 100
            ax.scatter(
                x, y,
                c=METHOD_COLORS[name],
                marker=METHOD_MARKERS[name],
                s=220,
                zorder=5,
                edgecolors='black',
                linewidths=0.7,
            )

    if isinstance(tau, int):
        ylabel = f'APP @ δ = {tau} px (%)'
    else:
        ylabel = 'APP (%)'
    ax.set_xlabel(xlabel, fontsize=22, fontweight='bold')
    ax.set_ylabel(ylabel, fontsize=22, fontweight='bold')
    ax.tick_params(axis='both', labelsize=18)
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontweight('bold')
    if use_log:
        ax.set_xscale('log')
    ax.set_ylim(-5, 108)
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.tight_layout()
    fig.savefig(outpath, bbox_inches='tight')
    print(f"  Saved → {outpath}")
    _save_with_legend(fig, ax, plotted_methods, outpath)
    plt.close(fig)


# ═══════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════

def main() -> None:
    print('=' * 64)
    print(' Tracking Method Benchmark — Average Point Precision (APP)')
    print('=' * 64)

    # ── Load data ──────────────────────────────────────
    print('\n[1/4] Loading ground truth centroids …')
    gt = load_ground_truth()
    for tid, c in gt.items():
        print(f'  Track {tid}: centroid @ frame 0 = ({c[0,0]:.2f}, {c[0,1]:.2f})')

    print('\n[2/4] Loading LocoTrack tracks …')
    loco_tracks, loco_time = load_locotrack()
    print(f'  Inference time: {loco_time:.3f} s')

    print('\n[3/4] Loading Ripple tracks …')
    ripple_tracks, ripple_anchors, ripple_meta = load_ripple()
    print(f'  Total anchors: {ripple_anchors}')
    print(f'  Track time: {ripple_meta["total_tracks_time_formatted"]}')

    print('\n[4/5] Loading TrackMate tracks …')
    tm_tracks, tm_lengths = load_trackmate()

    print('\n[5/6] Loading SLEAP tracks …')
    sleap_tracks = load_sleap()
    # Count missing frames across all tracks
    sleap_missing = sum(
        np.isnan(sleap_tracks[tid][:, 0]).sum() for tid in sleap_tracks
    )
    print(f'  Missing frames (NaN): {sleap_missing}')

    print('\n[6/6] Computing SLEAP training time …')
    sleap_training_s, sleap_n_labels = load_sleap_training_time()
    print(f'  SLEAP training time: {sleap_training_s:.1f} s ({sleap_training_s/60:.1f} min)')
    print(f'  SLEAP labeled frames: {sleap_n_labels}')

    # ── Compute APP ────────────────────────────────────
    print('\n' + '=' * 64)
    print(' Computing APP …')
    print('=' * 64)

    app_loco = compute_app(gt, loco_tracks)
    app_ripple = compute_app(gt, ripple_tracks)
    app_tm = compute_app(gt, tm_tracks)
    app_sleap = compute_app(gt, sleap_tracks)
    app_manual = {tau: 1.0 for tau in THRESHOLDS}
    app_manual['average'] = 1.0

    # Per-track APP
    pt_loco = compute_app_per_track(gt, loco_tracks)
    pt_ripple = compute_app_per_track(gt, ripple_tracks)
    pt_tm = compute_app_per_track(gt, tm_tracks)
    pt_sleap = compute_app_per_track(gt, sleap_tracks)
    pt_manual = {tid: {tau: 1.0 for tau in THRESHOLDS + ['average']} for tid in gt}

    # Print table
    hdr = f'{"δ (px)":>8} | {"LocoTrack":>10} | {"Ripple":>10} | {"TrackMate":>10} | {"SLEAP":>10} | {"Manual":>10}'
    print(f'\n{hdr}')
    print('-' * len(hdr))
    for tau in THRESHOLDS:
        print(
            f'{tau:>8d} | '
            f'{app_loco[tau]*100:>9.2f}% | '
            f'{app_ripple[tau]*100:>9.2f}% | '
            f'{app_tm[tau]*100:>9.2f}% | '
            f'{app_sleap[tau]*100:>9.2f}% | '
            f'{app_manual[tau]*100:>9.2f}%'
        )
    print('-' * len(hdr))
    print(
        f'{"avg":>8} | '
        f'{app_loco["average"]*100:>9.2f}% | '
        f'{app_ripple["average"]*100:>9.2f}% | '
        f'{app_tm["average"]*100:>9.2f}% | '
        f'{app_sleap["average"]*100:>9.2f}% | '
        f'{app_manual["average"]*100:>9.2f}%'
    )

    # ── X-axis data for each plot ──────────────────────
    methods = ['LocoTrack', 'Ripple', 'TrackMate', 'SLEAP']
    app_dicts = [app_loco, app_ripple, app_tm, app_sleap]
    pt_dicts = [pt_loco, pt_ripple, pt_tm, pt_sleap]

    # Ripple computational time
    ripple_comp_no_cache = ripple_anchors * 0.02443 + 0.76   # seconds
    ripple_comp_cached   = ripple_anchors * 0.02443          # seconds

    # TrackMate computational time: DoG 0.1 s + Adv. Kalman 45.2 s
    tm_comp_time = 0.1 + 45.2

    # SLEAP computational time = training time
    sleap_comp_time = sleap_training_s

    # Real time
    # Per-track manual annotation times (MM:SS): T1=10:30, T2=08:15, T3=07:45
    manual_real_time_s = (10 * 60 + 30) + (8 * 60 + 15) + (7 * 60 + 45)
    ripple_real_time_s = ripple_meta['total_tracks_time_ms'] / 1000.0
    tm_real_time_s = 5.1 * 60  # 5.1 minutes
    sleap_real_s = sleap_training_s + ripple_real_time_s  # training + annotation

    # Convert real times to minutes
    loco_real_min = loco_time / 60.0
    ripple_real_min = ripple_real_time_s / 60.0
    tm_real_min = tm_real_time_s / 60.0
    sleap_real_min = sleap_real_s / 60.0
    manual_real_min = manual_real_time_s / 60.0

    x_data = {
        # [LocoTrack, Ripple, TrackMate, SLEAP]
        'params':              [0, 0, 360, 0],
        'annotations':         [0, ripple_anchors, 0, ripple_anchors],
        'comp_time_no_cache':  [loco_time, ripple_comp_no_cache, tm_comp_time, sleap_comp_time],
        'comp_time_cached':    [loco_time, ripple_comp_cached, tm_comp_time, sleap_comp_time],
        'real_time':           [loco_time, ripple_real_time_s, tm_real_time_s, sleap_real_s],
    }

    # Per-track x-values: annotations are per-track, not total
    n_tracks = len(gt)
    x_data_per_track = dict(x_data)  # shallow copy — shares non-annotation lists
    x_data_per_track['annotations'] = [
        0,
        ripple_anchors / n_tracks,   # mean anchors per track
        0,
        ripple_anchors / n_tracks,   # SLEAP uses same anchor set
    ]

    print(f'\nRipple comp time (no cache) : {ripple_comp_no_cache:.3f} s')
    print(f'Ripple comp time (cached)   : {ripple_comp_cached:.3f} s')
    print(f'SLEAP comp time             : {sleap_comp_time:.1f} s ({sleap_comp_time/60:.1f} min)')
    print(f'SLEAP real time             : {sleap_real_s:.1f} s ({sleap_real_s/60:.1f} min)')
    print(f'Manual real time            : {manual_real_time_s} s  ({manual_real_time_s/60:.1f} min)')

    # ── Export machine-readable summary ────────────────
    def _app_to_json(d: dict) -> dict:
        return {str(k): round(v * 100, 2) for k, v in d.items()}

    def _pt_to_json(ptd: dict) -> dict:
        return {str(tid): _app_to_json(ptd[tid]) for tid in sorted(ptd)}

    summary = {
        'experiment': 'pinned_down_jelly',
        'n_frames': N_FRAMES,
        'resolution': [W, H],
        'thresholds': THRESHOLDS,
        'methods': [str(m) for m in methods],
        'aggregate_app': {
            name: _app_to_json(d)
            for name, d in zip(methods, app_dicts)
        },
        'per_track_app': {
            name: _pt_to_json(d)
            for name, d in zip(methods, pt_dicts)
        },
        'annotations': {
            'LocoTrack': 0,
            'Ripple': ripple_anchors,
            'TrackMate': 0,
            'SLEAP': ripple_anchors,
        },
        'ripple_total_anchors': ripple_anchors,
        'sleap_n_labels': sleap_n_labels,
        'timing': {
            'locotrack_inference_s': round(loco_time, 3),
            'ripple_comp_no_cache_s': round(ripple_comp_no_cache, 3),
            'ripple_comp_cached_s': round(ripple_comp_cached, 3),
            'trackmate_comp_s': round(tm_comp_time, 1),
            'sleap_comp_s': round(sleap_comp_time, 1),
            'sleap_real_s': round(sleap_real_s, 1),
            'ripple_real_s': round(ripple_real_time_s, 1),
            'trackmate_real_min': round(tm_real_min, 2),
            'manual_real_min': round(manual_real_min, 2),
        },
        'real_time_min': {
            'LocoTrack': round(loco_real_min, 3),
            'Ripple': round(ripple_real_min, 3),
            'TrackMate': round(tm_real_min, 2),
            'SLEAP': round(sleap_real_min, 3),
        },
    }
    json_path = os.path.join(BASE_DIR, 'benchmark_results.json')
    with open(json_path, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f'\n  Exported summary → {json_path}')

    # ── Generate plots ─────────────────────────────────
    print('\nGenerating plots …')
    out_dir = os.path.join(BASE_DIR, 'plots')
    os.makedirs(out_dir, exist_ok=True)

    plot_configs = [
        ('params',             'Hyper-Parameter Configurations',
         'APP vs. Hyper-Parameter Configurations'),
        ('annotations',        'Manual Annotations',
         'APP vs. Manual Annotations'),
        ('comp_time_no_cache', 'Computational Time (s)',
         'APP vs. Computational Time'),
        ('comp_time_cached',   'Computational Time (s)',
         'APP vs. Computational Time (Cached Optical Flow)'),
        ('real_time',          'Real Time (s)',
         'APP vs. Real Time'),
    ]

    for key, xlabel, title in plot_configs:
        sub_dir = os.path.join(out_dir, key)
        os.makedirs(sub_dir, exist_ok=True)
        for tau in THRESHOLDS + ['average']:
            tau_str = f'delta_{tau}' if isinstance(tau, int) else 'average'
            tau_label = f'{title} (δ = {tau} px)' if isinstance(tau, int) else title
            _make_plot(
                x_values=x_data[key],
                method_names=methods,
                app_dicts=app_dicts,
                xlabel=xlabel,
                title=tau_label,
                tau=tau,
                outpath=os.path.join(sub_dir, f'{tau_str}.svg'),
            )

    # ── Per-track plots ────────────────────────────────
    print('\nGenerating per-track plots …')

    for key, xlabel, title in plot_configs:
        sub_dir = os.path.join(out_dir, 'per_track', key)
        os.makedirs(sub_dir, exist_ok=True)
        for tau in THRESHOLDS + ['average']:
            tau_str = f'delta_{tau}' if isinstance(tau, int) else 'average'
            tau_label = f'{title} (δ = {tau} px)' if isinstance(tau, int) else title
            _make_plot_per_track(
                x_values=x_data_per_track[key],
                method_names=methods,
                per_track_dicts=pt_dicts,
                xlabel=xlabel,
                title=tau_label,
                tau=tau,
                outpath=os.path.join(sub_dir, f'{tau_str}.svg'),
            )

    print('\nDone ✓')


if __name__ == '__main__':
    main()
