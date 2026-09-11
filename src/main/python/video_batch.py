"""Frame-range helpers for clip-batched optical flow.

Frame indices are 0-based and inclusive. A clip [start, end] has
(end - start + 1) frames and (end - start) flow pairs.
"""

from __future__ import annotations

import os


def resolve_frame_range(request, total_frames=None):
    """Return (start, end) inclusive, or (None, None) for the full video."""
    if not request:
        return None, None
    start = request.get("frame_start", request.get("start_frame"))
    end = request.get("frame_end", request.get("end_frame"))
    if start is None and end is None:
        return None, None
    try:
        start = 0 if start is None else int(start)
        end = None if end is None else int(end)
    except (TypeError, ValueError):
        return None, None
    start = max(0, start)
    if total_frames is not None:
        last = max(0, int(total_frames) - 1)
        if end is None:
            end = last
        end = min(end, last)
        start = min(start, last)
    if end is not None and end < start:
        end = start
    return start, end


def has_frame_range(frame_start, frame_end):
    return frame_start is not None and frame_end is not None


def clip_frame_count(frame_start, frame_end, fallback=None):
    if has_frame_range(frame_start, frame_end):
        return int(frame_end) - int(frame_start) + 1
    return fallback


def flow_cache_key(video_path, frame_start=None, frame_end=None, extra=""):
    key = str(video_path)
    if extra:
        key = f"{key}_{extra}"
    if has_frame_range(frame_start, frame_end):
        key = f"{key}#f{int(frame_start)}-{int(frame_end)}"
    return key


def flow_range_token(frame_start, frame_end):
    if not has_frame_range(frame_start, frame_end):
        return ""
    return f"f{int(frame_start)}-{int(frame_end)}"


def filename_matches_range(filename, frame_start, frame_end):
    """Clip NPZs must include _f{start}-{end}_; full-video files must not."""
    if filename is None:
        return False
    token = flow_range_token(frame_start, frame_end)
    if token:
        return f"_{token}_" in filename or filename.endswith(f"_{token}_optical_flow.npz")
    import re
    return re.search(r"_f\d+-\d+_", os.path.basename(filename)) is None


def slice_flow_to_range(flows, frame_start, frame_end):
    """Slice a full-video flow array down to a clip, or return clip-sized data as-is."""
    if flows is None or not has_frame_range(frame_start, frame_end):
        return flows
    expected_pairs = int(frame_end) - int(frame_start)
    n_pairs = int(flows.shape[0])
    if n_pairs == expected_pairs:
        return flows
    start = int(frame_start)
    end = int(frame_end)
    if start >= 0 and end <= n_pairs and end > start:
        return flows[start:end]
    raise ValueError(
        f"Flow has {n_pairs} pairs but clip [{start}, {end}] needs {expected_pairs}"
    )


def to_local_frame(global_frame, frame_start):
    if frame_start is None:
        return int(global_frame)
    return int(global_frame) - int(frame_start)


def to_global_frame(local_frame, frame_start):
    if frame_start is None:
        return int(local_frame)
    return int(local_frame) + int(frame_start)


def remap_anchors(anchors, frame_start):
    """Convert global (frame, x, y) anchors to local frames for a clip-sized flow."""
    if not anchors or frame_start is None:
        return anchors
    remapped = []
    for item in anchors:
        if isinstance(item, dict):
            frame = to_local_frame(item["frame"], frame_start)
            remapped.append({**item, "frame": frame})
        else:
            frame, x, y = item[0], item[1], item[2]
            remapped.append((to_local_frame(frame, frame_start), x, y))
    return remapped


def shift_track_frames(track_obj, frame_start):
    """Rewrite exported track JSON frames from local to global indices."""
    if not track_obj or frame_start is None:
        return track_obj
    frames = track_obj.get("frames") or track_obj.get("annotations")
    if not frames:
        return track_obj
    for item in frames:
        item["frame"] = to_global_frame(item.get("frame", 0), frame_start)
    return track_obj


def memory_status_payload():
    available_bytes = None
    total_bytes = None
    try:
        import psutil
        vm = psutil.virtual_memory()
        available_bytes = int(vm.available)
        total_bytes = int(vm.total)
    except Exception:
        try:
            with open("/proc/meminfo", "r") as handle:
                info = {}
                for line in handle:
                    parts = line.split()
                    if len(parts) >= 2:
                        info[parts[0].rstrip(":")] = int(parts[1]) * 1024
                available_bytes = info.get("MemAvailable")
                total_bytes = info.get("MemTotal")
        except Exception:
            pass
    return {
        "status": "ok",
        "available_bytes": available_bytes,
        "total_bytes": total_bytes,
        "available_mb": None if available_bytes is None else available_bytes / (1024 * 1024),
        "available_gb": None if available_bytes is None else available_bytes / (1024 ** 3),
        "total_gb": None if total_bytes is None else total_bytes / (1024 ** 3),
    }


def load_avi_range(video_path, frame_start=None, frame_end=None):
    """Read an AVI slice with OpenCV. Returns (T,H,W,3) uint8 RGB and full T."""
    import cv2
    import numpy as np

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open AVI: {video_path}")
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
    start = 0 if frame_start is None else max(0, int(frame_start))
    end = (total - 1 if total > 0 else 0) if frame_end is None else int(frame_end)
    if total > 0:
        end = min(end, total - 1)
        start = min(start, end)
    if start > 0:
        cap.set(cv2.CAP_PROP_POS_FRAMES, start)
    frames = []
    for _ in range(end - start + 1):
        ok, frame = cap.read()
        if not ok:
            break
        if frame.ndim == 2:
            frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)
        else:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(frame)
    cap.release()
    if not frames:
        raise RuntimeError(f"No AVI frames read from {video_path} [{start}, {end}]")
    return np.stack(frames, axis=0), total if total > 0 else len(frames)


def load_tiff_range(video_path, frame_start=None, frame_end=None):
    """Load a TIFF slice without reading unused pages when possible.

    Returns (volume T,H,W float32, full_T, full_H, full_W).
    """
    import numpy as np
    import tifffile

    path = str(video_path)
    start = 0 if frame_start is None else max(0, int(frame_start))

    with tifffile.TiffFile(path) as tif:
        num_pages = len(tif.pages)
        series_frames = num_pages
        full_h = full_w = None
        if tif.series:
            shape = tif.series[0].shape
            if len(shape) >= 3:
                series_frames = int(shape[0])
                full_h, full_w = int(shape[1]), int(shape[2])
            elif len(shape) >= 2:
                full_h, full_w = int(shape[0]), int(shape[1])
        imagej_stack = series_frames > num_pages

        full_t = series_frames if imagej_stack else num_pages
        end = (full_t - 1) if frame_end is None else min(int(frame_end), full_t - 1)
        start = min(start, end)

        if imagej_stack:
            vol = None
            try:
                store = tifffile.imread(path, aszarr=True)
                try:
                    import zarr
                    zarr_arr = zarr.open(store, mode="r")
                    vol = np.array(zarr_arr[start:end + 1])
                finally:
                    try:
                        store.close()
                    except Exception:
                        pass
            except Exception:
                vol = None
            if vol is None:
                vol = tifffile.imread(path)
                vol = _ensure_thw(vol)
                vol = vol[start:end + 1]
            vol = _ensure_thw(vol)
        elif num_pages > 1:
            frames = []
            for index in range(start, end + 1):
                frames.append(tif.pages[index].asarray())
            vol = np.stack(frames, axis=0)
            vol = _ensure_thw(vol)
        else:
            vol = tif.asarray()
            vol = _ensure_thw(vol)
            full_t = vol.shape[0]
            end = min(end, full_t - 1)
            start = min(start, end)
            if start != 0 or end != full_t - 1:
                vol = vol[start:end + 1]
            if full_h is None:
                full_h, full_w = int(vol.shape[1]), int(vol.shape[2])

    vol = vol.astype(np.float32, copy=False)
    if full_h is None:
        full_h, full_w = int(vol.shape[1]), int(vol.shape[2])
    return vol, int(full_t), int(full_h), int(full_w)


def _ensure_thw(vol):
    import numpy as np

    if vol.ndim == 2:
        return vol[None, ...]
    if vol.ndim == 4:
        if vol.shape[-1] in (3, 4):
            vol = np.mean(vol[..., :3], axis=-1)
        elif vol.shape[1] in (3, 4):
            vol = np.mean(vol[:, :3, :, :], axis=1)
        else:
            raise ValueError(f"Unexpected 4D TIFF shape: {vol.shape}")
    if vol.ndim != 3:
        raise ValueError(f"Expected 3D volume (T,H,W), got {vol.shape}")
    if vol.shape[-1] <= 10 and vol.shape[-1] < vol.shape[0] and vol.shape[-1] < vol.shape[1]:
        vol = np.moveaxis(vol, -1, 0)
    return vol
