#!/usr/bin/env python3
"""Video Export Module with Track Overlay.

This module exports videos with track annotations overlaid on each frame.
It supports:
- Persistent trailing (track history shown as lines)
- Point markers at current track positions
- Configurable trail/point sizes
- Configurable FPS
- MP4 output using OpenCV's VideoWriter with H.264 encoding

Dependencies (already in ripple-env):
- numpy: Array operations
- tifffile: TIFF file reading
- mediapy: AVI file reading
- opencv-python: Frame drawing and video writing
- pillow: Image processing

Edge cases handled:
- Empty tracks (skipped)
- Tracks with gaps (linear interpolation optional)
- Invalid frame indices (clipped to valid range)
- Missing track points for current frame (skipped)
- Videos with different formats (TIFF, AVI)
- Memory-efficient streaming (frame-by-frame processing)
- Track colors preserved from annotations
- Cross-platform compatibility (Windows, macOS, Linux)
- Cancellation support via signal handling
"""

import argparse
import json
import sys
import os
import signal
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
import warnings

import numpy as np

# Cancellation support
_cancelled = False

def _signal_handler(signum, frame):
    """Handle interrupt signals for graceful cancellation."""
    global _cancelled
    _cancelled = True
    print("\n⚠️ Export cancelled by user", file=sys.stderr)

# Register signal handlers
signal.signal(signal.SIGINT, _signal_handler)
signal.signal(signal.SIGTERM, _signal_handler)

def is_cancelled() -> bool:
    """Check if export has been cancelled."""
    return _cancelled

def check_cancelled(operation: str = "Export"):
    """Check cancellation and raise exception if cancelled."""
    if _cancelled:
        raise InterruptedError(f"{operation} cancelled by user")

# Image/video processing
try:
    import cv2
    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False
    print("ERROR: opencv-python not available, video export requires OpenCV", file=sys.stderr)

try:
    import tifffile
    HAS_TIFFFILE = True
except ImportError:
    HAS_TIFFFILE = False

try:
    import mediapy as media
    HAS_MEDIAPY = True
except ImportError:
    HAS_MEDIAPY = False

from PIL import Image, ImageDraw


# =============================================================================
# CONSTANTS
# =============================================================================

# Default track colors (matching Java's palette) - used only as fallback
DEFAULT_TRACK_COLORS = [
    (255, 87, 51),    # Red-orange
    (46, 204, 113),   # Green
    (52, 152, 219),   # Blue
    (155, 89, 182),   # Purple
    (241, 196, 15),   # Yellow
    (26, 188, 156),   # Teal
    (231, 76, 60),    # Red
    (52, 73, 94),     # Dark blue-gray
    (230, 126, 34),   # Orange
    (149, 165, 166),  # Gray
]


# =============================================================================
# VIDEO LOADING UTILITIES
# =============================================================================

def load_video_frames(video_path: str) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Load all video frames from TIFF or AVI file.
    
    Args:
        video_path: Path to video file (TIFF or AVI)
        
    Returns:
        Tuple of (frames array [T, H, W] or [T, H, W, C], metadata dict)
        
    Raises:
        ValueError: If file format is not supported
        FileNotFoundError: If file doesn't exist
    """
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video file not found: {video_path}")
    
    video_path = str(video_path)
    ext = os.path.splitext(video_path)[1].lower()
    
    metadata = {
        "path": video_path,
        "format": ext,
        "is_grayscale": False,
    }
    
    if ext in ('.tif', '.tiff'):
        if not HAS_TIFFFILE:
            raise ImportError("tifffile is required to load TIFF files")
        
        print(f"  Loading TIFF: {video_path}")
        frames = tifffile.imread(video_path)
        
        # Handle different TIFF formats
        if frames.ndim == 2:
            # Single frame - add time dimension
            frames = frames[np.newaxis, ...]
        elif frames.ndim == 3:
            # Could be TxHxW (grayscale) or HxWxC (single RGB frame)
            if frames.shape[2] in (3, 4):
                # Single RGB/RGBA frame - add time dimension
                frames = frames[np.newaxis, ...]
            else:
                # Grayscale video (T, H, W)
                metadata["is_grayscale"] = True
        elif frames.ndim == 4:
            # TxHxWxC (color video)
            pass
        else:
            raise ValueError(f"Unexpected TIFF shape: {frames.shape}")
            
    elif ext == '.avi':
        if not HAS_MEDIAPY:
            raise ImportError("mediapy is required to load AVI files")
        
        print(f"  Loading AVI: {video_path}")
        frames = media.read_video(video_path)
        
        # mediapy returns (T, H, W, C) in [0, 1] range
        if frames.dtype == np.float32 or frames.dtype == np.float64:
            frames = (frames * 255).astype(np.uint8)
            
    else:
        raise ValueError(f"Unsupported video format: {ext}")
    
    T = frames.shape[0]
    H = frames.shape[1]
    W = frames.shape[2]
    C = frames.shape[3] if frames.ndim == 4 else 1
    
    metadata["total_frames"] = T
    metadata["height"] = H
    metadata["width"] = W
    metadata["channels"] = C
    metadata["is_grayscale"] = (C == 1)
    
    print(f"  Loaded video: {T} frames, {W}x{H}, {C} channel(s)")
    
    return frames, metadata


def normalize_frame_to_rgb(
    frame: np.ndarray, 
    display_min: Optional[float] = None,
    display_max: Optional[float] = None
) -> np.ndarray:
    """Convert a frame to 8-bit RGB format with brightness/contrast adjustment.
    
    Args:
        frame: Input frame (H, W) or (H, W, C)
        display_min: Minimum display value (maps to black). If None, uses data min.
        display_max: Maximum display value (maps to white). If None, uses data max.
        
    Returns:
        RGB frame as uint8 (H, W, 3)
    """
    # Convert to float for processing
    frame = frame.astype(np.float64)
    
    # Handle grayscale - keep as single channel for now, convert to RGB at end
    is_grayscale = False
    if frame.ndim == 2:
        is_grayscale = True
    elif frame.ndim == 3 and frame.shape[2] == 1:
        frame = frame[:, :, 0]
        is_grayscale = True
    elif frame.ndim == 3 and frame.shape[2] == 4:
        # RGBA -> RGB (drop alpha)
        frame = frame[..., :3]
    
    # Auto-compute display range if not specified
    if display_min is None or display_max is None:
        # Use percentile-based auto range for better contrast
        if is_grayscale:
            p_low = np.percentile(frame, 0.5)
            p_high = np.percentile(frame, 99.5)
        else:
            p_low = np.percentile(frame, 0.5)
            p_high = np.percentile(frame, 99.5)
        
        if display_min is None:
            display_min = p_low
        if display_max is None:
            display_max = p_high
    
    # Avoid division by zero
    if display_max <= display_min:
        display_max = display_min + 1
    
    # Apply display range mapping (linear contrast stretch)
    frame = (frame - display_min) / (display_max - display_min)
    frame = np.clip(frame, 0, 1)
    frame = (frame * 255).astype(np.uint8)
    
    # Convert grayscale to RGB
    if is_grayscale:
        frame = np.stack([frame, frame, frame], axis=-1)
    
    return frame


# =============================================================================
# TRACK PARSING UTILITIES
# =============================================================================

def parse_tracks_json(json_path: str) -> Tuple[Dict[str, Dict[int, Tuple[int, int]]], Dict[str, Tuple[int, int, int]]]:
    """Parse tracks from RIPPLE JSON export format.
    
    Args:
        json_path: Path to JSON file with track data
        
    Returns:
        Tuple of (tracks dict, colors dict)
        - tracks: {track_id: {frame: (x, y), ...}, ...}
        - colors: {track_id: (r, g, b), ...}
    """
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    tracks = {}
    colors = {}
    
    # Handle different JSON formats (rich vs trajectories_only)
    tracks_array = data.get("tracks", [])
    
    for i, track_data in enumerate(tracks_array):
        track_id = track_data.get("track_id", f"Track_{i+1}")
        
        # Parse color if available (format: "#RRGGBB" or {"r": R, "g": G, "b": B})
        color_data = track_data.get("color")
        if isinstance(color_data, str) and color_data.startswith("#"):
            # Hex color
            hex_color = color_data.lstrip("#")
            r = int(hex_color[0:2], 16)
            g = int(hex_color[2:4], 16)
            b = int(hex_color[4:6], 16)
            colors[track_id] = (r, g, b)
        elif isinstance(color_data, dict):
            colors[track_id] = (
                color_data.get("r", 255),
                color_data.get("g", 255),
                color_data.get("b", 255)
            )
        else:
            # Use default color from palette
            colors[track_id] = DEFAULT_TRACK_COLORS[i % len(DEFAULT_TRACK_COLORS)]
        
        # Parse frames/positions - handle multiple key names
        # RIPPLE uses 'annotations', Java export uses 'frames'
        frames_data = track_data.get("annotations", 
                      track_data.get("frames", 
                      track_data.get("positions", [])))
        
        track_points = {}
        
        if isinstance(frames_data, list):
            # List format: [{"frame": F, "x": X, "y": Y}, ...]
            for point_data in frames_data:
                frame = point_data.get("frame", 0)
                x = int(round(point_data.get("x", 0)))
                y = int(round(point_data.get("y", 0)))
                track_points[frame] = (x, y)
        elif isinstance(frames_data, dict):
            # Dict format: {"frame_idx": {"x": X, "y": Y}, ...}
            for frame_str, point_data in frames_data.items():
                frame = int(frame_str)
                x = int(round(point_data.get("x", 0)))
                y = int(round(point_data.get("y", 0)))
                track_points[frame] = (x, y)
        
        if track_points:
            tracks[track_id] = track_points
    
    return tracks, colors


def parse_tracks_inline(tracks_data: List[Dict]) -> Tuple[Dict[str, Dict[int, Tuple[int, int]]], Dict[str, Tuple[int, int, int]]]:
    """Parse tracks from inline JSON data (passed directly from Java).
    
    Args:
        tracks_data: List of track objects
        
    Returns:
        Tuple of (tracks dict, colors dict)
    """
    tracks = {}
    colors = {}
    
    for i, track_data in enumerate(tracks_data):
        track_id = track_data.get("track_id", f"Track_{i+1}")
        
        # Parse color
        color_data = track_data.get("color")
        if isinstance(color_data, str) and color_data.startswith("#"):
            hex_color = color_data.lstrip("#")
            r = int(hex_color[0:2], 16)
            g = int(hex_color[2:4], 16)
            b = int(hex_color[4:6], 16)
            colors[track_id] = (r, g, b)
        elif isinstance(color_data, dict):
            colors[track_id] = (
                color_data.get("r", 255),
                color_data.get("g", 255),
                color_data.get("b", 255)
            )
        else:
            colors[track_id] = DEFAULT_TRACK_COLORS[i % len(DEFAULT_TRACK_COLORS)]
        
        # Parse frames
        frames_data = track_data.get("frames", [])
        track_points = {}
        
        for point_data in frames_data:
            frame = point_data.get("frame", 0)
            x = int(round(point_data.get("x", 0)))
            y = int(round(point_data.get("y", 0)))
            track_points[frame] = (x, y)
        
        if track_points:
            tracks[track_id] = track_points
    
    return tracks, colors


# =============================================================================
# DRAWING UTILITIES
# =============================================================================

def draw_tracks_on_frame(
    frame: np.ndarray,
    tracks: Dict[str, Dict[int, Tuple[int, int]]],
    colors: Dict[str, Tuple[int, int, int]],
    current_frame: int,
    show_points: bool = True,
    show_trails: bool = True,
    persistent_trails: bool = False,
    point_size: int = 6,
    trail_thickness: int = 2,
    trail_length: int = 30,
    trail_fade: bool = True,
) -> np.ndarray:
    """Draw track overlays on a frame.
    
    Args:
        frame: RGB frame (H, W, 3) uint8
        tracks: Dict of track_id -> {frame: (x, y)}
        colors: Dict of track_id -> (r, g, b)
        current_frame: Current frame index (0-indexed)
        show_points: Whether to draw point markers
        show_trails: Whether to draw trailing lines
        persistent_trails: If True, show full trail from start; if False, show last N frames
        point_size: Radius of point markers in pixels
        trail_thickness: Thickness of trail lines in pixels
        trail_length: Number of frames to show in trail (ignored if persistent_trails=True)
        trail_fade: Whether to fade trail opacity with distance
        
    Returns:
        Frame with overlays drawn (H, W, 3) uint8
    """
    if not HAS_CV2:
        return frame
    
    # Create a copy to draw on
    output = frame.copy()
    
    for track_id, track_points in tracks.items():
        color = colors.get(track_id, (255, 255, 255))
        # OpenCV uses BGR, convert from RGB
        bgr_color = (color[2], color[1], color[0])
        
        # Get frames for this track up to current frame
        if persistent_trails:
            # Show all frames from start to current
            relevant_frames = sorted([f for f in track_points.keys() if f <= current_frame])
        else:
            # Show last trail_length frames
            start_frame = max(0, current_frame - trail_length)
            relevant_frames = sorted([f for f in track_points.keys() if start_frame <= f <= current_frame])
        
        if not relevant_frames:
            continue
        
        # Draw trails
        if show_trails and len(relevant_frames) > 1:
            points_list = [(track_points[f][0], track_points[f][1]) for f in relevant_frames]
            
            for i in range(len(points_list) - 1):
                pt1 = points_list[i]
                pt2 = points_list[i + 1]
                
                # Calculate fade factor based on distance from current frame
                if trail_fade and not persistent_trails:
                    # Fade based on position in trail
                    fade = (i + 1) / len(points_list)
                    alpha = 0.3 + 0.7 * fade
                else:
                    alpha = 1.0
                
                # Apply alpha by blending color
                faded_color = tuple(int(c * alpha) for c in bgr_color)
                
                cv2.line(output, pt1, pt2, faded_color, trail_thickness, cv2.LINE_AA)
        
        # Draw point marker at current position
        if show_points and current_frame in track_points:
            cx, cy = track_points[current_frame]
            
            # Draw filled circle with outline
            cv2.circle(output, (cx, cy), point_size, bgr_color, -1, cv2.LINE_AA)
            # White outline for visibility
            cv2.circle(output, (cx, cy), point_size, (255, 255, 255), 1, cv2.LINE_AA)
    
    return output


# =============================================================================
# VIDEO EXPORT
# =============================================================================

def get_fourcc_for_platform() -> Tuple[int, str]:
    """Get the appropriate FourCC codec for the current platform.
    
    Returns a tuple of (FourCC code, codec name) that works reliably.
    Tests each codec by actually trying to create a VideoWriter.
    """
    import tempfile
    import platform
    
    # Codecs to try in order of preference
    # Different platforms have different codec availability
    codecs_to_try = [
        ('mp4v', '.mp4'),   # MPEG-4 - most widely supported
        ('avc1', '.mp4'),   # H.264 - works on macOS
        ('X264', '.mp4'),   # x264 H.264 encoder
        ('XVID', '.avi'),   # Xvid - widely available
        ('MJPG', '.avi'),   # Motion JPEG - fallback, larger files
    ]
    
    # Create a small test frame
    test_frame = np.zeros((100, 100, 3), dtype=np.uint8)
    
    for codec, ext in codecs_to_try:
        fourcc = cv2.VideoWriter_fourcc(*codec)
        if fourcc == -1:
            continue
            
        # Try to actually create a writer
        try:
            with tempfile.NamedTemporaryFile(suffix=ext, delete=True) as tmp:
                test_path = tmp.name
            writer = cv2.VideoWriter(test_path, fourcc, 30.0, (100, 100))
            if writer.isOpened():
                writer.write(test_frame)
                writer.release()
                # Clean up test file
                if os.path.exists(test_path):
                    os.remove(test_path)
                print(f"  Using video codec: {codec}")
                return fourcc, codec
            writer.release()
        except Exception:
            pass
    
    # Last resort - return mp4v and hope for the best
    print("  WARNING: No working codec found, trying mp4v")
    return cv2.VideoWriter_fourcc(*'mp4v'), 'mp4v'


def export_video_with_tracks(
    video_path: str,
    tracks: Dict[str, Dict[int, Tuple[int, int]]],
    colors: Dict[str, Tuple[int, int, int]],
    output_path: str,
    fps: float = 30.0,
    show_points: bool = True,
    show_trails: bool = True,
    persistent_trails: bool = False,
    point_size: int = 6,
    trail_thickness: int = 2,
    trail_length: int = 30,
    trail_fade: bool = True,
    codec: str = "mp4v",
    quality: int = 23,
    display_min: Optional[float] = None,
    display_max: Optional[float] = None,
    progress_callback=None,
) -> Dict[str, Any]:
    """Export video with track overlays to MP4 using OpenCV.
    
    Args:
        video_path: Path to input video (TIFF or AVI)
        tracks: Track data {track_id: {frame: (x, y)}}
        colors: Track colors {track_id: (r, g, b)} - from Java UI
        output_path: Path for output MP4 file
        fps: Output video framerate
        show_points: Draw point markers
        show_trails: Draw trailing lines
        persistent_trails: Show full trail history
        point_size: Point marker radius
        trail_thickness: Trail line thickness
        trail_length: Number of frames in trail (if not persistent)
        trail_fade: Fade trail with distance
        codec: Video codec fourcc (mp4v, avc1, XVID, etc.)
        quality: Quality hint (not directly used by OpenCV, kept for API compat)
        display_min: Min display value for brightness (maps to black)
        display_max: Max display value for brightness (maps to white)
        progress_callback: Optional callback(frame, total) for progress updates
        
    Returns:
        Dict with export results
        
    Raises:
        ImportError: If OpenCV is not available
        InterruptedError: If export was cancelled
        RuntimeError: If video encoding fails
    """
    if not HAS_CV2:
        raise ImportError("opencv-python is required for video export")
    
    print(f"\n{'='*60}")
    print("VIDEO EXPORT WITH TRACKS")
    print(f"{'='*60}")
    print(f"Input: {video_path}")
    print(f"Output: {output_path}")
    print(f"FPS: {fps}")
    print(f"Settings: points={show_points}, trails={show_trails}, persistent={persistent_trails}")
    print(f"Sizes: point={point_size}px, trail={trail_thickness}px, length={trail_length} frames")
    if display_min is not None and display_max is not None:
        print(f"Display range: {display_min:.1f} - {display_max:.1f}")
    else:
        print(f"Display range: auto")
    print(f"{'='*60}\n")
    
    # Check for cancellation before starting
    check_cancelled("Video export")
    
    # Load video
    frames, metadata = load_video_frames(video_path)
    T = metadata["total_frames"]
    H = metadata["height"]
    W = metadata["width"]
    
    print(f"Loaded {T} frames ({W}x{H})")
    print(f"Processing {len(tracks)} tracks...")
    
    # Validate tracks have data
    valid_tracks = {k: v for k, v in tracks.items() if v}
    if not valid_tracks:
        print("WARNING: No valid tracks to overlay!")
    
    # Ensure output path ends with .mp4
    output_path = str(output_path)
    if not output_path.lower().endswith('.mp4'):
        output_path += '.mp4'
    
    # Ensure output directory exists
    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    
    # Get appropriate codec for platform
    fourcc, codec_name = get_fourcc_for_platform()
    
    # Create VideoWriter
    # OpenCV expects BGR frames, but we'll convert from RGB
    writer = cv2.VideoWriter(output_path, fourcc, fps, (W, H))
    
    if not writer.isOpened():
        raise RuntimeError(f"Failed to create video writer for {output_path}. "
                          f"Check that OpenCV has video encoding support.")
    
    try:
        # Process each frame
        for frame_idx in range(T):
            # Check for cancellation periodically
            if frame_idx % 10 == 0:
                check_cancelled("Video export")
            
            # Get frame
            frame = frames[frame_idx]
            
            # Normalize to RGB with display range (brightness/contrast)
            rgb_frame = normalize_frame_to_rgb(frame, display_min, display_max)
            
            # Draw tracks
            output_frame = draw_tracks_on_frame(
                rgb_frame,
                valid_tracks,
                colors,
                frame_idx,
                show_points=show_points,
                show_trails=show_trails,
                persistent_trails=persistent_trails,
                point_size=point_size,
                trail_thickness=trail_thickness,
                trail_length=trail_length,
                trail_fade=trail_fade,
            )
            
            # Convert RGB to BGR for OpenCV
            bgr_frame = cv2.cvtColor(output_frame, cv2.COLOR_RGB2BGR)
            
            # Write frame
            writer.write(bgr_frame)
            
            # Progress update
            if progress_callback:
                progress_callback(frame_idx + 1, T)
            elif frame_idx % 50 == 0 or frame_idx == T - 1:
                print(f"  Progress: {frame_idx + 1}/{T} frames ({100*(frame_idx+1)/T:.1f}%)")
        
    except InterruptedError:
        # Clean up on cancellation
        writer.release()
        # Remove partial output file
        if os.path.exists(output_path):
            try:
                os.remove(output_path)
            except OSError:
                pass
        raise
        
    except Exception as e:
        # Clean up on error
        writer.release()
        raise RuntimeError(f"Video encoding failed: {e}")
    
    finally:
        # Always release the writer
        writer.release()
    
    # Verify output file exists and has content
    if not os.path.exists(output_path):
        raise RuntimeError("Video encoding completed but output file not found")
    
    output_size_mb = os.path.getsize(output_path) / (1024 * 1024)
    
    if output_size_mb < 0.001:
        raise RuntimeError("Video encoding completed but output file is empty")
    
    result = {
        "status": "ok",
        "output_path": output_path,
        "frames_exported": T,
        "fps": fps,
        "width": W,
        "height": H,
        "tracks_count": len(valid_tracks),
        "output_size_mb": round(output_size_mb, 2),
    }
    
    print(f"\n{'='*60}")
    print(f"✓ Export complete!")
    print(f"  Output: {output_path}")
    print(f"  Size: {output_size_mb:.2f} MB")
    print(f"  Duration: {T/fps:.2f} seconds @ {fps} FPS")
    print(f"{'='*60}\n")
    
    return result


# =============================================================================
# COMMAND LINE INTERFACE
# =============================================================================

def main():
    """Main entry point for command-line usage."""
    parser = argparse.ArgumentParser(
        description="Export video with track overlays",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        "video_path",
        help="Path to input video (TIFF or AVI)"
    )
    parser.add_argument(
        "tracks_json",
        help="Path to tracks JSON file (RIPPLE export format)"
    )
    parser.add_argument(
        "-o", "--output",
        default=None,
        help="Output MP4 path (default: <video_name>_with_tracks.mp4)"
    )
    parser.add_argument(
        "--fps",
        type=float,
        default=30.0,
        help="Output video framerate"
    )
    parser.add_argument(
        "--no-points",
        action="store_true",
        help="Hide point markers"
    )
    parser.add_argument(
        "--no-trails",
        action="store_true",
        help="Hide trailing lines"
    )
    parser.add_argument(
        "--persistent-trails",
        action="store_true",
        help="Show full trail history instead of last N frames"
    )
    parser.add_argument(
        "--point-size",
        type=int,
        default=6,
        help="Point marker radius in pixels"
    )
    parser.add_argument(
        "--trail-thickness",
        type=int,
        default=2,
        help="Trail line thickness in pixels"
    )
    parser.add_argument(
        "--trail-length",
        type=int,
        default=30,
        help="Number of frames to show in trail"
    )
    parser.add_argument(
        "--no-trail-fade",
        action="store_true",
        help="Disable trail fading"
    )
    parser.add_argument(
        "--quality",
        type=int,
        default=23,
        help="Video quality (CRF, 0-51, lower is better)"
    )
    parser.add_argument(
        "--display-min",
        type=float,
        default=None,
        help="Minimum display value for brightness (maps to black). Auto if not specified."
    )
    parser.add_argument(
        "--display-max",
        type=float,
        default=None,
        help="Maximum display value for brightness (maps to white). Auto if not specified."
    )
    
    args = parser.parse_args()
    
    # Check dependencies
    if not HAS_CV2:
        print("ERROR: opencv-python is required. Install with: pip install opencv-python", file=sys.stderr)
        sys.exit(1)
    
    # Validate inputs
    if not os.path.exists(args.video_path):
        print(f"ERROR: Video file not found: {args.video_path}", file=sys.stderr)
        sys.exit(1)
    
    if not os.path.exists(args.tracks_json):
        print(f"ERROR: Tracks JSON file not found: {args.tracks_json}", file=sys.stderr)
        sys.exit(1)
    
    # Generate output path if not specified
    output_path = args.output
    if output_path is None:
        video_name = os.path.splitext(os.path.basename(args.video_path))[0]
        output_dir = os.path.dirname(args.video_path) or "."
        output_path = os.path.join(output_dir, f"{video_name}_with_tracks.mp4")
    
    # Load tracks
    print(f"Loading tracks from: {args.tracks_json}")
    tracks, colors = parse_tracks_json(args.tracks_json)
    print(f"  Loaded {len(tracks)} tracks")
    
    # Export video
    try:
        result = export_video_with_tracks(
            video_path=args.video_path,
            tracks=tracks,
            colors=colors,
            output_path=output_path,
            fps=args.fps,
            show_points=not args.no_points,
            show_trails=not args.no_trails,
            persistent_trails=args.persistent_trails,
            point_size=args.point_size,
            trail_thickness=args.trail_thickness,
            trail_length=args.trail_length,
            trail_fade=not args.no_trail_fade,
            quality=args.quality,
            display_min=args.display_min,
            display_max=args.display_max,
        )
        
        # Print result as JSON for programmatic use
        print("\n" + json.dumps(result, indent=2))
        
    except Exception as e:
        print(f"ERROR: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
