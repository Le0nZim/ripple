
#!/usr/bin/env python3
"""Persistent RAFT-Based Tracking Server with In-Memory Model Caching.

This server implements a long-running process architecture that maintains
RAFT model weights and conda environment state in memory, reducing
the initialization overhead associated with repeated Python
interpreter instantiation and model loading.

Key architectural features:
1. Single-load model initialization with persistent GPU memory allocation
2. Unix domain socket inter-process communication interface
3. Request-response protocol for tracking operations
4. Cross-request optical flow caching with LRU eviction
5. Graceful shutdown signaling and resource cleanup

Performance characteristics:
- Reduces overhead by eliminating repeated model loading
- Maintains low response latency for cached optical flow queries
- Supports concurrent request handling via socket multiplexing

Optimization heritage from RAFT_v4.py:
- Cached offset grid structures
- LRU-based tensor shape memoization
- Optimized tensor conversion pathways
- Segment interpolation caching with memory bounds
- Vectorized array operations
- Pre-allocated memory buffers

Format support:
- TIFF/TIF: Direct pixel intensity input
- AVI: Automatic resolution normalization with mediapy integration

Execution modes:
- Full-frame analysis (corridor_width=0)
- Adaptive corridor search with predicted centerline
- Backward temporal propagation for arbitrary anchor placement

Dependencies:
- tifffile: TIFF format I/O
- mediapy: AVI format I/O and frame manipulation
- PyTorch + torchvision: RAFT model inference
- NumPy: Numerical operations
"""

import argparse
import json
import sys
import socket
import os
import sys
import signal
import time
import atexit
import gc
from collections import OrderedDict
from pathlib import Path
from functools import lru_cache
from time import perf_counter as _pc

# ============================================================================
# CRITICAL: Parse --gpu argument BEFORE importing torch/CUDA libraries
# CUDA_VISIBLE_DEVICES must be set before any CUDA initialization occurs
# ============================================================================
def _early_gpu_selection():
    """Parse --gpu argument early and set CUDA_VISIBLE_DEVICES before torch import."""
    for i, arg in enumerate(sys.argv):
        if arg == "--gpu" and i + 1 < len(sys.argv):
            gpu_id = sys.argv[i + 1]
            os.environ["CUDA_VISIBLE_DEVICES"] = gpu_id
            print(f"🎯 GPU Selection: Using GPU {gpu_id} (CUDA_VISIBLE_DEVICES={gpu_id})")
            return gpu_id
        elif arg.startswith("--gpu="):
            gpu_id = arg.split("=", 1)[1]
            os.environ["CUDA_VISIBLE_DEVICES"] = gpu_id
            print(f"🎯 GPU Selection: Using GPU {gpu_id} (CUDA_VISIBLE_DEVICES={gpu_id})")
            return gpu_id
    return None

_SELECTED_GPU = _early_gpu_selection()
# ============================================================================

# Ensure locotrack_pytorch is importable for model imports
# (contains the canonical LocoTrack model code and weights).
_SERVER_DIR = Path(__file__).resolve().parent
_RIPPLE_ROOT = _SERVER_DIR.parents[2]  # <repo>/src/main/python -> <repo>
_LOCOTRACK_DIR = _RIPPLE_ROOT / "locotrack_pytorch"
# Add locotrack_pytorch first so its models package takes precedence
if str(_LOCOTRACK_DIR) not in sys.path:
    sys.path.insert(0, str(_LOCOTRACK_DIR))

# Fix Windows console encoding to handle Unicode characters
# This prevents UnicodeEncodeError on Windows when printing special characters
if sys.platform == 'win32':
    import io
    # Set stdout/stderr to UTF-8 with error replacement for Windows console
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    if hasattr(sys.stderr, 'reconfigure'):
        sys.stderr.reconfigure(encoding='utf-8', errors='replace')

import numpy as np
import torch
import torch.nn.functional as F
import tifffile
import warnings
import logging

# Suppress verbose torch.compile/dynamo warnings that can appear during compilation
# These are internal PyTorch messages about FakeTensor, dynamo, etc.
warnings.filterwarnings("ignore", message=".*FakeTensor.*")
warnings.filterwarnings("ignore", message=".*torch._dynamo.*")
warnings.filterwarnings("ignore", category=UserWarning, module="torch._dynamo.*")
warnings.filterwarnings("ignore", category=UserWarning, module="torch._inductor.*")
# Suppress specific logging from torch internals
logging.getLogger("torch._dynamo").setLevel(logging.ERROR)
logging.getLogger("torch._inductor").setLevel(logging.ERROR)
logging.getLogger("torch.fx").setLevel(logging.ERROR)

try:
    import zarr  # For memory-mapped TIFF access (optional)
    HAS_ZARR = True
except ImportError:
    HAS_ZARR = False
try:
    import psutil  # For memory checking (optional)
    HAS_PSUTIL = True
except ImportError:
    HAS_PSUTIL = False
import mediapy as media
import cv2  # For DIS optical flow

# For blob detection in auto-track propagation (similar to jellyfish_tracker)
try:
    from skimage.feature import blob_log
    from skimage import measure
    HAS_SKIMAGE = True
except ImportError:
    HAS_SKIMAGE = False
    print("WARNING: skimage not available, blob detection will be disabled")

# Memory optimization constants
# If available RAM drops below this, use memory-efficient mode
LOW_MEMORY_THRESHOLD_MB = 4000  # 4 GB
# If flow array exceeds this size, store as float16
FLOAT16_THRESHOLD_MB = 500  # 500 MB
# Systems with less than this total RAM will use incremental allocation by default
INCREMENTAL_ALLOC_THRESHOLD_GB = 32  # 32 GB

def get_total_system_memory_gb():
    """Get total system memory in GB.
    
    Cross-platform implementation:
    - Uses psutil if available (works on Windows, macOS, Linux)
    - Falls back to /proc/meminfo on Linux
    - Falls back to sysctl on macOS
    - Falls back to wmic on Windows
    - Returns 32GB as conservative default if all methods fail
    """
    # Method 1: psutil (cross-platform, preferred)
    if HAS_PSUTIL:
        try:
            return psutil.virtual_memory().total / (1024 ** 3)
        except Exception:
            pass
    
    # Method 2: /proc/meminfo (Linux)
    try:
        with open('/proc/meminfo', 'r') as f:
            for line in f:
                if line.startswith('MemTotal:'):
                    return int(line.split()[1]) / (1024 * 1024)  # KB to GB
    except Exception:
        pass
    
    # Method 3: sysctl (macOS)
    try:
        import subprocess
        result = subprocess.run(['sysctl', '-n', 'hw.memsize'], 
                                capture_output=True, text=True, timeout=5)
        if result.returncode == 0:
            return int(result.stdout.strip()) / (1024 ** 3)
    except Exception:
        pass
    
    # Method 4: wmic (Windows)
    try:
        import subprocess
        result = subprocess.run(['wmic', 'ComputerSystem', 'get', 'TotalPhysicalMemory'],
                                capture_output=True, text=True, timeout=5, shell=True)
        if result.returncode == 0:
            lines = result.stdout.strip().split('\n')
            for line in lines:
                line = line.strip()
                if line.isdigit():
                    return int(line) / (1024 ** 3)
    except Exception:
        pass
    
    # Default: assume 32GB (conservative - won't trigger incremental mode)
    print("  WARNING: Could not detect system memory, assuming 32GB")
    return 32

def parse_incremental_allocation_param(value):
    """Parse the incremental_allocation parameter from various input formats.
    
    Handles:
    - None -> None (auto-detect)
    - "auto" -> None (auto-detect)
    - True/False -> bool
    - "true"/"false" (case-insensitive) -> bool
    - "1"/"0" -> bool
    - 1/0 -> bool
    
    Returns:
        None for auto-detect, or bool for explicit setting
    """
    if value is None:
        return None
    
    # Handle string values
    if isinstance(value, str):
        value_lower = value.lower().strip()
        if value_lower in ('auto', 'none', ''):
            return None
        if value_lower in ('true', 'yes', '1', 'on', 'enabled'):
            return True
        if value_lower in ('false', 'no', '0', 'off', 'disabled'):
            return False
        # Unknown string - treat as auto
        print(f"  WARNING: Unknown incremental_allocation value '{value}', using auto-detect")
        return None
    
    # Handle boolean
    if isinstance(value, bool):
        return value
    
    # Handle numeric
    if isinstance(value, (int, float)):
        return bool(value)
    
    # Unknown type - auto-detect
    return None

def should_use_incremental_allocation(force_incremental=None):
    """Determine if incremental allocation should be used.
    
    Args:
        force_incremental: If not None, use this value. Otherwise auto-detect.
                          Can be bool, string ("true"/"false"/"auto"), or None.
        
    Returns:
        bool: True if incremental allocation should be used
    """
    # Parse the parameter to handle various input formats
    parsed = parse_incremental_allocation_param(force_incremental)
    
    if parsed is not None:
        return parsed
    
    # Auto-detect: use incremental for systems with < 32GB RAM
    total_ram = get_total_system_memory_gb()
    return total_ram < INCREMENTAL_ALLOC_THRESHOLD_GB

# RAFT model minimum resolution requirement
# RAFT downsamples input by 8x, and the correlation pyramid requires feature maps of at least 16x16
# Therefore minimum input size is 8 * 16 = 128 pixels in each dimension
RAFT_MINIMUM_RESOLUTION = 128

def get_available_memory_mb():
    """Get available system memory in MB."""
    if HAS_PSUTIL:
        return psutil.virtual_memory().available / (1024 * 1024)
    # Fallback: read from /proc/meminfo on Linux
    try:
        with open('/proc/meminfo', 'r') as f:
            for line in f:
                if line.startswith('MemAvailable:'):
                    return int(line.split()[1]) / 1024  # KB to MB
    except:
        pass
    return 8000  # Assume 8GB available if we can't check


class MemoryEfficientFlowArray:
    """Wrapper for optical flow arrays that uses float16 storage with float32 access.
    
    This class provides 50% memory reduction by storing flows as float16
    while providing transparent float32 access for computations.
    
    The precision loss is negligible for optical flow (typically < 0.001 pixels).
    """
    
    def __init__(self, flows, use_float16=True):
        """
        Args:
            flows: numpy array of optical flow (T, H, W, 2)
            use_float16: Whether to convert to float16 for storage
        """
        if use_float16:
            # Store as float16 for 50% memory reduction
            self._data = np.array(flows, dtype=np.float16, copy=True)
            self._is_float16 = True
        else:
            self._data = np.array(flows, dtype=np.float32, copy=True)
            self._is_float16 = False
        
        self._shape = self._data.shape
        self._dtype = np.float32  # Report as float32 for compatibility
    
    @property
    def shape(self):
        return self._shape
    
    @property
    def dtype(self):
        return self._dtype
    
    @property
    def nbytes(self):
        return self._data.nbytes
    
    @property
    def ndim(self):
        return self._data.ndim
    
    def __getitem__(self, key):
        """Access flow data, converting to float32 if stored as float16."""
        result = self._data[key]
        if self._is_float16:
            # Convert to float32 for computation
            if isinstance(result, np.ndarray):
                return result.astype(np.float32)
            else:
                # Single value - numpy will handle conversion
                return float(result)
        return result
    
    def get_raw(self):
        """Get the raw underlying array (float16 or float32)."""
        return self._data
    
    def as_float32(self):
        """Get a full float32 copy (for operations that require it)."""
        if self._is_float16:
            return self._data.astype(np.float32)
        return self._data
    
    def memory_savings_mb(self):
        """Return memory saved compared to float32 storage."""
        if self._is_float16:
            return self._data.nbytes / (1024 * 1024)  # Saved = same as current (50%)
        return 0


def load_flows_memory_efficient(flows_array):
    """Load flow array with automatic float16 compression if needed.
    
    This should be called when loading flows from disk (.npz files) to ensure
    memory-efficient storage when RAM is limited or array is large.
    
    Args:
        flows_array: numpy array from np.load (may be float32 or float16)
    
    Returns:
        MemoryEfficientFlowArray or np.ndarray depending on memory conditions
    """
    import gc
    
    # Calculate memory that float32 would need
    flow_size_mb = np.prod(flows_array.shape) * 4 / (1024 * 1024)
    available_mb = get_available_memory_mb()
    
    # Use float16 wrapper if array is large or RAM is low
    use_float16 = (flow_size_mb > FLOAT16_THRESHOLD_MB or 
                   available_mb < LOW_MEMORY_THRESHOLD_MB)
    
    if use_float16:
        # Wrap in memory-efficient container (converts to float16 if not already)
        wrapper = MemoryEfficientFlowArray(flows_array, use_float16=True)
        saved_mb = flow_size_mb - (wrapper.nbytes / (1024 * 1024))
        print(f"  💾 Memory-efficient loading: {wrapper.nbytes/(1024*1024):.0f} MB "
              f"(saved {saved_mb:.0f} MB, avail RAM: {available_mb:.0f} MB)")
        # Force garbage collection to free the memory-mapped array
        gc.collect()
        return wrapper
    else:
        # Standard float32 storage - copy to avoid memory-mapped access issues
        result = np.array(flows_array, dtype=np.float32, copy=True)
        gc.collect()
        return result


def wrap_flows_memory_efficient(flows_array, label="computed"):
    """Wrap computed flow array with float16 compression if beneficial.
    
    Call this after computing flows to convert to memory-efficient storage.
    This also triggers garbage collection to free the original float32 array.
    
    Args:
        flows_array: numpy array of computed optical flow
        label: Label for logging (e.g., "DIS", "LocoTrack")
    
    Returns:
        MemoryEfficientFlowArray or np.ndarray depending on memory conditions
    """
    import gc
    
    flow_size_mb = np.prod(flows_array.shape) * 4 / (1024 * 1024)
    available_mb = get_available_memory_mb()
    
    use_float16 = (flow_size_mb > FLOAT16_THRESHOLD_MB or 
                   available_mb < LOW_MEMORY_THRESHOLD_MB)
    
    if use_float16:
        wrapper = MemoryEfficientFlowArray(flows_array, use_float16=True)
        saved_mb = flow_size_mb - (wrapper.nbytes / (1024 * 1024))
        print(f"  💾 {label} flow compressed to float16: {wrapper.nbytes/(1024*1024):.0f} MB "
              f"(saved {saved_mb:.0f} MB)")
        # Force garbage collection to free the original float32 array
        del flows_array
        gc.collect()
        # Also clear CUDA cache to free any GPU memory from computation
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return wrapper
    else:
        return flows_array


# =============================================================================
# GPU MEMORY MANAGEMENT FOR RAFT
# =============================================================================

def get_available_gpu_memory_gb(device_idx=0):
    """Get available GPU memory in GB.
    
    Returns:
        float: Available GPU memory in GB, or 0 if CUDA not available
    """
    if not torch.cuda.is_available():
        return 0.0
    try:
        props = torch.cuda.get_device_properties(device_idx)
        reserved = torch.cuda.memory_reserved(device_idx)
        available = (props.total_memory - reserved) / (1024**3)
        return available
    except Exception:
        return 0.0


def estimate_raft_memory_gb(height, width):
    """Estimate GPU memory required for RAFT inference on given resolution.
    
    RAFT's memory usage scales quadratically with image size due to
    the correlation pyramid and feature maps. This is an empirical formula
    fitted to measurements on TITAN RTX (24GB) with torchvision RAFT.
    
    Args:
        height: Image height in pixels
        width: Image width in pixels
    
    Returns:
        float: Estimated peak GPU memory in GB
    """
    # Empirical formula derived from testing:
    # Resolution   -> Peak Memory
    # 512x512      -> 0.20 GB
    # 1024x1024    -> 1.92 GB  
    # 1280x1024    -> 2.90 GB
    # 1920x1080    -> 6.96 GB
    # 1920x1440    -> 12.17 GB
    # 2048x1440    -> 13.80 GB
    # 2048x1536    -> 15.65 GB
    # 2048x1600    -> 16.95 GB
    # 2048x1800    -> OOM (>24 GB)
    # 2048x2040    -> ~27 GB (OOM)
    # 
    # Fitted model: mem = 0.046 + 1.94e-7 * pixels + 1.52e-12 * pixels^2
    # This captures both linear overhead and quadratic scaling from correlation volume
    pixels = height * width
    estimated = 0.046 + 1.94e-7 * pixels + 1.52e-12 * (pixels ** 2)
    return estimated


def get_max_safe_resolution_for_raft(available_gpu_gb, original_height, original_width, safety_margin=0.85):
    """Calculate the maximum safe resolution for RAFT given available GPU memory.
    
    Args:
        available_gpu_gb: Available GPU memory in GB
        original_height: Original video height
        original_width: Original video width  
        safety_margin: Fraction of available memory to use (default 0.85 = 85%)
    
    Returns:
        tuple: (target_height, target_width, needs_resize)
            - Returns original dimensions if they fit
            - Returns scaled dimensions if resize is needed
            - needs_resize is True if dimensions were reduced
    """
    target_memory = available_gpu_gb * safety_margin
    
    # Check if original resolution fits
    estimated = estimate_raft_memory_gb(original_height, original_width)
    if estimated <= target_memory:
        return original_height, original_width, False
    
    # Need to scale down - find the scale factor
    # Memory model is approximately quadratic in pixels: mem ∝ pixels^2
    # Scaling dimensions by s reduces pixels by s^2, so memory by s^4
    # But we'll use iterative refinement since the model is more complex
    scale = (target_memory / estimated) ** 0.5  # Start with sqrt estimate
    
    # Apply scale and round to multiple of 8 (required by RAFT)
    new_height = max(64, int(original_height * scale) // 8 * 8)
    new_width = max(64, int(original_width * scale) // 8 * 8)
    
    # Verify the new resolution fits (iterate if needed due to rounding)
    for _ in range(5):  # Max 5 iterations for convergence
        new_estimated = estimate_raft_memory_gb(new_height, new_width)
        if new_estimated <= target_memory:
            break
        # Scale down more aggressively
        new_height = max(64, int(new_height * 0.9) // 8 * 8)
        new_width = max(64, int(new_width * 0.9) // 8 * 8)
    
    return new_height, new_width, True


def cleanup_gpu_memory():
    """Force cleanup of GPU memory after an error or when switching operations.
    
    This should be called in exception handlers to prevent memory leaks.
    """
    import gc
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        # Reset peak memory stats for future tracking
        try:
            torch.cuda.reset_peak_memory_stats()
        except Exception:
            pass


class MemoryMappedVideoAccessor:
    """Memory-efficient video accessor that reads patches on demand from TIFF files.
    
    Instead of loading the entire video into RAM (which can be 10-20 GB for large TIFFs),
    this class uses tifffile's memory-mapped access to read only the small patches needed
    for blob detection. This reduces RAM usage from ~20GB to a few MB.
    
    For AVI files, falls back to loading the video into memory (usually smaller).
    """
    
    def __init__(self, video_path, target_width=None, target_height=None):
        """
        Args:
            video_path: Path to video file (TIFF or AVI)
            target_width: Target width (if resizing needed)
            target_height: Target height (if resizing needed)
        """
        self.video_path = str(video_path)
        self.target_width = target_width
        self.target_height = target_height
        self._tiff = None
        self._memmap = None
        self._video_array = None  # Only used for AVI or when memmap fails
        self.is_avi = self.video_path.lower().endswith('.avi')
        self.shape = None
        self.ndim = 3
        self._needs_resize = False
        self._orig_shape = None
        
        # Cache for resized frames (avoids reloading + resizing for each patch)
        # Only used when _needs_resize is True
        self._frame_cache = {}  # {frame_idx: resized_frame}
        self._cache_max_size = 4  # Keep last 4 frames in cache
        self._cache_order = []  # LRU order
        
        self._initialize()
    
    def _initialize(self):
        """Initialize video access.
        
        For TIFFs, we first check if all pages are memmappable. If not all pages
        can be memory-mapped (common with ImageJ TIFFs having mixed page layouts),
        we load the entire video into memory once. This avoids the very slow
        tifffile.asarray(out='memmap') call that happens for such TIFFs.
        """
        import time
        t_start = time.time()
        
        if self.is_avi:
            # AVI: Load into memory (usually smaller than TIFFs)
            import mediapy as media
            video = media.read_video(self.video_path)
            if video.ndim == 4 and video.shape[-1] == 3:
                # Convert to grayscale
                self._video_array = np.mean(video[..., :3], axis=-1).astype(np.float32)
            else:
                self._video_array = video.astype(np.float32)
            self.shape = self._video_array.shape
            self._orig_shape = self.shape
            print(f"  Loaded AVI for blob detection: {self.shape} ({time.time()-t_start:.2f}s)")
        else:
            # TIFF: Check if fully memmappable before attempting memmap
            try:
                self._tiff = tifffile.TiffFile(self.video_path)
                
                # Check if ALL pages are memmappable
                # Mixed memmappability causes very slow asarray(out='memmap') calls
                all_memmappable = all(page.is_memmappable for page in self._tiff.pages)
                
                if all_memmappable:
                    # Fast path: true memory-mapped access
                    self._memmap = self._tiff.asarray(out='memmap')
                    
                    # Handle different TIFF layouts
                    if self._memmap.ndim == 4:
                        # RGB TIFF - we'll convert per-patch
                        self._orig_shape = self._memmap.shape[:3]
                        self.ndim = 4
                    else:
                        self._orig_shape = self._memmap.shape
                        self.ndim = 3
                    
                    self.shape = self._orig_shape
                    
                    # Check if resize is needed
                    if self.target_width and self.target_height:
                        if self.shape[2] != self.target_width or self.shape[1] != self.target_height:
                            self._needs_resize = True
                            self.shape = (self.shape[0], self.target_height, self.target_width)
                    
                    print(f"  Memory-mapped TIFF for blob detection: {self._orig_shape} -> {self.shape} ({time.time()-t_start:.3f}s)")
                else:
                    # Slow path: load into memory (but only do this once!)
                    # This is still faster than repeated slow memmap creation
                    print(f"  TIFF has mixed memmappable pages, loading into memory...")
                    self._tiff.close()
                    self._tiff = None
                    
                    vol = tifffile.imread(self.video_path)
                    if vol.ndim == 4:
                        vol = np.mean(vol[..., :3], axis=-1)
                    self._video_array = vol.astype(np.float32)
                    self.shape = self._video_array.shape
                    self._orig_shape = self.shape
                    
                    # Check if resize is needed
                    if self.target_width and self.target_height:
                        if self.shape[2] != self.target_width or self.shape[1] != self.target_height:
                            self._needs_resize = True
                            self.shape = (self.shape[0], self.target_height, self.target_width)
                    
                    print(f"  Loaded TIFF into memory for blob detection: {self._orig_shape} -> {self.shape} ({time.time()-t_start:.2f}s)")
                    
            except Exception as e:
                print(f"  Warning: Memory-mapped access failed ({e}), loading into memory")
                if self._tiff:
                    self._tiff.close()
                    self._tiff = None
                vol = tifffile.imread(self.video_path)
                if vol.ndim == 4:
                    vol = np.mean(vol[..., :3], axis=-1)
                self._video_array = vol.astype(np.float32)
                self.shape = self._video_array.shape
                self._orig_shape = self.shape
    
    def get_patch(self, t, y_min, y_max, x_min, x_max):
        """Get a patch from the video, converting to float32.
        
        Args:
            t: Frame index (0-indexed)
            y_min, y_max: Y range (exclusive end)
            x_min, x_max: X range (exclusive end)
            
        Returns:
            float32 patch of shape (y_max-y_min, x_max-x_min)
        """
        if self._video_array is not None:
            # In-memory array (AVI, mixed-memmappable TIFF, or fallback)
            if self._needs_resize:
                # Need to resize in-memory frame on demand
                import cv2
                if t in self._frame_cache:
                    # Use cached resized frame
                    if t in self._cache_order:
                        self._cache_order.remove(t)
                    self._cache_order.append(t)
                    return self._frame_cache[t][y_min:y_max, x_min:x_max]
                
                # Load and resize frame
                frame = self._video_array[t]
                frame = cv2.resize(frame.astype(np.float32), 
                                  (self.target_width, self.target_height),
                                  interpolation=cv2.INTER_LINEAR)
                
                # Cache the resized frame
                self._frame_cache[t] = frame
                self._cache_order.append(t)
                
                # Evict oldest frames if cache is full
                while len(self._cache_order) > self._cache_max_size:
                    oldest = self._cache_order.pop(0)
                    if oldest in self._frame_cache:
                        del self._frame_cache[oldest]
                
                return frame[y_min:y_max, x_min:x_max]
            else:
                # Direct access to in-memory array (no resize needed)
                patch = self._video_array[t, y_min:y_max, x_min:x_max]
                return patch.astype(np.float32) if patch.dtype != np.float32 else patch
        
        # Memory-mapped TIFF access
        if self._needs_resize:
            # Use cached resized frame if available (avoids expensive reload + resize)
            if t in self._frame_cache:
                # Move to end of LRU order
                if t in self._cache_order:
                    self._cache_order.remove(t)
                self._cache_order.append(t)
                return self._frame_cache[t][y_min:y_max, x_min:x_max]
            
            # Need to load full frame and resize
            import cv2
            if self.ndim == 4:
                frame = np.mean(self._memmap[t, :, :, :3], axis=-1)
            else:
                frame = self._memmap[t]
            
            frame = cv2.resize(frame.astype(np.float32), 
                              (self.target_width, self.target_height),
                              interpolation=cv2.INTER_LINEAR)
            
            # Cache the resized frame
            self._frame_cache[t] = frame
            self._cache_order.append(t)
            
            # Evict oldest frames if cache is full
            while len(self._cache_order) > self._cache_max_size:
                oldest = self._cache_order.pop(0)
                if oldest in self._frame_cache:
                    del self._frame_cache[oldest]
            
            return frame[y_min:y_max, x_min:x_max]
        
        # Direct memory-mapped access (most efficient)
        if self.ndim == 4:
            # RGB TIFF - convert patch to grayscale
            patch_rgb = self._memmap[t, y_min:y_max, x_min:x_max, :3]
            patch = np.mean(patch_rgb, axis=-1).astype(np.float32)
        else:
            patch = self._memmap[t, y_min:y_max, x_min:x_max].astype(np.float32)
        
        return patch
    
    def close(self):
        """Close the TIFF file handle."""
        if self._tiff is not None:
            self._tiff.close()
            self._tiff = None
        self._memmap = None
        self._video_array = None
        self._frame_cache.clear()
        self._cache_order.clear()
    
    def __del__(self):
        self.close()


# Performance optimization: Enable cudnn benchmarking for consistent input sizes
if torch.cuda.is_available():
    torch.backends.cudnn.benchmark = True
    # Enable TF32 for Ampere+ GPUs (RTX 30xx, 40xx) for faster computation
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

# Import the optimized tracking functions
# These are copied here to avoid external dependencies
# In production, you could import from raft_baseline_track_optimized


# =============================================================================
# OPTIMIZED UTILITY FUNCTIONS (from RAFT_v4.py)
# =============================================================================

def flow_to_color(flow, max_flow=None):
    """
    Convert optical flow to color visualization using HSV color space.
    Optimized with vectorized operations and cv2 fallback.
    
    Args:
        flow: (H, W, 2) array with (dx, dy) flow vectors
        max_flow: Maximum flow magnitude for normalization. If None, auto-scale.
    
    Returns:
        (H, W, 3) uint8 RGB image
    """
    H, W = flow.shape[:2]
    fx, fy = flow[:, :, 0], flow[:, :, 1]
    
    # OPTIMIZATION: Use hypot for faster magnitude calculation
    mag = np.hypot(fx, fy)
    ang = np.arctan2(fy, fx)
    
    # Create HSV image
    hsv = np.zeros((H, W, 3), dtype=np.uint8)
    
    # Hue encodes direction (0-360 degrees -> 0-180 in OpenCV)
    hsv[:, :, 0] = ((ang + np.pi) * (180.0 / (2 * np.pi))).astype(np.uint8)
    
    # Saturation is always maximum
    hsv[:, :, 1] = 255
    
    # Value encodes magnitude
    if max_flow is None:
        max_flow = mag.max()
    if max_flow > 0:
        # OPTIMIZATION: Use multiply instead of divide where possible
        hsv[:, :, 2] = np.clip(mag * (255.0 / max_flow), 0, 255).astype(np.uint8)
    else:
        hsv[:, :, 2] = 0
    
    # Convert HSV to RGB - prefer cv2 for speed
    try:
        import cv2
        rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
    except ImportError:
        # Fallback: vectorized HSV to RGB conversion without cv2
        # This is a proper vectorized implementation
        h = hsv[:, :, 0].astype(np.float32) / 180.0  # 0-1 range
        s = hsv[:, :, 1].astype(np.float32) / 255.0
        v = hsv[:, :, 2].astype(np.float32) / 255.0
        
        c = v * s
        x = c * (1 - np.abs((h * 6) % 2 - 1))
        m = v - c
        
        hi = (h * 6).astype(np.int32) % 6
        
        rgb = np.zeros((H, W, 3), dtype=np.float32)
        
        mask0 = (hi == 0)
        mask1 = (hi == 1)
        mask2 = (hi == 2)
        mask3 = (hi == 3)
        mask4 = (hi == 4)
        mask5 = (hi == 5)
        
        rgb[mask0] = np.stack([c[mask0], x[mask0], np.zeros_like(c[mask0])], axis=-1)
        rgb[mask1] = np.stack([x[mask1], c[mask1], np.zeros_like(c[mask1])], axis=-1)
        rgb[mask2] = np.stack([np.zeros_like(c[mask2]), c[mask2], x[mask2]], axis=-1)
        rgb[mask3] = np.stack([np.zeros_like(c[mask3]), x[mask3], c[mask3]], axis=-1)
        rgb[mask4] = np.stack([x[mask4], np.zeros_like(c[mask4]), c[mask4]], axis=-1)
        rgb[mask5] = np.stack([c[mask5], np.zeros_like(c[mask5]), x[mask5]], axis=-1)
        
        rgb = ((rgb + m[:, :, np.newaxis]) * 255).astype(np.uint8)
    
    return rgb


@lru_cache(maxsize=128)
def _get_tensor_shape_cache(H, W):
    """Cache tensor shapes for reuse."""
    return (3, H, W)


def _to_3ch_tensor01(frame_hw: np.ndarray) -> torch.Tensor:
    """OPTIMIZED (RAFT_v4): (H,W)[0..1] -> (3,H,W) float32 tensor on CPU.
    
    Converts normalized [0,1] grayscale to 3-channel tensor in [-1,1] range
    as required by torchvision RAFT models.
    """
    assert frame_hw.ndim == 2, f"expected (H,W), got {frame_hw.shape}"
    t = torch.from_numpy(frame_hw).float().clamp_(0.0, 1.0)
    # Convert [0, 1] to [-1, 1] as required by torchvision RAFT
    t = t * 2.0 - 1.0
    return torch.stack([t, t, t], dim=0)


def _to_3ch_tensor_avi(frame_hwc: np.ndarray) -> torch.Tensor:
    """Convert (H,W,3) uint8 [0..255] -> (3,H,W) float32 [-1..1] for AVI files.
    
    Match the original DeepMind implementation exactly: / 127.5 - 1.0
    If input is (H,W), replicate to 3 channels.
    """
    if frame_hwc.ndim == 2:
        # Grayscale - replicate to 3 channels
        frame_hwc = np.stack([frame_hwc, frame_hwc, frame_hwc], axis=-1)
    
    assert frame_hwc.ndim == 3 and frame_hwc.shape[-1] == 3, f"expected (H,W,3), got {frame_hwc.shape}"
    
    # EXACT match to DeepMind: / 127.5 - 1.0 -> [-1, 1]
    t = torch.from_numpy(frame_hwc.astype(np.float32))
    t = t / 127.5 - 1.0
    
    # Transpose to (3, H, W)
    t = t.permute(2, 0, 1)
    return t


def pad_to_multiple(x: torch.Tensor, mult: int = 8):
    """Pad tensor to multiple of mult."""
    H, W = x.shape[-2], x.shape[-1]
    ph = (mult - (H % mult)) % mult
    pw = (mult - (W % mult)) % mult
    if ph or pw:
        x = F.pad(x, (0, pw, 0, ph), mode="replicate")
    return x, ph, pw


_OFFSET_CACHE = {}


class SegmentCache:
    """Shared LRU cache for anchor segment interpolations."""

    def __init__(self, max_size=10000, rounding=0.25):
        self.max_size = max_size
        self.rounding = rounding
        self._store = OrderedDict()
        self.hits = 0
        self.misses = 0
        self.cache_access_count = {}  # OPTIMIZATION (RAFT_v4): Track access frequency for true LRU

    def __len__(self):
        return len(self._store)

    def _round(self, value: float) -> float:
        # Use configurable rounding parameter for cache key snapping
        if self.rounding <= 0:
            return round(float(value), 3)
        snapped = round(float(value) / self.rounding) * self.rounding
        return round(snapped, 4)

    def make_key(self, t0, x0, y0, t1, x1, y1, corridor_width):
        corridor_token = "auto" if corridor_width is None else float(corridor_width)
        return (
            int(t0),
            self._round(x0),
            self._round(y0),
            int(t1),
            self._round(x1),
            self._round(y1),
            corridor_token,
        )

    def clear(self):
        self._store.clear()
        self.hits = 0
        self.misses = 0
        self.cache_access_count.clear()  # OPTIMIZATION (RAFT_v4): Clear access counts too

    def get(self, key):
        try:
            value = self._store.pop(key)
        except KeyError:
            self.misses += 1
            return None
        self._store[key] = value
        self.hits += 1
        # OPTIMIZATION (RAFT_v4): Track access count for LRU statistics
        self.cache_access_count[key] = self.cache_access_count.get(key, 0) + 1
        return value

    def put(self, key, value):
        if key in self._store:
            self._store.pop(key)
        self._store[key] = value
        # OPTIMIZATION (RAFT_v4): Initialize access count for new entries
        self.cache_access_count[key] = 1
        if len(self._store) > self.max_size:
            # Evict least recently used (first in OrderedDict)
            evicted_key = self._store.popitem(last=False)[0]
            # Clean up access count
            if evicted_key in self.cache_access_count:
                del self.cache_access_count[evicted_key]

    def stats(self):
        accesses = self.hits + self.misses
        hit_rate = self.hits / accesses if accesses else 0.0
        # OPTIMIZATION (RAFT_v4): Include access count statistics
        total_accesses = sum(self.cache_access_count.values()) if self.cache_access_count else 0
        return {
            "segments_cached": len(self._store),
            "cache_hits": self.hits,
            "cache_misses": self.misses,
            "cache_hit_rate": hit_rate,
            "rounding": self.rounding,
            "total_cache_accesses": total_accesses,
            "most_accessed_count": max(self.cache_access_count.values()) if self.cache_access_count else 0,
        }


def _get_offset_grid(radius: int, device, dtype):
    """Get cached offset grid for given radius."""
    key = (radius, device, dtype)
    if key not in _OFFSET_CACHE:
        window = 2 * radius + 1
        ys = torch.arange(-radius, radius + 1, device=device, dtype=dtype)
        xs = torch.arange(-radius, radius + 1, device=device, dtype=dtype)
        yy, xx = torch.meshgrid(ys, xs, indexing="ij")
        K = window * window
        offsets = torch.stack([xx, yy], dim=-1).reshape(K, 2)
        _OFFSET_CACHE[key] = (offsets, -offsets, window, K)
    return _OFFSET_CACHE[key]


def _clamp_xy(x, y, W, H):
    """Clamp coordinates to valid range."""
    return int(np.clip(round(x), 0, W-1)), int(np.clip(round(y), 0, H-1))


# =============================================================================
# FLOW-BASED TRACK CORRECTION (SEGMENT BLENDING)
# =============================================================================

def segment_flow_blend(flows, anchors, flow_scale=1, input_dims=None, cancel_check=None, 
                        linear_interp_threshold=0):
    """
    Build track by blending forward and backward flow-based propagation.
    
    This approach blends the entire segment between anchors, which improves
    accuracy when the trajectory between anchors has drifted significantly.
    
    Performance comparison (tested on real trajectory):
    - Flow-only baseline: 65.9% accuracy
    - Original DP interpolation with anchor: 41.5% (WORSE than baseline!)
    - This method (full blend): 77.2% (corrects entire segment)
    
    Linear Interpolation Threshold:
        If two anchors are within N frames apart (where N = linear_interp_threshold),
        this function uses simple linear interpolation instead of optical flow for
        that segment. This helps avoid jitter/erratic displacement from unreliable
        optical flow in short segments where the flow field may be noisy.
    
    Args:
        flows: (T-1, flow_H, flow_W, 2) optical flow array, flow[t] is t -> t+1
               For DIS with downsample, these are the reduced dimensions.
        anchors: list of (frame, x, y) anchor points in INPUT coordinates
        flow_scale: Scale factor for converting input coords to flow coords.
                    Input coords / flow_scale = flow coords for lookup.
        input_dims: (H, W) of the input coordinate space. If None, uses flow * scale.
        linear_interp_threshold: If segment length (t1 - t0) <= this value, use 
                                  linear interpolation instead of optical flow.
                                  0 = disabled (always use optical flow).
        
    Returns:
        track: (T, 2) float32 array of (x, y) positions in INPUT coordinates
    """
    Tm1, flow_H, flow_W, _ = flows.shape
    T = Tm1 + 1
    
    # Determine input dimensions
    if input_dims is not None:
        input_H, input_W = input_dims
    else:
        input_H = flow_H * flow_scale
        input_W = flow_W * flow_scale
    
    if not anchors:
        return np.zeros((T, 2), dtype=np.float32)
    
    anchors = sorted(set((int(t), float(x), float(y)) for (t, x, y) in anchors), key=lambda a: a[0])
    all_pos = np.zeros((T, 2), dtype=np.float32)
    
    def get_flow(t, x, y):
        """Get flow at position (x, y) in input coords, handling scaling."""
        flow_x = int(np.clip(round(x / flow_scale), 0, flow_W - 1))
        flow_y = int(np.clip(round(y / flow_scale), 0, flow_H - 1))
        return flows[t, flow_y, flow_x]
    
    # Process segments between consecutive anchors
    for (t0, x0, y0), (t1, x1, y1) in zip(anchors[:-1], anchors[1:]):
        if cancel_check is not None:
            cancel_check()
        if t1 <= t0:
            continue
        
        segment_len = t1 - t0 + 1
        segment_gap = t1 - t0  # Number of frames between anchors (exclusive)
        
        # Check if this segment should use linear interpolation
        # If the gap between anchors is <= threshold, optical flow is unreliable
        use_linear_interp = (linear_interp_threshold > 0 and segment_gap <= linear_interp_threshold)
        
        if use_linear_interp:
            # Simple linear interpolation between anchors
            # More reliable than optical flow for short/noisy segments
            alpha = np.linspace(0, 1, segment_len)
            all_pos[t0:t1+1, 0] = (1 - alpha) * x0 + alpha * x1
            all_pos[t0:t1+1, 1] = (1 - alpha) * y0 + alpha * y1
        else:
            # Forward propagation from start anchor
            forward = np.zeros((segment_len, 2), dtype=np.float32)
            pos = np.array([x0, y0], dtype=np.float32)
            for i, t in enumerate(range(t0, t1 + 1)):
                if cancel_check is not None and (i % 10 == 0):
                    cancel_check()
                forward[i] = pos.copy()
                if t < t1:
                    dx, dy = get_flow(t, pos[0], pos[1])
                    pos[0] += float(dx)
                    pos[1] += float(dy)
            
            # Backward propagation from end anchor
            backward = np.zeros((segment_len, 2), dtype=np.float32)
            pos = np.array([x1, y1], dtype=np.float32)
            for i, t in enumerate(range(t1, t0 - 1, -1)):
                if cancel_check is not None and (i % 10 == 0):
                    cancel_check()
                backward[segment_len - 1 - (t1 - t)] = pos.copy()
                if t > t0:
                    dx, dy = get_flow(t - 1, pos[0], pos[1])
                    pos[0] -= float(dx)
                    pos[1] -= float(dy)
            
            # Linear blend: 100% forward at start, 100% backward at end
            alpha = np.linspace(0, 1, segment_len)
            blended = (1 - alpha[:, None]) * forward + alpha[:, None] * backward
            all_pos[t0:t1+1] = blended
    
    # Backward propagation: frames before first anchor
    # Note: Linear interp threshold doesn't apply here (only one anchor, no choice but flow)
    t_first, x_first, y_first = anchors[0]
    if t_first > 0:
        pos = np.array([x_first, y_first], dtype=np.float32)
        for t in range(t_first - 1, -1, -1):
            if cancel_check is not None and (t % 10 == 0):
                cancel_check()
            dx, dy = get_flow(t, pos[0], pos[1])
            pos[0] -= float(dx)
            pos[1] -= float(dy)
            all_pos[t] = pos.copy()
    
    # Forward propagation: frames after last anchor
    # Note: Linear interp threshold doesn't apply here (only one anchor, no choice but flow)
    t_last, x_last, y_last = anchors[-1]
    pos = np.array([x_last, y_last], dtype=np.float32)
    for t in range(t_last, T):
        if cancel_check is not None and (t % 10 == 0):
            cancel_check()
        all_pos[t] = pos.copy()
        if t < T - 1:
            dx, dy = get_flow(t, pos[0], pos[1])
            pos[0] += float(dx)
            pos[1] += float(dy)
    
    return all_pos


class FlowBlendTrackBuilder:
    """
    Track builder using flow-based segment blending.
    
    Blends forward and backward flow-based propagation between anchors
    for accurate trajectory reconstruction.
    
    Supports flow stored at reduced resolution (e.g., DIS with downsample).
    
    Linear Interpolation Threshold:
        If two anchors are within N frames apart (where N = linear_interp_threshold),
        the segment uses simple linear interpolation instead of optical flow.
        This helps avoid jitter/erratic displacement from unreliable optical flow
        in short segments where the flow field may be noisy.
    """
    
    def __init__(self, T, H, W, optical_flows_np, mode=None, correction_radius=None,
                 flow_scale=1, input_dims=None, cancel_check=None, linear_interp_threshold=0):
        """
        Args:
            T: Number of frames
            H, W: Input (display) dimensions
            optical_flows_np: Flow array (may be at reduced resolution for DIS)
            mode: Kept for API compatibility
            correction_radius: Kept for API compatibility
            flow_scale: Scale factor for DIS flow (input_coords / flow_scale = flow_coords)
            input_dims: (H, W) of input coordinates, if different from H, W
            linear_interp_threshold: If two anchors are within this many frames,
                                      use linear interpolation instead of optical flow.
                                      0 = disabled (always use optical flow).
        """
        self.T = T
        self.H = H
        self.W = W
        self.optical_flows_np = optical_flows_np
        self.flow_scale = flow_scale
        self.input_dims = input_dims if input_dims else (H, W)
        self.cancel_check = cancel_check
        self.linear_interp_threshold = linear_interp_threshold
        # mode and correction_radius kept for API compatibility but not used
    
    def build_track(self, anchors):
        """Build track from anchors using flow-based segment blending."""
        if not anchors:
            raise ValueError("Provide at least one anchor, e.g. (t, x, y).")
        
        track_float = segment_flow_blend(
            self.optical_flows_np, anchors,
            flow_scale=self.flow_scale,
            input_dims=self.input_dims,
            cancel_check=self.cancel_check,
            linear_interp_threshold=self.linear_interp_threshold,
        )
        
        # Convert to int32 with clamping
        input_H, input_W = self.input_dims
        track_int = np.zeros((self.T, 2), dtype=np.int32)
        track_int[:, 0] = np.clip(np.round(track_float[:, 0]), 0, input_W - 1).astype(np.int32)
        track_int[:, 1] = np.clip(np.round(track_float[:, 1]), 0, input_H - 1).astype(np.int32)
        
        return track_int
    
    def get_cache_stats(self):
        """Return placeholder stats for compatibility."""
        return {
            "segments_cached": 0,
            "cache_hit_rate": 0.0,
            "method": "flow_blend"
        }


def segment_flow_blend_blob(flows, anchors, video_frames_or_accessor, flow_scale=1, input_dims=None,
                             search_radius=15, blob_radius=5.0, is_rgb=False, frame_shape=None, cancel_check=None):
    """
    Build track using flow+blob propagation between anchors.
    
    This method uses exactly the same propagation logic as rough track creation
    with blob detection (_propagate_with_flows_blob), but applies it between anchors:
    
    1. For each segment between consecutive anchors:
       - Forward propagation with blob detection from start anchor to mid-point
       - Backward propagation with blob detection from end anchor to mid-point
       - No blending - each half uses the propagated positions directly
    2. Before first anchor: backward propagation with blob detection
    3. After last anchor: forward propagation with blob detection
    
    Args:
        flows: (T-1, flow_H, flow_W, 2) optical flow array, flow[t] is t -> t+1
        anchors: list of (frame, x, y) anchor points in INPUT coordinates
        video_frames_or_accessor: Either numpy array (T, H, W) or (T, H, W, 3), 
                                   or a MemoryMappedVideoAccessor for efficient access
        flow_scale: Scale factor for converting input coords to flow coords
        input_dims: (H, W) of the input coordinate space
        search_radius: Radius around flow-predicted position to search for blobs
        blob_radius: Expected radius of blob for detection (in pixels)
        is_rgb: Whether video frames are RGB (only used for numpy arrays)
        frame_shape: (H, W) of frames (required if using accessor)
        
    Returns:
        track: (T, 2) float32 array of (x, y) positions in INPUT coordinates
    """
    if not HAS_SKIMAGE:
        print("WARNING: skimage not available, falling back to pure flow blend")
        return segment_flow_blend(flows, anchors, flow_scale, input_dims, cancel_check=cancel_check)
    
    Tm1, flow_H, flow_W, _ = flows.shape
    T = Tm1 + 1
    
    # Determine input dimensions
    if input_dims is not None:
        input_H, input_W = input_dims
    else:
        input_H = flow_H * flow_scale
        input_W = flow_W * flow_scale
    
    # Determine frame dimensions for blob detection
    if frame_shape is not None:
        frame_H, frame_W = frame_shape
    elif hasattr(video_frames_or_accessor, 'shape'):
        if video_frames_or_accessor.ndim == 4:
            _, frame_H, frame_W, _ = video_frames_or_accessor.shape
        else:
            _, frame_H, frame_W = video_frames_or_accessor.shape
    else:
        # Accessor with shape attribute
        frame_H, frame_W = video_frames_or_accessor.shape[1], video_frames_or_accessor.shape[2]
    
    # Check if using memory-mapped accessor or numpy array
    use_accessor = hasattr(video_frames_or_accessor, 'get_patch')
    frames = None if use_accessor else video_frames_or_accessor
    accessor = video_frames_or_accessor if use_accessor else None
    
    if not anchors:
        return np.zeros((T, 2), dtype=np.float32)
    
    anchors = sorted(set((int(t), float(x), float(y)) for (t, x, y) in anchors), key=lambda a: a[0])
    all_pos = np.zeros((T, 2), dtype=np.float32)
    
    # Blob detection parameters (same as _propagate_with_flows_blob)
    sigma_center = blob_radius / np.sqrt(2.0)
    min_sigma = max(0.5, sigma_center * 0.7)
    max_sigma = sigma_center * 1.3
    
    def get_flow(t, x, y):
        """Get flow at position (x, y) in input coords, handling scaling."""
        flow_x = int(np.clip(round(x / flow_scale), 0, flow_W - 1))
        flow_y = int(np.clip(round(y / flow_scale), 0, flow_H - 1))
        return flows[t, flow_y, flow_x]
    
    def find_blob_position(t_frame, x_pred, y_pred):
        """Search for blob near predicted position (same logic as _propagate_with_flows_blob)."""
        r = int(search_radius)
        y_min = max(0, int(y_pred) - r)
        y_max = min(frame_H, int(y_pred) + r + 1)
        x_min = max(0, int(x_pred) - r)
        x_max = min(frame_W, int(x_pred) + r + 1)
        
        # Get patch (from accessor or numpy array)
        if use_accessor:
            patch = accessor.get_patch(t_frame, y_min, y_max, x_min, x_max)
        else:
            if is_rgb:
                patch_raw = frames[t_frame, y_min:y_max, x_min:x_max, :3]
                patch = np.mean(patch_raw, axis=-1).astype(np.float32)
            else:
                patch = frames[t_frame, y_min:y_max, x_min:x_max].astype(np.float32)
        
        if patch.size == 0:
            return x_pred, y_pred
        
        # Normalize patch for blob detection
        pmin, pmax = float(patch.min()), float(patch.max())
        patch_norm = (patch - pmin) / (pmax - pmin) if pmax > pmin else np.zeros_like(patch)
        
        chosen = None
        y_local = y_pred - y_min
        x_local = x_pred - x_min
        
        # Try blob_log detection first
        try:
            blobs = blob_log(
                patch_norm,
                min_sigma=min_sigma,
                max_sigma=max_sigma,
                num_sigma=5,
                threshold=0.02,
            )
            
            if len(blobs) > 0:
                # Find closest blob to predicted position
                dists = [np.hypot(by - y_local, bx - x_local) for (by, bx, _) in blobs]
                by, bx, _ = blobs[int(np.argmin(dists))]
                chosen = (y_min + by, x_min + bx)
        except Exception:
            pass
        
        # Fallback to region-based detection if no blob found
        if chosen is None:
            try:
                thresh = np.percentile(patch_norm, 70)
                mask = patch_norm > thresh
                if mask.any():
                    labels = measure.label(mask)
                    regions = measure.regionprops(labels)
                    candidates = []
                    for reg in regions:
                        eqr = reg.equivalent_diameter / 2
                        if abs(eqr - blob_radius) <= blob_radius * 0.5:
                            cy, cx = reg.centroid
                            candidates.append((np.hypot(cy - y_local, cx - x_local), cy, cx))
                    if candidates:
                        _, cy, cx = min(candidates)
                        chosen = (y_min + cy, x_min + cx)
            except Exception:
                pass
        
        if chosen is not None:
            return chosen[1], chosen[0]  # Return (x, y)
        else:
            return x_pred, y_pred
    
    # Process segments between consecutive anchors
    # Use the SAME propagation logic as _propagate_with_flows_blob:
    # - Forward propagation fills from start anchor to mid-point
    # - Backward propagation fills from end anchor to mid-point
    for (t0, x0, y0), (t1, x1, y1) in zip(anchors[:-1], anchors[1:]):
        if cancel_check is not None:
            cancel_check()
        if t1 <= t0:
            continue
        
        # Calculate mid-point frame (forward fills up to mid, backward fills from mid+1)
        mid_t = (t0 + t1) // 2
        
        # Forward propagation from start anchor (t0) to mid_t
        # Same logic as forward loop in _propagate_with_flows_blob
        pos = np.array([x0, y0], dtype=np.float32)
        for t in range(t0, mid_t + 1):
            if cancel_check is not None and (t % 10 == 0):
                cancel_check()
            # Record position
            all_pos[t] = pos.copy()
            
            if t < mid_t:
                # Get flow prediction
                dx, dy = get_flow(t, pos[0], pos[1])
                
                # Flow-predicted position
                x_pred = np.clip(pos[0] + float(dx), 0, input_W - 1)
                y_pred = np.clip(pos[1] + float(dy), 0, input_H - 1)
                
                # Refine with blob detection
                x_refined, y_refined = find_blob_position(t + 1, x_pred, y_pred)
                pos[0] = float(x_refined)
                pos[1] = float(y_refined)
        
        # Backward propagation from end anchor (t1) to mid_t+1
        # Same logic as backward loop in _propagate_with_flows_blob
        pos = np.array([x1, y1], dtype=np.float32)
        for t in range(t1, mid_t, -1):
            if cancel_check is not None and (t % 10 == 0):
                cancel_check()
            # Record position
            all_pos[t] = pos.copy()
            
            if t > mid_t + 1:
                # Iterative backward flow estimation (3 iterations as in original)
                x_est = pos[0]
                y_est = pos[1]
                
                for _ in range(3):
                    dx, dy = get_flow(t - 1, x_est, y_est)
                    x_est = np.clip(pos[0] - float(dx), 0, input_W - 1)
                    y_est = np.clip(pos[1] - float(dy), 0, input_H - 1)
                
                # Refine with blob detection
                x_refined, y_refined = find_blob_position(t - 1, x_est, y_est)
                pos[0] = float(x_refined)
                pos[1] = float(y_refined)
    
    # Backward propagation: frames before first anchor
    # Same logic as backward propagation in _propagate_with_flows_blob
    t_first, x_first, y_first = anchors[0]
    if t_first > 0:
        pos = np.array([x_first, y_first], dtype=np.float32)
        for t in range(t_first - 1, -1, -1):
            if cancel_check is not None and (t % 10 == 0):
                cancel_check()
            # Iterative backward flow estimation
            x_est = pos[0]
            y_est = pos[1]
            
            for _ in range(3):
                dx, dy = get_flow(t, x_est, y_est)
                x_est = np.clip(pos[0] - float(dx), 0, input_W - 1)
                y_est = np.clip(pos[1] - float(dy), 0, input_H - 1)
            
            # Refine with blob detection
            x_refined, y_refined = find_blob_position(t, x_est, y_est)
            pos[0] = float(x_refined)
            pos[1] = float(y_refined)
            
            # Record position
            all_pos[t] = pos.copy()
    
    # Forward propagation: frames after last anchor
    # Same logic as forward propagation in _propagate_with_flows_blob
    t_last, x_last, y_last = anchors[-1]
    pos = np.array([x_last, y_last], dtype=np.float32)
    for t in range(t_last, T):
        if cancel_check is not None and (t % 10 == 0):
            cancel_check()
        # Record position
        all_pos[t] = pos.copy()
        
        if t < T - 1:
            # Get flow prediction
            dx, dy = get_flow(t, pos[0], pos[1])
            
            # Flow-predicted position
            x_pred = np.clip(pos[0] + float(dx), 0, input_W - 1)
            y_pred = np.clip(pos[1] + float(dy), 0, input_H - 1)
            
            # Refine with blob detection
            x_refined, y_refined = find_blob_position(t + 1, x_pred, y_pred)
            pos[0] = float(x_refined)
            pos[1] = float(y_refined)
    
    return all_pos


class FlowBlendBlobTrackBuilder:
    """
    Track builder using flow-based segment blending with blob detection refinement.
    
    Combines forward and backward flow+blob propagation between anchors
    for more accurate trajectory reconstruction when tracking particle-like objects.
    
    Supports flow stored at reduced resolution (e.g., DIS with downsample).
    """
    
    def __init__(self, T, H, W, optical_flows_np, video_frames_or_accessor,
                 flow_scale=1, input_dims=None, search_radius=15, blob_radius=5.0,
                 is_rgb=False, frame_shape=None, cancel_check=None):
        """
        Args:
            T: Number of frames
            H, W: Input (display) dimensions
            optical_flows_np: Flow array (may be at reduced resolution for DIS)
            video_frames_or_accessor: Video frames or accessor for blob detection
            flow_scale: Scale factor for DIS flow
            input_dims: (H, W) of input coordinates
            search_radius: Radius around flow-predicted position to search for blobs
            blob_radius: Expected radius of blob for detection
            is_rgb: Whether video frames are RGB
            frame_shape: (H, W) of frames (required if using accessor)
        """
        self.T = T
        self.H = H
        self.W = W
        self.optical_flows_np = optical_flows_np
        self.video_frames_or_accessor = video_frames_or_accessor
        self.flow_scale = flow_scale
        self.input_dims = input_dims if input_dims else (H, W)
        self.search_radius = search_radius
        self.blob_radius = blob_radius
        self.is_rgb = is_rgb
        self.frame_shape = frame_shape
        self.cancel_check = cancel_check
    
    def build_track(self, anchors):
        """Build track from anchors using flow+blob segment blending."""
        if not anchors:
            raise ValueError("Provide at least one anchor, e.g. (t, x, y).")
        
        track_float = segment_flow_blend_blob(
            self.optical_flows_np, anchors, self.video_frames_or_accessor,
            flow_scale=self.flow_scale,
            input_dims=self.input_dims,
            search_radius=self.search_radius,
            blob_radius=self.blob_radius,
            is_rgb=self.is_rgb,
            frame_shape=self.frame_shape,
            cancel_check=self.cancel_check,
        )
        
        # Convert to int32 with clamping
        input_H, input_W = self.input_dims
        track_int = np.zeros((self.T, 2), dtype=np.int32)
        track_int[:, 0] = np.clip(np.round(track_float[:, 0]), 0, input_W - 1).astype(np.int32)
        track_int[:, 1] = np.clip(np.round(track_float[:, 1]), 0, input_H - 1).astype(np.int32)
        
        return track_int
    
    def get_cache_stats(self):
        """Return placeholder stats for compatibility."""
        return {
            "segments_cached": 0,
            "cache_hit_rate": 0.0,
            "method": "flow_blend_blob"
        }


class CorridorDPTrackBuilder:
    """
    Track builder using Corridor DP interpolation between anchors.
    
    This is the original RAFT_v4 optimized implementation that uses
    dynamic programming with adaptive corridor search for accurate
    trajectory reconstruction between anchor points.
    
    Features:
    - Cached segment interpolations to avoid redundant computation
    - Adaptive corridor width based on flow predictions
    - Bidirectional propagation support
    
    For DIS flow (flow_scale > 1):
    - Coordinates are scaled to flow resolution for DP
    - Flow values are divided by flow_scale (DIS stores pre-scaled values)
    - Results are scaled back to input resolution
    
    Linear Interpolation Threshold:
        If two anchors are within N frames apart (where N = linear_interp_threshold),
        the segment uses simple linear interpolation instead of DP optimization.
        This helps avoid jitter from unreliable optical flow in short segments.
    """
    
    def __init__(self, T, H, W, optical_flows_np, segment_cache=None,
                 corridor_width=None, flow_scale=1, input_dims=None, cancel_check=None,
                 linear_interp_threshold=0):
        """
        Args:
            T: Number of frames
            H, W: Input (display) dimensions
            optical_flows_np: Flow array (may be MemoryEfficientFlowArray or numpy array)
            segment_cache: Shared SegmentCache for caching interpolations
            corridor_width: Corridor width for DP (None=adaptive, 0=full frame)
            flow_scale: Scale factor for DIS flow (input_coords / flow_scale = flow_coords)
            input_dims: (H, W) of input coordinates, if different from H, W
            linear_interp_threshold: If two anchors are within this many frames,
                                      use linear interpolation instead of DP.
                                      0 = disabled (always use DP).
        """
        self.T = T
        self.H = H
        self.W = W
        self.flow_scale = flow_scale
        self.input_dims = input_dims if input_dims else (H, W)
        self.corridor_width = corridor_width
        self.cancel_check = cancel_check
        self.linear_interp_threshold = linear_interp_threshold
        
        # Handle MemoryEfficientFlowArray - convert to numpy for DP operations
        if isinstance(optical_flows_np, MemoryEfficientFlowArray):
            self.optical_flows_np = optical_flows_np.as_float32()
        else:
            self.optical_flows_np = optical_flows_np
        
        # For DIS flow, the flow values are pre-scaled to input coordinates.
        # The DP algorithm works in flow coordinate space, so we need to
        # scale down the flow values to match the flow resolution.
        if flow_scale != 1:
            # Create a copy with scaled flow values for DP
            self.optical_flows_dp = self.optical_flows_np.copy()
            self.optical_flows_dp[..., 0] /= flow_scale
            self.optical_flows_dp[..., 1] /= flow_scale
        else:
            self.optical_flows_dp = self.optical_flows_np
        
        # Use shared segment cache or create new one
        if segment_cache is None:
            self.segment_cache = SegmentCache(max_size=10000, rounding=0.25)
        else:
            self.segment_cache = segment_cache
        
        # Create the underlying IncrementalTrackBuilder at FLOW resolution
        flow_H, flow_W = self.optical_flows_np.shape[1], self.optical_flows_np.shape[2]
        self._builder = IncrementalTrackBuilder(
            T,
            flow_H,
            flow_W,
            self.optical_flows_dp,
            interpolate,
            corridor_width=corridor_width,
            segment_cache=self.segment_cache,
            cancel_check=self.cancel_check,
        )
    
    def build_track(self, anchors):
        """Build track from anchors using Corridor DP interpolation.
        
        If linear_interp_threshold > 0 and a segment between consecutive anchors
        has a gap <= threshold, that segment uses simple linear interpolation
        instead of DP optimization. This helps avoid jitter in short segments
        where optical flow may be unreliable.
        """
        if not anchors:
            raise ValueError("Provide at least one anchor, e.g. (t, x, y).")
        
        # Normalize anchors to consistent format
        anchors = sorted(set((int(t), float(x), float(y)) for (t, x, y) in anchors), key=lambda a: a[0])
        
        input_H, input_W = self.input_dims
        flow_H, flow_W = self.optical_flows_np.shape[1], self.optical_flows_np.shape[2]
        
        # Check if we need to handle linear interpolation for short segments
        if self.linear_interp_threshold > 0 and len(anchors) > 1:
            # Find which segments should use linear interpolation
            linear_segments = []  # List of (t0, x0, y0, t1, x1, y1) for linear interp
            dp_anchors = []  # Anchors for DP interpolation (excluding linear segments)
            
            # Check each segment
            prev_anchor = anchors[0]
            for anchor in anchors[1:]:
                t0, x0, y0 = prev_anchor
                t1, x1, y1 = anchor
                segment_gap = t1 - t0
                
                if segment_gap <= self.linear_interp_threshold:
                    # This segment should use linear interpolation
                    linear_segments.append((t0, x0, y0, t1, x1, y1))
                    # Don't add prev_anchor to dp_anchors (it's part of linear segment)
                    # But we need to handle the case where the next segment IS a DP segment
                else:
                    # Normal DP segment - include the start anchor
                    if not dp_anchors or dp_anchors[-1] != prev_anchor:
                        dp_anchors.append(prev_anchor)
                
                prev_anchor = anchor
            
            # Make sure the last anchor is included for DP if not part of linear segment
            if dp_anchors and anchors[-1] != dp_anchors[-1]:
                # Check if last segment was DP or linear
                if not linear_segments or linear_segments[-1][3:6] != anchors[-1]:
                    dp_anchors.append(anchors[-1])
            
            # If all segments are linear, we don't need DP at all
            if not dp_anchors or len(dp_anchors) < 2:
                # All segments use linear interpolation
                track_int = np.zeros((self.T, 2), dtype=np.int32)
                
                # Fill linear segments
                for (t0, x0, y0, t1, x1, y1) in linear_segments:
                    segment_len = t1 - t0 + 1
                    alpha = np.linspace(0, 1, segment_len)
                    track_int[t0:t1+1, 0] = np.clip(np.round((1 - alpha) * x0 + alpha * x1), 0, input_W - 1).astype(np.int32)
                    track_int[t0:t1+1, 1] = np.clip(np.round((1 - alpha) * y0 + alpha * y1), 0, input_H - 1).astype(np.int32)
                
                # For frames outside anchor range, use flow propagation
                # (Delegate to segment_flow_blend for consistency)
                track_float = segment_flow_blend(
                    self.optical_flows_np, anchors,
                    flow_scale=self.flow_scale,
                    input_dims=self.input_dims,
                    cancel_check=self.cancel_check,
                    linear_interp_threshold=self.linear_interp_threshold,
                )
                
                # Copy flow-based results for frames outside anchor range
                t_first = anchors[0][0]
                t_last = anchors[-1][0]
                if t_first > 0:
                    track_int[:t_first, 0] = np.clip(np.round(track_float[:t_first, 0]), 0, input_W - 1).astype(np.int32)
                    track_int[:t_first, 1] = np.clip(np.round(track_float[:t_first, 1]), 0, input_H - 1).astype(np.int32)
                if t_last < self.T - 1:
                    track_int[t_last+1:, 0] = np.clip(np.round(track_float[t_last+1:, 0]), 0, input_W - 1).astype(np.int32)
                    track_int[t_last+1:, 1] = np.clip(np.round(track_float[t_last+1:, 1]), 0, input_H - 1).astype(np.int32)
                
                return track_int
        
        # Standard DP path (no linear interpolation or all segments are DP)
        if self.flow_scale != 1:
            # Scale anchors from input coords to flow coords
            scaled_anchors = []
            for (t, x, y) in anchors:
                scaled_x = x / self.flow_scale
                scaled_y = y / self.flow_scale
                scaled_anchors.append((t, scaled_x, scaled_y))
            track_flow = self._builder.build_track(scaled_anchors)
            
            # Scale track back to input coordinates
            track_int = np.zeros((self.T, 2), dtype=np.int32)
            track_int[:, 0] = np.clip(np.round(track_flow[:, 0] * self.flow_scale), 0, input_W - 1).astype(np.int32)
            track_int[:, 1] = np.clip(np.round(track_flow[:, 1] * self.flow_scale), 0, input_H - 1).astype(np.int32)
            
            # Restore original anchor positions exactly
            for (t, x, y) in anchors:
                x_i = int(np.clip(round(x), 0, input_W - 1))
                y_i = int(np.clip(round(y), 0, input_H - 1))
                track_int[t] = (x_i, y_i)
            
            # Override linear interpolation segments
            if self.linear_interp_threshold > 0:
                for i in range(len(anchors) - 1):
                    t0, x0, y0 = anchors[i]
                    t1, x1, y1 = anchors[i + 1]
                    if t1 - t0 <= self.linear_interp_threshold:
                        segment_len = t1 - t0 + 1
                        alpha = np.linspace(0, 1, segment_len)
                        track_int[t0:t1+1, 0] = np.clip(np.round((1 - alpha) * x0 + alpha * x1), 0, input_W - 1).astype(np.int32)
                        track_int[t0:t1+1, 1] = np.clip(np.round((1 - alpha) * y0 + alpha * y1), 0, input_H - 1).astype(np.int32)
            
            return track_int
        else:
            # No scaling needed
            track_int = self._builder.build_track(anchors)
            
            # Restore anchor positions to ensure exact placement
            for (t, x, y) in anchors:
                x_i = int(np.clip(round(x), 0, input_W - 1))
                y_i = int(np.clip(round(y), 0, input_H - 1))
                track_int[t] = (x_i, y_i)
            
            # Override linear interpolation segments
            if self.linear_interp_threshold > 0:
                for i in range(len(anchors) - 1):
                    t0, x0, y0 = anchors[i]
                    t1, x1, y1 = anchors[i + 1]
                    if t1 - t0 <= self.linear_interp_threshold:
                        segment_len = t1 - t0 + 1
                        alpha = np.linspace(0, 1, segment_len)
                        track_int[t0:t1+1, 0] = np.clip(np.round((1 - alpha) * x0 + alpha * x1), 0, input_W - 1).astype(np.int32)
                        track_int[t0:t1+1, 1] = np.clip(np.round((1 - alpha) * y0 + alpha * y1), 0, input_H - 1).astype(np.int32)
            
            return track_int
    
    def get_cache_stats(self):
        """Return cache statistics from the underlying builder."""
        stats = self._builder.get_cache_stats()
        stats["method"] = "corridor_dp"
        return stats


# =============================================================================
# OPTIMIZED INTERPOLATION (LEGACY - for backward compatibility)
# =============================================================================

def interpolate(
    flows,
    frame1,
    click1,
    frame2,
    click2,
    radius=20,
    device=None,
    dtype=None,
    corridor_width=None,
    cancel_check=None,
):
    """
    OPTIMIZED Fast corridor DP interpolation between (frame1, click1) and (frame2, click2).
    Inputs:
      flows: (T-1, H, W, 2) (dx,dy) t->t+1, can be numpy array or pre-converted torch.Tensor
      corridor_width: If None, uses adaptive corridor. If 0 or negative, uses full frame.
      radius: Search radius (default 20 to match original DeepMind implementation)
    Returns:
      path: (frame2-frame1+1, 2) int (x,y)
      min_cost: float
    
    PERFORMANCE NOTE:
      If `flows` is already a torch.Tensor on the target device, this function
      uses it directly without copying. Pre-converting the flow array to a GPU
      tensor once and reusing it across multiple interpolate calls can provide
      a 4x+ speedup (conversion takes ~250-400ms per call otherwise).
    """
    # OPTIMIZATION: Check if flows is already a torch.Tensor to avoid repeated conversion
    # This is the main performance bottleneck - converting the entire flow array on each call
    if isinstance(flows, torch.Tensor):
        # Already a tensor - use directly (huge performance win when called multiple times)
        flows_t = flows
        if device is None:
            device = flows_t.device
        if dtype is None:
            dtype = flows_t.dtype
    else:
        # Convert numpy array to tensor (slow - try to avoid this in loops)
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if dtype is None:
            try:
                _dt = torch.as_tensor(flows).dtype if hasattr(flows, "dtype") else torch.float32
                dtype = _dt if _dt.is_floating_point else torch.float32
            except Exception:
                dtype = torch.float32
        flows_t = torch.as_tensor(flows, device=device, dtype=dtype)  # (T-1,H,W,2)
    Tm1, H, W, _ = flows_t.shape
    assert 0 <= frame1 < frame2 <= Tm1, f"frames out of range: {frame1}..{frame2} for T={Tm1+1}"

    # OPTIMIZATION: Inline clamp function
    def qclamp(v, lo, hi): return max(lo, min(hi, int(round(float(v)))))
    x1 = qclamp(click1[0], 0, W - 1)
    y1 = qclamp(click1[1], 0, H - 1)
    x2 = qclamp(click2[0], 0, W - 1)
    y2 = qclamp(click2[1], 0, H - 1)

    # OPTIMIZATION: Use cached offset grids
    offsets, neg_offsets, window, K = _get_offset_grid(radius, device, dtype)
    BIG = torch.finfo(dtype).max / 8

    # Determine if using full frame or corridor mode
    use_full_frame = (corridor_width is not None and corridor_width <= 0)
    
    if not use_full_frame:
        # Adaptive corridor width (RAFT_v4 style)
        band = max(24, int(round(1.5 * radius)))

        # Predict centerline by forward integrating flow and blending to straight line
        centers = []
        pos = torch.tensor([float(x1), float(y1)], device=device, dtype=torch.float32)
        for t in range(frame1, frame2 + 1):
            if cancel_check is not None and (t % 5 == 0):
                cancel_check()
            cx = int(round(float(pos[0]))); cy = int(round(float(pos[1])))
            cx = max(0, min(W - 1, cx)); cy = max(0, min(H - 1, cy))
            centers.append((cx, cy))
            if t < frame2:
                dx, dy = flows_t[t, cy, cx]
                pos[0] += float(dx); pos[1] += float(dy)
        
        if frame2 > frame1:
            # OPTIMIZATION: Vectorized blending
            alpha = np.linspace(0, 1, frame2 - frame1 + 1)
            lx = (1 - alpha) * x1 + alpha * x2
            ly = (1 - alpha) * y1 + alpha * y2
            for i in range(len(centers)):
                cx, cy = centers[i]
                centers[i] = (int(round(0.7 * cx + 0.3 * lx[i])), 
                             int(round(0.7 * cy + 0.3 * ly[i])))

    tiles, pred_k_tiles = [], []
    cost, prev_tile = None, None

    with torch.inference_mode():
        for t in range(frame1, frame2):
            if cancel_check is not None and (t % 2 == 0):
                cancel_check()
            if use_full_frame:
                # Full-frame mode: use entire frame
                y0, y1e, x0, x1e = 0, H, 0, W
            else:
                # Corridor mode: use band around predicted centers
                cx0, cy0 = centers[t - frame1]
                cx1, cy1 = centers[t - frame1 + 1]

                x0 = max(0, min(cx0, cx1) - band)
                x1e = min(W, max(cx0, cx1) + band + 1)
                y0 = max(0, min(cy0, cy1) - band)
                y1e = min(H, max(cy0, cy1) + band + 1)
                
            h, w = (y1e - y0), (x1e - x0)
            if h <= 0 or w <= 0:
                y0 = max(0, min(cy0, cy1) if not use_full_frame else 0)
                x0 = max(0, min(cx0, cx1) if not use_full_frame else 0)
                y1e = min(H, y0 + 1); x1e = min(W, x0 + 1)
                h, w = 1, 1
            cur_tile = (y0, y1e, x0, x1e)

            if cost is None:
                cost = torch.full((h, w), BIG, device=device, dtype=dtype)
                sy = int(np.clip(y1, y0, y1e - 1)); sx = int(np.clip(x1, x0, x1e - 1))
                cost[sy - y0, sx - x0] = 0
            else:
                cost_new = torch.full((h, w), BIG, device=device, dtype=dtype)
                if use_full_frame:
                    # Full frame: entire cost matrix carries forward
                    cost_new = cost
                else:
                    # Corridor: transfer overlapping region
                    py0, py1e, px0, px1e = prev_tile
                    ys_ = max(y0, py0); ye_ = min(y1e, py1e)
                    xs_ = max(x0, px0); xe_ = min(x1e, px1e)
                    if ys_ < ye_ and xs_ < xe_:
                        cost_new[ys_ - y0:ye_ - y0, xs_ - x0:xe_ - x0] = \
                            cost[ys_ - py0:ye_ - py0, xs_ - px0:xe_ - px0]
                cost = cost_new

            cpad = F.pad(cost[None, None, ...], (radius, radius, radius, radius), value=BIG)
            cpatch = F.unfold(cpad, kernel_size=window).squeeze(0).transpose(0, 1)  # (h*w, K)

            flow_hw2 = flows_t[t, y0:y1e, x0:x1e, :].permute(2, 0, 1).unsqueeze(0)  # (1,2,h,w)
            fpad = F.pad(flow_hw2, (radius, radius, radius, radius), value=0)
            funfold = F.unfold(fpad, kernel_size=window).squeeze(0).view(2, K, h * w).permute(2, 1, 0)

            l1 = (neg_offsets.unsqueeze(0) - funfold).abs_().sum(dim=2)  # (h*w, K)
            total = cpatch + l1
            new_cost, argk = total.min(dim=1)                            # (h*w), (h*w)
            cost = new_cost.view(h, w)

            pred_k_tiles.append(argk.view(h, w).to(torch.uint16).cpu())
            tiles.append(cur_tile)
            prev_tile = cur_tile

        fy0, fy1e, fx0, fx1e = tiles[-1]
        yy = int(np.clip(y2, fy0, fy1e - 1)) - fy0
        xx = int(np.clip(x2, fx0, fx1e - 1)) - fx0
        min_cost = float(cost[yy, xx].item())

    # Backtrack
    x, y = x2, y2
    path = [(x, y)]
    for step in range(len(pred_k_tiles) - 1, -1, -1):
        y0, y1e, x0, x1e = tiles[step]
        h, w = (y1e - y0), (x1e - x0)
        tx = int(np.clip(x - x0, 0, w - 1))
        ty = int(np.clip(y - y0, 0, h - 1))
        kidx = int(pred_k_tiles[step][ty, tx].item())
        r = kidx // window; c = kidx % window
        dy = r - radius; dx = c - radius
        y = int(np.clip(y + dy, 0, H - 1))
        x = int(np.clip(x + dx, 0, W - 1))
        path.append((x, y))
    path.reverse()
    return np.asarray(path, dtype=np.int32), min_cost


# =============================================================================
# INCREMENTAL TRACK BUILDER WITH CACHING
# =============================================================================

class IncrementalTrackBuilder:
    """
    OPTIMIZATION (RAFT_v4): Cache segment interpolations to avoid redundant computation.
    Additional features: corridor_width parameter, backward propagation support.
    
    PERFORMANCE OPTIMIZATION:
        Pre-converts the optical flow array to a GPU tensor once during initialization.
        This provides 4x+ speedup for corridor DP by avoiding repeated numpy->GPU transfers.
    """

    def __init__(
        self,
        T,
        H,
        W,
        optical_flows_np,
        interpolate_fn,
        *,
        corridor_width=None,
        segment_cache=None,
        max_cache_size=10000,
        cache_rounding=0.25,
        cancel_check=None,
        preconvert_to_gpu=True,  # OPTIMIZATION: Pre-convert flows to GPU tensor
    ):
        self.T = T
        self.H = H
        self.W = W
        self.interpolate_fn = interpolate_fn
        self.corridor_width = corridor_width
        self.cancel_check = cancel_check
        
        # Determine device for tensor operations
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._use_gpu = torch.cuda.is_available()
        
        # Handle MemoryEfficientFlowArray by extracting float32 data
        if isinstance(optical_flows_np, MemoryEfficientFlowArray):
            flows_numpy = optical_flows_np.as_float32()
        elif isinstance(optical_flows_np, torch.Tensor):
            flows_numpy = None  # Already a tensor
        else:
            flows_numpy = optical_flows_np
        
        # OPTIMIZATION: Pre-convert optical flows to GPU tensor once (GPU mode only)
        # This avoids the expensive CPU->GPU transfer on each interpolate call
        # which is the main performance bottleneck (~250-400ms per call, 81% of total time)
        #
        # NOTE: On CPU, we skip pre-conversion because:
        # 1. torch.as_tensor() on CPU is essentially zero-copy with numpy
        # 2. Keeping a CPU tensor provides no benefit over numpy
        # 3. The interpolate function handles CPU numpy efficiently
        if preconvert_to_gpu and flows_numpy is not None and self._use_gpu:
            # GPU mode: Convert to GPU tensor once (takes ~250ms but saves ~250ms per segment)
            self.optical_flows_tensor = torch.as_tensor(
                flows_numpy, device=self._device, dtype=torch.float32
            )
            self.optical_flows_np = flows_numpy  # Keep numpy for backward propagation
        elif isinstance(optical_flows_np, torch.Tensor):
            self.optical_flows_tensor = optical_flows_np
            self.optical_flows_np = optical_flows_np  # Already tensor
        else:
            # CPU mode or preconvert disabled: Use numpy directly (no overhead)
            self.optical_flows_tensor = None
            self.optical_flows_np = flows_numpy if flows_numpy is not None else optical_flows_np

        if segment_cache is None:
            self.segment_cache = SegmentCache(max_size=max_cache_size, rounding=cache_rounding)
        else:
            self.segment_cache = segment_cache

        self.total_cache_lookups = 0
        self.total_cache_hits = 0

    def _get_or_compute_segment(self, t0, x0, y0, t1, x1, y1):
        if self.cancel_check is not None:
            self.cancel_check()
        key = self.segment_cache.make_key(t0, x0, y0, t1, x1, y1, corridor_width=self.corridor_width)
        self.total_cache_lookups += 1

        seg_xy = self.segment_cache.get(key)
        if seg_xy is not None:
            self.total_cache_hits += 1
            return seg_xy, True

        # OPTIMIZATION: Use pre-converted GPU tensor if available
        flows_to_use = self.optical_flows_tensor if self.optical_flows_tensor is not None else self.optical_flows_np
        
        seg_xy, _ = self.interpolate_fn(
            flows_to_use,
            t0,
            (x0, y0),
            t1,
            (x1, y1),
            corridor_width=self.corridor_width,
            cancel_check=self.cancel_check,
        )
        seg_xy = np.asarray(seg_xy, dtype=np.float32)
        self.segment_cache.put(key, seg_xy)
        return seg_xy, False

    def build_track(self, anchors):
        """
        OPTIMIZATION (RAFT_v4): Build track from anchors using cached segments where possible.
        Supports bidirectional propagation - first anchor can be at any frame.
        """
        if not anchors:
            raise ValueError("Provide at least one anchor, e.g. (t, x, y).")

        anchors = sorted(set((int(t), float(x), float(y)) for (t, x, y) in anchors), key=lambda a: a[0])
        
        # Validate anchor frames
        for t, x, y in anchors:
            if not (0 <= t < self.T):
                raise ValueError(f"Anchor frame {t} out of range [0, {self.T - 1}]")

        all_pos = np.zeros((self.T, 2), dtype=np.int32)

        # Place anchors
        for t, x, y in anchors:
            x_i, y_i = _clamp_xy(x, y, self.W, self.H)
            all_pos[t] = (x_i, y_i)

        # Interpolate between anchors using cache
        for (t0, x0, y0), (t1, x1, y1) in zip(anchors[:-1], anchors[1:]):
            if self.cancel_check is not None:
                self.cancel_check()
            if t1 <= t0:
                continue

            # Use cached segment if available!
            seg_xy, was_cached = self._get_or_compute_segment(t0, x0, y0, t1, x1, y1)

            # OPTIMIZATION (RAFT_v4): Vectorized clamping
            seg_clipped = seg_xy.copy()
            seg_clipped[:, 0] = np.clip(np.round(seg_xy[:, 0]), 0, self.W - 1).astype(np.int32)
            seg_clipped[:, 1] = np.clip(np.round(seg_xy[:, 1]), 0, self.H - 1).astype(np.int32)
            all_pos[t0:t1+1] = seg_clipped

        # Backward propagation: before first anchor (if not at frame 0)
        t_first, x_first, y_first = anchors[0]
        if t_first > 0:
            pos = np.array([x_first, y_first], dtype=np.float32)
            for t in range(t_first - 1, -1, -1):
                if self.cancel_check is not None and (t % 10 == 0):
                    self.cancel_check()
                # Backward propagation using reverse flow approximation
                # flow[t] gives displacement t -> t+1, negate to go t+1 -> t
                x_i, y_i = _clamp_xy(pos[0], pos[1], self.W, self.H)
                flow_val = self._get_flow_value(t, y_i, x_i)
                pos[0] -= flow_val[0]
                pos[1] -= flow_val[1]
                x_i, y_i = _clamp_xy(pos[0], pos[1], self.W, self.H)
                all_pos[t] = (x_i, y_i)

        # Forward propagate after last anchor
        t_last, x_last, y_last = anchors[-1]
        pos = np.array([x_last, y_last], dtype=np.float32)
        for t in range(t_last, self.T):
            if self.cancel_check is not None and (t % 10 == 0):
                self.cancel_check()
            x_i, y_i = _clamp_xy(pos[0], pos[1], self.W, self.H)
            all_pos[t] = (x_i, y_i)
            if t < self.T - 1:
                flow_val = self._get_flow_value(t, y_i, x_i)
                pos[0] += flow_val[0]
                pos[1] += flow_val[1]

        return all_pos  # (T,2) ints (x,y)
    
    def _get_flow_value(self, t, y, x):
        """Get flow value at (t, y, x) from either numpy array or tensor."""
        if isinstance(self.optical_flows_np, torch.Tensor):
            flow = self.optical_flows_np[t, y, x]
            return (float(flow[0].item()), float(flow[1].item()))
        else:
            flow = self.optical_flows_np[t, y, x]
            return (float(flow[0]), float(flow[1]))

    def get_cache_stats(self):
        stats = self.segment_cache.stats()
        accesses = stats["cache_hits"] + stats["cache_misses"]
        stats["total_cache_accesses"] = accesses
        stats["builder_cache_hits"] = self.total_cache_hits
        stats["builder_cache_lookups"] = self.total_cache_lookups
        stats["builder_hit_rate"] = (
            self.total_cache_hits / self.total_cache_lookups if self.total_cache_lookups else 0.0
        )
        return stats


# =============================================================================
# PERSISTENT RAFT MODEL MANAGER
# =============================================================================

class RAFTModelManager:
    """Manages persistent RAFT model instance with optimized inference."""
    
    def __init__(self, device=None, model_size="large"):
        self.device = device if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model_size = model_size
        self.model = None
        # torch.compile can improve performance but has been observed to hit
        # rare Inductor/CUDA-graphs assertions on some systems. Keep it configurable.
        self._torch_compile_enabled = str(os.getenv("RIPPLE_TORCH_COMPILE", "1")).strip().lower() not in (
            "0",
            "false",
            "no",
            "off",
        )
        self._torch_compile_mode = str(os.getenv("RIPPLE_TORCH_COMPILE_MODE", "reduce-overhead")).strip() or "reduce-overhead"
        # Default: disable cudagraph trees because they can trigger TLS assertions (PyTorch 2.9.x).
        self._disable_cudagraph_trees = str(os.getenv("RIPPLE_TORCH_DISABLE_CUDAGRAPH_TREES", "1")).strip().lower() not in (
            "0",
            "false",
            "no",
            "off",
        )
        self._model_compiled = False
        # Pre-allocated tensors for common video sizes (reduces memory fragmentation)
        self._tensor_cache = {}
        self._load_model()
    
    def _load_model(self):
        """Load RAFT model once and keep in memory."""
        print(f"Loading RAFT {self.model_size} model on {self.device}...")
        try:
            from torchvision.models.optical_flow import (
                raft_large, raft_small, Raft_Large_Weights, Raft_Small_Weights
            )
            if self.model_size.lower().startswith("l"):
                weights = Raft_Large_Weights.DEFAULT
                self.model = raft_large(weights=weights).to(self.device).eval()
            else:
                weights = Raft_Small_Weights.DEFAULT
                self.model = raft_small(weights=weights).to(self.device).eval()
            
            # Verify model is actually on the expected device
            model_device = next(self.model.parameters()).device
            print(f"  Model parameters are on: {model_device}")
            if model_device.type != self.device.type:
                print(f"  WARNING: Model is on {model_device} but expected {self.device}!")
            
            if self.device.type == "cuda":
                mem_after = torch.cuda.memory_allocated(0) / 1024**2
                print(f"  GPU memory after model load: {mem_after:.1f} MB")
            
            # OPTIMIZATION: Use torch.compile for PyTorch 2.0+ (significant speedup)
            # Stability note:
            #   On PyTorch 2.9.x, Inductor's cudagraph_trees can throw internal AssertionError
            #   (torch._C._is_key_in_tls) in some environments. We disable it by default.
            self._model_compiled = False
            if hasattr(torch, "compile") and self.device.type == "cuda" and self._torch_compile_enabled:
                try:
                    if self._disable_cudagraph_trees:
                        try:
                            import torch._inductor.config as _ind_cfg

                            if hasattr(_ind_cfg, "triton") and hasattr(_ind_cfg.triton, "cudagraph_trees"):
                                _ind_cfg.triton.cudagraph_trees = False
                            if hasattr(_ind_cfg, "triton") and hasattr(_ind_cfg.triton, "cudagraphs"):
                                _ind_cfg.triton.cudagraphs = False
                            try:
                                trees_val = getattr(_ind_cfg.triton, "cudagraph_trees", None)
                                graphs_val = getattr(_ind_cfg.triton, "cudagraphs", None)
                                print(f"  Inductor config: triton.cudagraph_trees={trees_val}, triton.cudagraphs={graphs_val}")
                            except Exception:
                                print("  Inductor: disabled cudagraph_trees for stability")
                        except Exception as e:
                            print(f"  Note: could not update Inductor cudagraph config ({e})")

                    self.model = torch.compile(self.model, mode=self._torch_compile_mode)
                    self._model_compiled = True
                    print("✓ RAFT model compiled with torch.compile for faster inference")
                except Exception as e:
                    self._model_compiled = False
                    print(f"Note: torch.compile failed ({e}), using eager mode")
            
            print(f"✓ RAFT model loaded successfully")
        except Exception as e:
            print(f"ERROR: Failed to load RAFT model: {e}", file=sys.stderr)
            sys.exit(1)
    
    def compute_optical_flow(self, v01: np.ndarray, batch_pairs: int = 2, is_avi: bool = False, cancel_check=None) -> np.ndarray:
        """Compute optical flow using the persistent model.
        
        OPTIMIZED: Increased default batch size, added AMP support, uses non_blocking transfers.
        
        Args:
            v01: Video frames. For TIFF: (T,H,W) float32 [0,1]. For AVI: (T,H,W,3) uint8 [0,255] RGB
            batch_pairs: Number of frame pairs to process together (default increased to 2)
            is_avi: If True, use AVI preprocessing matching DeepMind (RGB uint8 -> [-1,1])
            
        Raises:
            torch.cuda.OutOfMemoryError: If GPU runs out of memory during inference
        """
        if is_avi:
            T, H, W, C = v01.shape
            assert C == 3, f"Expected RGB video (T,H,W,3), got {v01.shape}"
        else:
            T, H, W = v01.shape
        
        flows = np.zeros((T - 1, H, W, 2), dtype=np.float32)
        pair_indices = [(t, t + 1) for t in range(T - 1)]
        
        # Choose the correct tensor conversion function
        tensor_converter = _to_3ch_tensor_avi if is_avi else _to_3ch_tensor01
        
        # OPTIMIZATION: Use automatic mixed precision on CUDA for ~2x speedup
        use_amp = self.device.type == "cuda"
        
        # Track local tensors for cleanup on error
        local_tensors = []
        
        try:
            for start in range(0, len(pair_indices), batch_pairs):
                if cancel_check is not None:
                    cancel_check()
                chunk = pair_indices[start:start + batch_pairs]
                batch_0, batch_1 = [], []
                
                for (t0, t1) in chunk:
                    img0 = tensor_converter(v01[t0])
                    img1 = tensor_converter(v01[t1])
                    batch_0.append(img0)
                    batch_1.append(img1)
                
                # OPTIMIZATION: Use non_blocking=True for async CPU->GPU transfer
                I0 = torch.stack(batch_0, dim=0).to(self.device, non_blocking=True)
                I1 = torch.stack(batch_1, dim=0).to(self.device, non_blocking=True)
                local_tensors.extend([I0, I1])
                
                I0p, ph, pw = pad_to_multiple(I0, mult=8)
                I1p, _, _ = pad_to_multiple(I1)
                local_tensors.extend([I0p, I1p])
                
                # OPTIMIZATION: Use AMP autocast for faster inference
                with torch.no_grad():
                    if use_amp:
                        with torch.amp.autocast('cuda'):
                            pred = self.model(I0p, I1p)
                            flow = pred[-1] if isinstance(pred, (list, tuple)) else pred
                    else:
                        pred = self.model(I0p, I1p)
                        flow = pred[-1] if isinstance(pred, (list, tuple)) else pred
                
                flow = flow[..., :H, :W]
                # OPTIMIZATION: Avoid contiguous() call if already contiguous
                flow_perm = flow.permute(0, 2, 3, 1)
                flow_np = (flow_perm if flow_perm.is_contiguous() else flow_perm.contiguous()).cpu().numpy()
                
                for k, (t0, _) in enumerate(chunk):
                    flows[t0] = flow_np[k]
                
                # Clean up this batch's tensors immediately
                del I0, I1, I0p, I1p, flow, flow_np
                local_tensors.clear()
                
                # Progress indicator for long computations
                if (start + batch_pairs) % 10 == 0:
                    print(f"  Processed {start + batch_pairs}/{len(pair_indices)} frame pairs...")
            
            # OPTIMIZATION: Only clear cache periodically, not after every batch
            if self.device.type == "cuda":
                torch.cuda.empty_cache()
            
            return flows
            
        except torch.cuda.OutOfMemoryError:
            # Clean up any allocated tensors before re-raising
            for tensor in local_tensors:
                try:
                    del tensor
                except:
                    pass
            local_tensors.clear()
            cleanup_gpu_memory()
            raise
        except AssertionError as e:
            # Inductor/CUDA-graphs can throw internal TLS assertions in some setups.
            # Provide a clear, actionable message.
            for tensor in local_tensors:
                try:
                    del tensor
                except Exception:
                    pass
            local_tensors.clear()
            cleanup_gpu_memory()

            msg = str(e).strip() or repr(e)
            raise RuntimeError(
                "RAFT_INDUCTOR_ASSERTION: A low-level PyTorch Inductor assertion occurred during RAFT inference. "
                "This is usually caused by torch.compile/Inductor CUDA-graphs on some systems. "
                "Workarounds: set RIPPLE_TORCH_COMPILE=0 (disable torch.compile) or set RIPPLE_TORCH_DISABLE_CUDAGRAPH_TREES=1. "
                f"Original error: {msg}"
            ) from e
        except ValueError as e:
            # Catch RAFT's minimum resolution error and provide a clearer message
            error_msg = str(e)
            if "Feature maps are too small" in error_msg or "at least 16" in error_msg:
                # Clean up tensors
                for tensor in local_tensors:
                    try:
                        del tensor
                    except:
                        pass
                local_tensors.clear()
                cleanup_gpu_memory()
                raise RuntimeError(
                    f"RAFT_RESOLUTION_ERROR: Video resolution {W}x{H} is too small for RAFT. "
                    f"Minimum required resolution is {RAFT_MINIMUM_RESOLUTION}x{RAFT_MINIMUM_RESOLUTION}. "
                    f"This should have been auto-upscaled - please report this as a bug."
                ) from e
            raise
    
    def unload_model(self):
        """Unload the RAFT model from GPU to free VRAM.
        
        The model will be reloaded on next use via ensure_loaded().
        """
        if self.model is None:
            print("  RAFT model already unloaded")
            return False
            
        print("🧹 Unloading RAFT model from GPU...")
        
        # Report initial state
        if torch.cuda.is_available():
            mem_before = torch.cuda.memory_allocated(0) / 1024**2
            print(f"   Before: {mem_before:.1f} MB allocated")
        
        # Clear tensor cache
        self._tensor_cache.clear()
        
        # Reset torch.compile/dynamo state (releases compiled CUDA graphs)
        try:
            torch._dynamo.reset()
        except (ImportError, AttributeError):
            pass
        
        # Move model to CPU before deleting (helps release GPU memory)
        try:
            self.model = self.model.cpu()
        except:
            pass
        
        # Delete the model
        del self.model
        self.model = None
        
        # Garbage collection
        import gc
        gc.collect()
        gc.collect()
        
        # Clear CUDA cache
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            
            mem_after = torch.cuda.memory_allocated(0) / 1024**2
            print(f"   After: {mem_after:.1f} MB allocated")
            print(f"✓ RAFT model unloaded, freed {mem_before - mem_after:.1f} MB")
        else:
            print("✓ RAFT model unloaded")
        
        return True
    
    def ensure_loaded(self):
        """Ensure the model is loaded (reload if previously unloaded)."""
        if self.model is None:
            self._load_model()


# =============================================================================
# CPU-ONLY MODEL MANAGER (DIS Optical Flow)
# =============================================================================

class CPUModelManager:
    """Lightweight model manager for CPU-only operation using DIS optical flow.
    
    This manager provides a compatible interface to RAFTModelManager but uses
    OpenCV's DIS (Dense Inverse Search) optical flow instead of RAFT.
    DIS is much faster on CPU and provides reasonable quality for tracking.
    
    CPU Mode Limitations:
    - No RAFT optical flow (GPU required)
    - No LocoTrack point tracking (GPU required)
    - DIS flow is less accurate but much faster on CPU
    - TrackMate-style DoG detection still works
    - TrackPy particle tracking still works
    """
    
    def __init__(self, device=None):
        self.device = device if device else torch.device("cpu")
        self.model = None  # No GPU model loaded
        self.model_size = "cpu"
        self._tensor_cache = {}
        
        # DIS optical flow preset (PRESET_MEDIUM is a good balance)
        # PRESET_ULTRAFAST = 0, PRESET_FAST = 1, PRESET_MEDIUM = 2
        self.dis_preset = cv2.DISOPTICAL_FLOW_PRESET_MEDIUM
        
        print(f"CPUModelManager initialized")
        print(f"  Using OpenCV DIS optical flow (preset: MEDIUM)")
        print(f"  Device: {self.device}")
        print(f"  Note: RAFT/LocoTrack unavailable in CPU mode")
    
    def compute_optical_flow(self, v01: np.ndarray, batch_pairs: int = 1, is_avi: bool = False, cancel_check=None) -> np.ndarray:
        """Compute optical flow using DIS algorithm on CPU.
        
        This is a drop-in replacement for RAFTModelManager.compute_optical_flow()
        that uses OpenCV's DIS algorithm instead of RAFT.
        
        Args:
            v01: Video frames. For TIFF: (T,H,W) float32 [0,1]. For AVI: (T,H,W,3) uint8 [0,255] RGB
            batch_pairs: Ignored (DIS processes one pair at a time)
            is_avi: If True, input is RGB (T,H,W,3), otherwise grayscale (T,H,W)
            
        Returns:
            Optical flow array of shape (T-1, H, W, 2)
        """
        if is_avi:
            T, H, W, C = v01.shape
            assert C == 3, f"Expected RGB video (T,H,W,3), got {v01.shape}"
        else:
            T, H, W = v01.shape
        
        flows = np.zeros((T - 1, H, W, 2), dtype=np.float32)
        
        # Create DIS optical flow object
        dis = cv2.DISOpticalFlow_create(self.dis_preset)
        
        print(f"  Computing DIS optical flow: {T-1} frame pairs...")
        
        for t in range(T - 1):
            if cancel_check is not None:
                cancel_check()
            # Convert frames to grayscale uint8 for DIS
            if is_avi:
                # RGB to grayscale
                frame0 = cv2.cvtColor(v01[t], cv2.COLOR_RGB2GRAY)
                frame1 = cv2.cvtColor(v01[t + 1], cv2.COLOR_RGB2GRAY)
            else:
                # Float [0,1] to uint8 [0,255]
                frame0 = (v01[t] * 255).astype(np.uint8)
                frame1 = (v01[t + 1] * 255).astype(np.uint8)
            
            # Compute flow
            flow = dis.calc(frame0, frame1, None)
            flows[t] = flow
            
            # Progress every 50 frames
            if (t + 1) % 50 == 0 or t == T - 2:
                print(f"    Frame {t + 1}/{T - 1}")
        
        return flows
    
    def unload_model(self):
        """No-op for CPU mode (no GPU model to unload)."""
        print("  CPUModelManager: No GPU model to unload")
        return False
    
    def ensure_loaded(self):
        """No-op for CPU mode (no model to load)."""
        pass


# =============================================================================
# PERSISTENT TRACKING SERVER
# =============================================================================


class OperationCancelledError(Exception):
    """Raised when a long-running operation is cancelled by the user."""
    pass


class TrackingServer:
    """Socket server for handling tracking requests (Unix socket or TCP)."""
    
    def __init__(self, socket_path, model_manager, locotrack_manager=None, tcp_host=None, tcp_port=9876):
        self.socket_path = socket_path
        self.tcp_host = tcp_host
        self.tcp_port = tcp_port
        self.use_tcp = tcp_host is not None
        self.model_manager = model_manager  # RAFT model manager
        self.locotrack_manager = locotrack_manager  # LocoTrack model manager (loaded lazily)
        self.running = False
        self.server_socket = None
        
        # Cancellation support for long-running operations
        import threading
        self._cancelled = threading.Event()  # Thread-safe cancellation flag
        self._operation_in_progress = threading.Event()  # Track if an operation is running
        self._operation_lock = threading.Lock()  # Prevent concurrent operations
        
        # Flow cache: {video_path: {flows_np, timestamp, metadata}}
        self.flow_cache = {}
        self._current_video_path = None  # Track which video is currently loaded

        # Video metadata: {video_path: {'original_shape': (T,H,W), 'resized_shape': (T,H,W), 'is_avi': bool}}
        self.video_metadata = {}

        # Segment cache (shared across videos, includes corridor_width in key)
        self.segment_cache = SegmentCache(max_size=10000, rounding=0.25)
        
        # Video accessor cache for blob detection (avoids slow re-initialization)
        # Key: (video_path, target_width, target_height) -> MemoryMappedVideoAccessor
        self._video_accessor_cache = {}
        self._video_accessor_cache_path = None  # Track which video is cached
    
    def is_cancelled(self):
        """Check if the current operation has been cancelled."""
        return self._cancelled.is_set()
    
    def _check_cancelled(self, operation_name="operation"):
        """Check cancellation flag and raise exception if cancelled.
        
        Call this periodically in long-running operations to allow cancellation.
        """
        if self._cancelled.is_set():
            print(f"⚠️ {operation_name} cancelled by user")
            raise OperationCancelledError(f"{operation_name} cancelled")
    
    def _clear_flow_cache_for_new_video(self, new_video_path):
        """Clear flow cache when switching to a different video to free memory.
        
        This is critical for memory management - large flow arrays (4-9 GB) 
        should not persist when the user switches to a different video.
        """
        import gc
        
        # Extract just the video filename (without directory and extension)
        new_base = os.path.basename(new_video_path)
        new_base = os.path.splitext(new_base)[0]
        # Remove common suffixes that indicate same video with different processing
        for suffix in ['_locotrack', '_trackpy', '_dis', '_raft', '_optical_flow']:
            if suffix in new_base:
                new_base = new_base.split(suffix)[0]
        
        current_base = None
        if self._current_video_path:
            current_base = os.path.basename(self._current_video_path)
            current_base = os.path.splitext(current_base)[0]
            for suffix in ['_locotrack', '_trackpy', '_dis', '_raft', '_optical_flow']:
                if suffix in current_base:
                    current_base = current_base.split(suffix)[0]
        
        # If switching to a different video, clear the cache
        if current_base and new_base != current_base:
            old_cache_size = sum(
                entry.get('flows_np').nbytes if hasattr(entry.get('flows_np'), 'nbytes') else 0
                for entry in self.flow_cache.values()
            ) / (1024 * 1024)
            
            print(f"🧹 Switching video: {current_base} → {new_base}")
            if old_cache_size > 10:
                print(f"   Clearing {old_cache_size:.0f} MB flow cache")
            
            # Clear all server caches
            self.flow_cache.clear()
            self.video_metadata.clear()
            self.segment_cache.clear()
            
            # Clear video accessor cache
            self._clear_video_accessor_cache()
            
            # Clear global offset cache
            global _OFFSET_CACHE
            _OFFSET_CACHE.clear()
            
            # Clear model manager's tensor cache if it exists
            if hasattr(self.model_manager, '_tensor_cache'):
                self.model_manager._tensor_cache.clear()
            
            # Force multiple garbage collection passes
            gc.collect()
            gc.collect()
            gc.collect()
            
            # Clear torch.compile dynamo cache if available (PyTorch 2.0+)
            try:
                torch._dynamo.reset()
            except (ImportError, AttributeError):
                pass
            
            # Also clear CUDA cache if available (frees GPU memory)
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
                print(f"   CUDA cache cleared")
        
        self._current_video_path = new_video_path
    
    def _clear_video_accessor_cache(self):
        """Clear the video accessor cache and close any open file handles."""
        for cache_key, accessor in self._video_accessor_cache.items():
            try:
                accessor.close()
            except Exception as e:
                print(f"  Warning: Failed to close accessor for {cache_key}: {e}")
        self._video_accessor_cache.clear()
        self._video_accessor_cache_path = None
    
    def _get_or_create_video_accessor(self, video_path, target_width, target_height, use_cache=True):
        """Get a cached video accessor or create a new one.
        
        This avoids the expensive re-initialization of video accessors, especially for:
        - AVI files (full load into memory each time)
        - TIFFs with mixed memmappable pages (slow memmap creation)
        
        Args:
            video_path: Path to video file
            target_width: Target width for resizing (or None)
            target_height: Target height for resizing (or None)
            use_cache: If True, cache the accessor for reuse (default True).
                       If False, create a new accessor each time (caller must close it).
            
        Returns:
            MemoryMappedVideoAccessor instance (cached or new)
        """
        # Normalize path for cache key
        video_path = os.path.abspath(video_path)
        
        # If caching is disabled, create a new accessor without caching
        if not use_cache:
            print(f"  Creating new video accessor (caching disabled)")
            return MemoryMappedVideoAccessor(video_path, target_width, target_height)
        
        cache_key = (video_path, target_width, target_height)
        
        # Clear cache if switching to a different video
        if self._video_accessor_cache_path and self._video_accessor_cache_path != video_path:
            print(f"  Clearing video accessor cache for new video")
            self._clear_video_accessor_cache()
        
        # Return cached accessor if available
        if cache_key in self._video_accessor_cache:
            print(f"  Using cached video accessor")
            return self._video_accessor_cache[cache_key]
        
        # Create new accessor and cache it
        print(f"  Creating new video accessor (will be cached)")
        accessor = MemoryMappedVideoAccessor(video_path, target_width, target_height)
        self._video_accessor_cache[cache_key] = accessor
        self._video_accessor_cache_path = video_path
        
        return accessor

    def _build_flow_filename(self, base_output_path, method, resized_shape=None, original_shape=None, **params):
        """Build a descriptive filename for optical flow based on method and parameters.
        
        Args:
            base_output_path: Original output path (used to extract directory and base name)
            method: Flow method ('raft', 'locotrack', 'trackpy', 'dis')
            resized_shape: (T, H, W) tuple of the processed video shape
            original_shape: (T, H, W) tuple of the original video shape (before compression)
            **params: Method-specific parameters
            
        Returns:
            Full path with descriptive filename
        """
        if base_output_path is None:
            return None
            
        directory = os.path.dirname(base_output_path)
        basename = os.path.basename(base_output_path)
        
        # Extract video name from basename (remove method suffix and extension)
        # Expected format: videoname_method_optical_flow.npz
        video_name = basename
        for suffix in ['_raft_optical_flow.npz', '_locotrack_optical_flow.npz', 
                       '_trackpy_optical_flow.npz', '_dis_optical_flow.npz',
                       '_optical_flow.npz', '.npz']:
            if video_name.endswith(suffix):
                video_name = video_name[:-len(suffix)]
                break
        
        # ALWAYS include working resolution in filename for reliable matching
        # This ensures Java can filter flow files by resolution from filename alone
        size_suffix = ""
        if resized_shape is not None and len(resized_shape) >= 3:
            T, H, W = resized_shape[0], resized_shape[1], resized_shape[2]
            # Always include resolution for unambiguous file matching
            size_suffix = f"_{W}x{H}"
        
        # Build method-specific parameter suffix
        if method == 'raft':
            # RAFT is simple - just method and optional size
            param_suffix = ""
        elif method == 'locotrack':
            # LocoTrack: include DoG configuration
            radius = params.get('radius', 2.5)
            threshold = params.get('threshold', 0.0)
            kernel = params.get('kernel', 'gaussian_rbf')
            flow_smooth = params.get('flow_smoothing', 15.0)
            temporal = params.get('temporal_smooth_factor', 0.1)
            seed_frames = params.get('seed_frames', [0])
            median = params.get('median_filter', False)
            subpixel = params.get('subpixel', True)
            invert = params.get('invert', False)
            
            # Normalize seed_frames to list of ints
            if seed_frames is None:
                seed_frames = [0]
            elif isinstance(seed_frames, (int, float)):
                seed_frames = [int(seed_frames)]
            else:
                seed_frames = [int(s) for s in seed_frames]
            
            # Compact parameter encoding with clearer seed frame label
            if seed_frames and len(seed_frames) > 0:
                if len(seed_frames) == 1:
                    seed_str = f"seed{seed_frames[0]}"
                else:
                    seed_str = "seeds" + "_".join(str(s) for s in sorted(seed_frames))
            else:
                seed_str = "seed0"
            
            param_suffix = (f"_r{radius:.1f}_t{threshold:.2f}_k{kernel[:3]}"
                           f"_fs{flow_smooth:.0f}_ts{temporal:.2f}_{seed_str}")
            if median:
                param_suffix += "_med"
            if not subpixel:
                param_suffix += "_nosp"
            if invert:
                param_suffix += "_inv"
        elif method == 'trackpy':
            # Trackpy: now uses DoG detection (same as LocoTrack)
            radius = params.get('radius', 2.5)
            threshold = params.get('threshold', 0.0)
            search_range = params.get('search_range', 15)
            memory = params.get('memory', 5)
            kernel = params.get('kernel', 'gaussian_rbf')
            flow_smooth = params.get('flow_smoothing', 15.0)
            smooth_factor = params.get('smooth_factor', 0.1)
            median_filter = params.get('median_filter', False)
            subpixel = params.get('subpixel', True)
            invert = params.get('invert', False)
            
            param_suffix = (f"_r{radius:.1f}_t{threshold:.2f}_sr{search_range}"
                           f"_m{memory}_k{kernel[:3]}_fs{flow_smooth:.0f}_sf{smooth_factor:.2f}")
            if median_filter:
                param_suffix += "_med"
            if not subpixel:
                param_suffix += "_nosp"
            if invert:
                param_suffix += "_inv"
        elif method == 'dis':
            # DIS: include downsample factor
            ds_factor = params.get('downsample_factor', 2)
            param_suffix = f"_ds{ds_factor}"
        else:
            param_suffix = ""
        
        # Construct final filename
        filename = f"{video_name}_{method}{size_suffix}{param_suffix}_optical_flow.npz"
        return os.path.join(directory, filename)
    
    def _find_existing_flow_file(self, base_output_path, method, target_width=None, target_height=None, **params):
        """Find an existing flow file matching the video, method, resolution, AND parameters.
        
        CRITICAL: This function must validate that cached files match not just resolution,
        but also the specific parameters used to compute the flow. Different parameters
        produce different flow fields!
        
        Args:
            base_output_path: Expected output path (used to extract directory and base name)
            method: Flow method ('raft', 'locotrack', 'trackpy', 'dis')
            target_width: Expected width (None = uncompressed/original resolution)
            target_height: Expected height (None = uncompressed/original resolution)
            **params: Method-specific parameters that must match the cached file.
                For DIS: downsample_factor
                For LocoTrack: radius, threshold, kernel, flow_smoothing, temporal_smooth_factor,
                              seed_frames, median_filter, subpixel, invert
                For Trackpy: radius, threshold, search_range, memory, kernel, flow_smoothing,
                            median_filter, subpixel, diameter (legacy), minmass (legacy), invert
                For RAFT: (no additional params, just resolution)
            
        Returns:
            Path to existing file if found with matching resolution AND parameters, None otherwise
        """
        if base_output_path is None:
            return None
            
        import glob
        import re
        
        directory = os.path.dirname(base_output_path)
        basename = os.path.basename(base_output_path)
        
        # Extract video name from basename
        video_name = basename
        for suffix in ['_raft_optical_flow.npz', '_locotrack_optical_flow.npz', 
                       '_trackpy_optical_flow.npz', '_dis_optical_flow.npz',
                       '_optical_flow.npz', '.npz']:
            if video_name.endswith(suffix):
                video_name = video_name[:-len(suffix)]
                break
        
        # Search for files matching pattern: videoname_method[_params]_optical_flow.npz
        pattern = os.path.join(directory, f"{video_name}_{method}*_optical_flow.npz")
        matching_files = glob.glob(pattern)
        
        if not matching_files:
            return None
        
        # Filter by resolution: files have resolution suffix like _WxH_ or no resolution suffix.
        # Resolution pattern: _(\d+)x(\d+) followed by more suffix or _optical_flow.npz.
        resolution_pattern = re.compile(r'_(\d+)x(\d+)(?:_|_optical_flow\.npz)')
        
        filtered_files = []
        for f in matching_files:
            fname = os.path.basename(f)
            match = resolution_pattern.search(fname)
            
            if target_width and target_height:
                # Looking for a specific compressed resolution
                if match:
                    file_w, file_h = int(match.group(1)), int(match.group(2))
                    if file_w == target_width and file_h == target_height:
                        filtered_files.append(f)
                # Skip files without resolution suffix - they're uncompressed
            else:
                # Backward compatibility:
                # Historically, some caches had no resolution suffix. New caches ALWAYS include it.
                # If no target resolution is provided, accept BOTH tagged and untagged files.
                filtered_files.append(f)

        if not (target_width and target_height):
            # Prefer legacy no-resolution-tag files when caller didn't specify resolution.
            no_tag = []
            tagged = []
            for f in filtered_files:
                fname = os.path.basename(f)
                if resolution_pattern.search(fname):
                    tagged.append(f)
                else:
                    no_tag.append(f)
            filtered_files = no_tag if no_tag else tagged
        
        if not filtered_files:
            print(f"  No {method} flow file found matching resolution "
                  f"{'%dx%d' % (target_width, target_height) if target_width else 'original'}")
            return None
        
        # CRITICAL: Filter by parameters if provided
        # Build expected parameter pattern based on method and params
        if params:
            param_filtered = self._filter_files_by_params(filtered_files, method, params)
            if param_filtered:
                filtered_files = param_filtered
            else:
                # No exact match found - log and return None to trigger recomputation
                print(f"  ⚠ Found {len(filtered_files)} {method} file(s) but none match current parameters")
                if params:
                    print(f"    Requested params: {params}")
                return None
        
        # Return the most recently modified file
        filtered_files.sort(key=lambda f: os.path.getmtime(f), reverse=True)
        return filtered_files[0]
    
    def _filter_files_by_params(self, files, method, params):
        """Filter flow files to those matching the given parameters.
        
        Args:
            files: List of file paths to filter
            method: Flow method name
            params: Dict of parameters to match
            
        Returns:
            List of files that match the parameters (may be empty)
        """
        import re
        
        if not params:
            return files
        
        matched = []
        
        for f in files:
            fname = os.path.basename(f)
            
            if method == 'raft':
                # RAFT has no additional parameters beyond resolution
                matched.append(f)
                
            elif method == 'dis':
                # DIS: check downsample_factor
                ds = params.get('downsample_factor', 2)
                # Pattern: _ds{N}_ in filename
                ds_match = re.search(r'_ds(\d+)', fname)
                if ds_match:
                    file_ds = int(ds_match.group(1))
                    if file_ds == ds:
                        matched.append(f)
                else:
                    # Legacy file without ds suffix - assume ds=2 (old default)
                    if ds == 2:
                        matched.append(f)
                        
            elif method == 'locotrack':
                # LocoTrack: check radius, threshold, kernel, flow_smoothing, etc.
                if self._locotrack_params_match_filename(fname, params):
                    matched.append(f)
                    
            elif method == 'trackpy':
                # Trackpy: check radius/diameter, threshold, search_range, etc.
                if self._trackpy_params_match_filename(fname, params):
                    matched.append(f)
        
        return matched
    
    def _locotrack_params_match_filename(self, fname, params):
        """Check if LocoTrack filename matches the given parameters.
        
        Filename format: video_locotrack_WxH_r{R}_t{T}_k{K}_fs{FS}_ts{TS}_seed{N}[_med][_nosp]_optical_flow.npz
        """
        import re
        
        # Extract parameters from filename
        radius_match = re.search(r'_r(\d+\.?\d*)_', fname)
        threshold_match = re.search(r'_t(\d+\.?\d*)_', fname)
        kernel_match = re.search(r'_k([a-z]+)_', fname)
        fs_match = re.search(r'_fs(\d+\.?\d*)_', fname)
        ts_match = re.search(r'_ts(\d+\.?\d*)_', fname)
        seed_match = re.search(r'_seed(\d+|s[\d_]+)', fname)
        has_med = '_med' in fname
        has_nosp = '_nosp' in fname
        has_inv = '_inv' in fname or '_invTrue' in fname
        
        # Compare with requested params (with tolerance for floats)
        radius = params.get('radius', 2.5)
        threshold = params.get('threshold', 0.0)
        kernel = params.get('kernel', 'gaussian_rbf')[:3]  # First 3 chars
        flow_smoothing = params.get('flow_smoothing', 15.0)
        temporal = params.get('temporal_smooth_factor', 0.1)
        median_filter = params.get('median_filter', False)
        subpixel = params.get('subpixel', True)
        invert = params.get('invert', False)
        seed_frames = params.get('seed_frames', [0])
        
        # Validate each parameter
        if radius_match:
            file_r = float(radius_match.group(1))
            if abs(file_r - radius) > 0.15:  # Tolerance for rounding
                return False
        
        if threshold_match:
            file_t = float(threshold_match.group(1))
            if abs(file_t - threshold) > 0.015:
                return False
        
        if kernel_match:
            file_k = kernel_match.group(1)
            if file_k != kernel[:len(file_k)]:
                return False
        
        if fs_match:
            file_fs = float(fs_match.group(1))
            if abs(file_fs - flow_smoothing) > 1.0:
                return False
        
        if ts_match:
            file_ts = float(ts_match.group(1))
            if abs(file_ts - temporal) > 0.015:
                return False
        
        # Check boolean flags
        if has_med != median_filter:
            return False
        if has_nosp != (not subpixel):
            return False
        if has_inv != invert:
            return False
        
        # Check seed frames (if present in filename)
        # Normalize seed_frames: None or empty list -> [0] (default)
        if seed_frames is None or (isinstance(seed_frames, list) and len(seed_frames) == 0):
            seed_frames = [0]
        elif isinstance(seed_frames, (int, float)):
            seed_frames = [int(seed_frames)]
        
        if seed_match:
            seed_str = seed_match.group(1)
            if seed_str.startswith('s'):
                # Multiple seeds: seeds0_5_10
                file_seeds = [int(s) for s in seed_str[1:].split('_') if s]
            else:
                # Single seed: seed0
                file_seeds = [int(seed_str)]
            
            if sorted(file_seeds) != sorted(seed_frames):
                return False
        
        return True
    
    def _trackpy_params_match_filename(self, fname, params):
        """Check if Trackpy filename matches the given parameters.
        
        Filename formats:
        - DoG mode: video_trackpy_WxH_r{R}_t{T}_sr{SR}_m{M}_k{K}_fs{FS}[_med][_nosp]_optical_flow.npz
        - Legacy mode: video_trackpy_WxH_d{D}_mm{MM}_sr{SR}_m{M}_k{K}_fs{FS}_optical_flow.npz
        """
        import re
        
        # Detect which mode the file was created in
        is_legacy_file = '_d' in fname and '_mm' in fname
        use_legacy = params.get('use_legacy_detection', False) or params.get('diameter') is not None
        
        if is_legacy_file:
            # Legacy trackpy detection mode
            diameter_match = re.search(r'_d(\d+)_', fname)
            minmass_match = re.search(r'_mm(\d+)_', fname)
            
            diameter = params.get('diameter')
            if diameter is None and params.get('radius'):
                diameter = int(params.get('radius') * 2)
            minmass = params.get('minmass', 0)
            
            if diameter_match and diameter is not None:
                if int(diameter_match.group(1)) != diameter:
                    return False
            if minmass_match:
                if int(minmass_match.group(1)) != minmass:
                    return False
        else:
            # DoG detection mode (same as LocoTrack)
            radius_match = re.search(r'_r(\d+\.?\d*)_', fname)
            threshold_match = re.search(r'_t(\d+\.?\d*)_', fname)
            
            radius = params.get('radius', 2.5)
            threshold = params.get('threshold', 0.0)
            
            if radius_match:
                file_r = float(radius_match.group(1))
                if abs(file_r - radius) > 0.15:
                    return False
            if threshold_match:
                file_t = float(threshold_match.group(1))
                if abs(file_t - threshold) > 0.015:
                    return False
        
        # Common parameters for both modes
        sr_match = re.search(r'_sr(\d+)_', fname)
        memory_match = re.search(r'_m(\d+)_', fname)
        kernel_match = re.search(r'_k([a-z]+)_', fname)
        fs_match = re.search(r'_fs(\d+\.?\d*)_', fname)
        sf_match = re.search(r'_sf(\d+\.?\d*)', fname)
        has_med = '_med' in fname
        has_nosp = '_nosp' in fname
        
        search_range = params.get('search_range', 15)
        memory = params.get('memory', 5)
        kernel = params.get('kernel', 'gaussian_rbf')[:3]
        flow_smoothing = params.get('flow_smoothing', 15.0)
        smooth_factor = params.get('smooth_factor', 0.1)
        median_filter = params.get('median_filter', False)
        subpixel = params.get('subpixel', True)
        invert = params.get('invert', False)
        has_inv = '_inv' in fname
        
        if sr_match and int(sr_match.group(1)) != search_range:
            return False
        if memory_match and int(memory_match.group(1)) != memory:
            return False
        if kernel_match and kernel_match.group(1) != kernel[:len(kernel_match.group(1))]:
            return False
        if fs_match:
            file_fs = float(fs_match.group(1))
            if abs(file_fs - flow_smoothing) > 1.0:
                return False
        if sf_match:
            file_sf = float(sf_match.group(1))
            if abs(file_sf - smooth_factor) > 0.015:
                return False
        if has_med != median_filter:
            return False
        if has_nosp != (not subpixel):
            return False
        if has_inv != invert:
            return False
        
        return True

    def _normalize_flow_output_path(self, request, output_path):
        """Normalize flow output_path for compute_* handlers.

        Many clients pass an output directory ("--output-dir"). For flow
        compute/load, we need a *base flow path* (ending in "_optical_flow.npz")
        so we can derive a stable video prefix and build descriptive filenames.

        Accepts either:
          - directory path -> synthesize "<dir>/<video>_optical_flow.npz"
          - file path -> return unchanged
        """
        if not output_path:
            return output_path

        try:
            is_dir = os.path.isdir(output_path)
        except Exception:
            is_dir = False

        if not is_dir:
            return output_path

        video_path = request.get("video_path")
        video_name = request.get("video_name")
        if video_name:
            base_name = str(video_name)
        elif video_path:
            base_name = os.path.splitext(os.path.basename(str(video_path)))[0]
        else:
            base_name = "video"

        return os.path.join(output_path, f"{base_name}_optical_flow.npz")
    
    def _get_locotrack_manager(self):
        """Lazily initialize LocoTrack model manager on first use."""
        if self.locotrack_manager is None:
            print("Initializing LocoTrack model manager...")
            try:
                from locotrack_flow import LocoTrackModelManager
                self.locotrack_manager = LocoTrackModelManager(
                    device=self.model_manager.device,
                    model_size="base"
                )
            except Exception as e:
                raise RuntimeError(f"Failed to initialize LocoTrack: {e}")

        # The server's clear_memory() can unload the model while keeping the manager instance.
        # Ensure we reload on next use.
        if hasattr(self.locotrack_manager, "ensure_loaded"):
            self.locotrack_manager.ensure_loaded()
        return self.locotrack_manager
        
    def start(self):
        """Start the server (Unix socket or TCP based on configuration)."""
        if self.use_tcp:
            self._start_tcp_server()
        else:
            self._start_unix_server()
    
    def _start_unix_server(self):
        """Start Unix socket server (Linux/Mac)."""
        # Remove old socket if exists
        if os.path.exists(self.socket_path):
            os.unlink(self.socket_path)
        
        # Create Unix socket
        self.server_socket = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self.server_socket.bind(self.socket_path)
        self.server_socket.listen(5)  # Allow multiple pending connections for cancel commands
        self.running = True
        
        print(f"✓ Tracking server listening on: {self.socket_path}")
        print("Ready to accept tracking requests...")
        
        import threading
        
        # Handle connections - use threading to allow cancel commands during operations
        while self.running:
            try:
                conn, _ = self.server_socket.accept()
                # Handle client in a thread so we can accept cancel commands
                client_thread = threading.Thread(target=self._handle_client, args=(conn,), daemon=True)
                client_thread.start()
            except Exception as e:
                if self.running:
                    print(f"ERROR handling client: {e}", file=sys.stderr)
    
    def _start_tcp_server(self):
        """Start TCP socket server (Windows)."""
        import threading
        
        # Create TCP socket
        self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.server_socket.bind((self.tcp_host, self.tcp_port))
        self.server_socket.listen(5)  # Allow multiple pending connections for cancel commands
        self.running = True
        
        print(f"✓ Tracking server listening on: {self.tcp_host}:{self.tcp_port}")
        print("Ready to accept tracking requests...")
        
        # Handle connections - use threading to allow cancel commands during operations
        while self.running:
            try:
                conn, addr = self.server_socket.accept()
                print(f"  Connection from {addr}")
                # Handle client in a thread so we can accept cancel commands
                client_thread = threading.Thread(target=self._handle_client, args=(conn,), daemon=True)
                client_thread.start()
            except Exception as e:
                if self.running:
                    print(f"ERROR handling client: {e}", file=sys.stderr)
    
    def _handle_client(self, conn):
        """Handle a client connection."""
        try:
            # Receive request (format: JSON with newline delimiter)
            data = b""
            while True:
                chunk = conn.recv(4096)
                if not chunk:
                    break
                data += chunk
                if b"\n" in data:
                    break
            
            if not data:
                return
            
            # Parse request
            request = json.loads(data.decode('utf-8'))
            command = request.get("command")
            
            # Handle cancel command immediately without acquiring lock
            if command == "cancel":
                if self._operation_in_progress.is_set():
                    self._cancelled.set()
                    response = {"status": "ok", "message": "Cancellation requested"}
                    print("⚠️ Cancellation requested by client")
                else:
                    response = {"status": "ok", "message": "No operation in progress to cancel"}
                # Send response and return immediately
                response_data = json.dumps(response).encode('utf-8') + b"\n"
                conn.sendall(response_data)
                conn.close()
                return
            
            # Handle status command - check if operation is in progress
            if command == "status":
                busy = self._operation_in_progress.is_set()
                response = {
                    "status": "ok", 
                    "busy": busy,
                    "message": "Operation in progress" if busy else "Server idle"
                }
                response_data = json.dumps(response).encode('utf-8') + b"\n"
                conn.sendall(response_data)
                conn.close()
                return
            
            # For long-running operations, check if server is busy
            long_running_commands = {
                "compute_flow", "compute_dis_flow", "compute_locotrack_flow", 
                "compute_trackpy_flow", "propagate_track", "optimize_track",
                "optimize_tracks", "visualize_flow", "preview_trackpy_trajectories",
                # These can be slow (I/O heavy) and should support Cancel
                "compress_video", "preview_dog_detection", "preview_trackpy_detection",
                # Fine-tuning is a long-running GPU operation
                "finetune_locotrack",
            }
            
            if command in long_running_commands:
                # Try to acquire lock to prevent concurrent operations
                if not self._operation_lock.acquire(blocking=False):
                    response = {
                        "status": "error", 
                        "message": "Server busy - another operation is in progress. Cancel it first or wait.",
                        "busy": True
                    }
                    response_data = json.dumps(response).encode('utf-8') + b"\n"
                    conn.sendall(response_data)
                    conn.close()
                    return
                
                # Mark operation as in progress and clear any previous cancellation
                self._cancelled.clear()
                self._operation_in_progress.set()
            
            try:
                if command == "ping":
                    response = {"status": "ok", "message": "pong"}
                elif command == "get_gpu_info":
                    response = self._get_gpu_info()
                elif command == "clear_cache":
                    response = self._clear_cache(request)
                elif command == "unload_gpu_models" or command == "clear_memory":
                    response = self._clear_memory(request)
                elif command == "compress_video":
                    response = self._compress_video(request)
                elif command == "compute_flow":
                    response = self._compute_flow(request)
                elif command == "compute_dis_flow":
                    response = self._compute_dis_flow(request)
                elif command == "compute_locotrack_flow":
                    response = self._compute_locotrack_flow(request)
                elif command == "compute_trackpy_flow":
                    response = self._compute_trackpy_flow(request)
                elif command == "load_flow":
                    response = self._load_flow(request)
                elif command == "preview_dog_detection":
                    response = self._preview_dog_detection(request)
                elif command == "preview_trackpy_detection":
                    response = self._preview_trackpy_detection(request)
                elif command == "preview_trackpy_trajectories":
                    response = self._preview_trackpy_trajectories(request)
                elif command == "propagate_track":
                    response = self._propagate_track(request)
                elif command == "optimize_track":
                    response = self._optimize_track(request)
                elif command == "optimize_tracks":
                    response = self._optimize_tracks(request)
                elif command == "physics_optimize_global":
                    response = self._physics_optimize_global(request)
                elif command == "get_mesh_preview":
                    response = self._get_mesh_preview(request)
                elif command == "visualize_flow":
                    response = self._visualize_flow(request)
                elif command == "finetune_locotrack":
                    response = self._finetune_locotrack(request)
                elif command == "stop":
                    response = {"status": "ok", "message": "stopping"}
                    self.running = False
                else:
                    response = {"status": "error", "message": f"Unknown command: {command}"}
            except OperationCancelledError as e:
                response = {"status": "cancelled", "message": str(e)}
            finally:
                # Release lock and clear in-progress flag for long-running operations
                if command in long_running_commands:
                    self._operation_in_progress.clear()
                    self._operation_lock.release()
            
            # Send response
            response_data = json.dumps(response).encode('utf-8') + b"\n"
            conn.sendall(response_data)
            
        except Exception as e:
            # CRITICAL: Clean up GPU memory on any error to prevent memory leaks
            # This is especially important for CUDA OOM errors where tensors may
            # still be allocated on the GPU
            cleanup_gpu_memory()

            # Some exceptions (e.g., AssertionError()) stringify to an empty message.
            # Always include the exception type so callers (Java UI) get actionable output.
            raw_msg = str(e).strip()
            if not raw_msg:
                raw_msg = repr(e)

            error_type = type(e).__name__
            error_msg = raw_msg
            if raw_msg and not raw_msg.startswith(error_type):
                error_msg = f"{error_type}: {raw_msg}"

            # Provide more helpful message for OOM errors
            if "CUDA out of memory" in error_msg or "OutOfMemoryError" in error_msg:
                error_msg = (
                    f"GPU memory error: {error_msg}. "
                    "Try reducing video resolution or using DIS optical flow method instead of RAFT."
                )

            # Include a trimmed traceback for debugging. This is safe for local IPC
            # and dramatically reduces time-to-fix when Java surfaces only exit code 1.
            try:
                import traceback as _traceback

                tb = _traceback.format_exc()
                tb_trimmed = tb[-4000:] if tb and len(tb) > 4000 else tb
            except Exception:
                tb_trimmed = None

            error_response = {
                "status": "error",
                "message": error_msg,
                "error_type": error_type,
            }
            if tb_trimmed:
                error_response["traceback"] = tb_trimmed
            try:
                conn.sendall(json.dumps(error_response).encode('utf-8') + b"\n")
            except:
                pass
        finally:
            conn.close()
    
    def start_socket_server(self):
        """Start the Unix socket server."""
        self.server_socket = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self.server_socket.bind(self.socket_path)
        self.server_socket.listen(1)
        self.running = True
        
        print(f"✓ Tracking server listening on: {self.socket_path}")
        print("Ready to accept tracking requests...")
        
        # Handle connections
        while self.running:
            try:
                conn, _ = self.server_socket.accept()
                self._handle_client(conn)
            except Exception as e:
                if self.running:
                    print(f"ERROR handling client: {e}", file=sys.stderr)

    def _get_gpu_info(self):
        """Get information about available GPUs.
        
        Returns:
            Dict with:
                - cuda_available: bool
                - gpu_count: int
                - gpus: List of dicts with index, name, memory_total, memory_free
                - current_device: int (currently selected CUDA device)
        """
        try:
            if not torch.cuda.is_available():
                return {
                    "status": "ok",
                    "cuda_available": False,
                    "gpu_count": 0,
                    "gpus": [],
                    "current_device": -1
                }
            
            gpu_count = torch.cuda.device_count()
            gpus = []
            
            for i in range(gpu_count):
                props = torch.cuda.get_device_properties(i)
                # Get memory info
                try:
                    mem_total = props.total_memory / (1024 ** 3)  # GB
                    torch.cuda.set_device(i)
                    mem_free = (props.total_memory - torch.cuda.memory_reserved(i)) / (1024 ** 3)
                except:
                    mem_total = 0
                    mem_free = 0
                
                gpus.append({
                    "index": i,
                    "name": props.name,
                    "memory_total_gb": round(mem_total, 2),
                    "memory_free_gb": round(mem_free, 2),
                    "compute_capability": f"{props.major}.{props.minor}"
                })
            
            # Get current device
            current_device = torch.cuda.current_device() if gpu_count > 0 else -1
            
            return {
                "status": "ok",
                "cuda_available": True,
                "gpu_count": gpu_count,
                "gpus": gpus,
                "current_device": current_device,
                "cuda_version": torch.version.cuda
            }
        except Exception as e:
            return {
                "status": "error",
                "message": f"Error getting GPU info: {str(e)}",
                "cuda_available": False,
                "gpu_count": 0,
                "gpus": []
            }

    def _clear_cache(self, request):
        """Explicitly clear all cached flow data and free memory.
        
        This should be called when opening a new video to ensure the old
        video's flow data is freed from memory.
        """
        import gc
        
        old_cache_size = sum(
            entry.get('flows_np').nbytes if hasattr(entry.get('flows_np'), 'nbytes') else 0
            for entry in self.flow_cache.values()
        ) / (1024 * 1024)
        
        print(f"🧹 Clearing cache: {old_cache_size:.0f} MB flow data")
        
        # Clear all server caches
        self.flow_cache.clear()
        self.video_metadata.clear()
        self.segment_cache.clear()
        self._current_video_path = None
        
        # Clear global offset cache (small but complete cleanup)
        global _OFFSET_CACHE
        _OFFSET_CACHE.clear()
        
        # Clear model manager's tensor cache if it exists
        if hasattr(self.model_manager, '_tensor_cache'):
            self.model_manager._tensor_cache.clear()
        
        # Clear LocoTrack manager cache if loaded
        if self.locotrack_manager is not None:
            if hasattr(self.locotrack_manager, '_tensor_cache'):
                self.locotrack_manager._tensor_cache.clear()
            if hasattr(self.locotrack_manager, 'clear_cache'):
                self.locotrack_manager.clear_cache()
        
        # Force multiple garbage collection passes to clean up circular references
        gc.collect()
        gc.collect()
        gc.collect()
        
        # Clear torch.compile dynamo cache if available (PyTorch 2.0+)
        # This can hold ~500MB of compiled graphs
        try:
            torch._dynamo.reset()
            print("   torch._dynamo cache reset")
        except (ImportError, AttributeError):
            pass  # Not available in older PyTorch
        
        # Clear CUDA cache
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()  # Wait for all GPU ops to complete
            print("   CUDA cache cleared and synchronized")
        
        # Report final memory state
        try:
            import psutil
            mem = psutil.virtual_memory()
            print(f"   System memory: {mem.available / (1024**3):.1f} GB available")
        except ImportError:
            pass
        
        return {
            "status": "ok",
            "message": f"Cache cleared ({old_cache_size:.0f} MB freed)"
        }

    def _clear_memory(self, request):
        """Clear flow cache (RAM) and unload GPU models (VRAM).
        
        Called when switching between ANY flow methods to free memory.
        This clears:
        - Flow cache (numpy arrays, can be several GB)
        - Segment cache (anchor segment interpolations)
        - Offset cache (kernel/window precomputed data)
        - GPU models (RAFT ~500MB, LocoTrack ~500MB)
        - Tensor caches
        - torch.compile cached graphs
        
        Models will be automatically reloaded when needed.
        """
        import gc
        global _OFFSET_CACHE
        
        unloaded = []
        
        print("🧹 Clearing memory before flow method switch...")
        
        # Report initial state
        if torch.cuda.is_available():
            vram_before = torch.cuda.memory_allocated(0) / 1024**2
            print(f"   VRAM before: {vram_before:.1f} MB")
        else:
            vram_before = 0
        
        # 1. Clear flow cache (main RAM consumer - can be several GB)
        old_flow_size = sum(
            entry.get('flows_np').nbytes if hasattr(entry.get('flows_np'), 'nbytes') else 0
            for entry in self.flow_cache.values()
        ) / (1024 * 1024)
        if old_flow_size > 0:
            print(f"   Clearing {old_flow_size:.0f} MB flow cache (RAM)")
        self.flow_cache.clear()
        self.video_metadata.clear()
        
        # 1b. Clear segment cache (anchor segment interpolation results)
        segment_cache_size = len(self.segment_cache._store) if hasattr(self.segment_cache, '_store') else 0
        if segment_cache_size > 0:
            print(f"   Clearing segment cache ({segment_cache_size} entries)")
        self.segment_cache.clear()
        
        # 1c. Clear offset cache (precomputed kernel/window data)
        offset_cache_size = len(_OFFSET_CACHE)
        if offset_cache_size > 0:
            print(f"   Clearing offset cache ({offset_cache_size} entries)")
        _OFFSET_CACHE.clear()
        
        # 2. Unload RAFT model (GPU)
        if hasattr(self.model_manager, 'unload_model'):
            if self.model_manager.unload_model():
                unloaded.append("RAFT")
        
        # 3. Unload LocoTrack model (GPU)
        if self.locotrack_manager is not None:
            if hasattr(self.locotrack_manager, 'unload_model'):
                if self.locotrack_manager.unload_model():
                    unloaded.append("LocoTrack")
            elif hasattr(self.locotrack_manager, 'model') and self.locotrack_manager.model is not None:
                try:
                    self.locotrack_manager.model = self.locotrack_manager.model.cpu()
                except:
                    pass
                del self.locotrack_manager.model
                self.locotrack_manager.model = None
                unloaded.append("LocoTrack")
        
        # 4. Clear tensor caches
        if hasattr(self.model_manager, '_tensor_cache'):
            self.model_manager._tensor_cache.clear()
        if self.locotrack_manager and hasattr(self.locotrack_manager, '_tensor_cache'):
            self.locotrack_manager._tensor_cache.clear()
        
        # 5. Garbage collection
        gc.collect()
        gc.collect()
        gc.collect()
        
        # 6. Clear torch dynamo cache (releases compiled CUDA graphs)
        try:
            torch._dynamo.reset()
        except (ImportError, AttributeError):
            pass
        
        # 7. Clear CUDA cache
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            
            vram_after = torch.cuda.memory_allocated(0) / 1024**2
            vram_freed = vram_before - vram_after
            print(f"   VRAM after: {vram_after:.1f} MB (freed {vram_freed:.1f} MB)")
        else:
            vram_freed = 0
        
        models_str = ', '.join(unloaded) if unloaded else 'none loaded'
        print(f"✓ Memory cleared: {old_flow_size:.0f} MB flow cache, models: {models_str}")
        
        return {
            "status": "ok",
            "message": f"Cleared {old_flow_size:.0f} MB flow cache, unloaded: {models_str}",
            "flow_cache_freed_mb": old_flow_size,
            "vram_freed_mb": vram_freed,
            "unloaded_models": unloaded,
        }

    def _compress_video(self, request):
        """Memory-efficient video compression using OpenCV and tifffile.
        
        This is MUCH faster than ImageJ's frame-by-frame approach while still
        being memory-efficient (processes one frame at a time):
        - Uses tifffile.TiffFile for memory-mapped frame access
        - Uses OpenCV for fast CPU-based resizing (one frame at a time)
        - Preserves original dtype (uint8/uint16)
        - Peak memory: ~2 frames worth instead of entire video
        
        Request parameters:
            input_path: Path to input video (TIFF or AVI)
            output_path: Path to save compressed TIFF
            target_width: Target width in pixels
            target_height: Target height in pixels
            
        Returns:
            status, output_path, original_shape, compressed_shape, elapsed_time
        """
        # Check for cancellation before starting
        self._check_cancelled("Video compression")

        input_path = request.get("input_path")
        output_path = request.get("output_path")
        target_width = request.get("target_width")
        target_height = request.get("target_height")
        
        if not input_path or not output_path:
            return {"status": "error", "message": "Missing input_path or output_path"}
        if not target_width or not target_height:
            return {"status": "error", "message": "Missing target_width or target_height"}
        
        print(f"=" * 60)
        print(f"Memory-Efficient Video Compression")
        print(f"=" * 60)
        print(f"Input:  {input_path}")
        print(f"Output: {output_path}")
        print(f"Target: {target_width}x{target_height}")
        
        import cv2
        start_time = _pc()
        
        try:
            video_path_str = str(input_path)
            
            if video_path_str.lower().endswith('.avi'):
                # AVI: Use OpenCV VideoCapture for frame-by-frame access
                return self._compress_avi_frame_by_frame(
                    video_path_str, output_path, target_width, target_height, start_time
                )
            else:
                # TIFF: Use tifffile for frame-by-frame access
                return self._compress_tiff_frame_by_frame(
                    video_path_str, output_path, target_width, target_height, start_time
                )
                
        except Exception as e:
            print(f"[ERROR] Compression failed: {e}")
            import traceback
            traceback.print_exc()
            return {"status": "error", "message": str(e)}
    
    def _compress_tiff_frame_by_frame(self, input_path, output_path, target_width, target_height, start_time):
        """Compress TIFF frame-by-frame to avoid loading entire video into memory."""
        import cv2
        
        with tifffile.TiffFile(input_path) as tif:
            n_frames = len(tif.pages)
            if n_frames == 0:
                return {"status": "error", "message": "No frames found in TIFF"}
            
            # Get info from first frame
            first_page = tif.pages[0]
            original_shape = first_page.shape
            original_dtype = first_page.dtype
            H, W = original_shape[:2]
            
            # Check if it's RGB
            is_rgb = len(original_shape) == 3 and original_shape[-1] in (3, 4)
            
            print(f"  TIFF: {W}x{H} x {n_frames} frames, dtype={original_dtype}, RGB={is_rgb}")
            print(f"  Processing frame-by-frame (memory-efficient mode)")
            
            # Collect all resized frames first (memory-efficient for reasonable sizes)
            # We need to do this because imagej=True requires knowing all frames upfront
            resized_frames = []
            for i in range(n_frames):
                if i % 10 == 0:
                    self._check_cancelled("Video compression")
                # Read single frame
                frame = tif.pages[i].asarray()
                
                # Handle RGB -> grayscale if needed
                if is_rgb or (frame.ndim == 3 and frame.shape[-1] in (3, 4)):
                    frame = np.mean(frame[..., :3], axis=-1).astype(original_dtype)
                
                # Resize with OpenCV
                resized_frame = cv2.resize(frame, (target_width, target_height), 
                                           interpolation=cv2.INTER_LINEAR)
                resized_frames.append(resized_frame)
                
                # Progress update every 50 frames
                if (i + 1) % 50 == 0:
                    elapsed = _pc() - start_time
                    fps = (i + 1) / elapsed
                    print(f"  Resizing frame {i+1}/{n_frames} ({fps:.1f} fps)")
            
            # Stack and save as ImageJ-compatible TIFF
            print(f"  Writing ImageJ-compatible TIFF...")
            self._check_cancelled("Video compression")
            resized_stack = np.stack(resized_frames, axis=0)
            tifffile.imwrite(output_path, resized_stack, imagej=True)
        
        total_time = _pc() - start_time
        print(f"✓ Compression complete in {total_time:.2f}s ({n_frames/total_time:.1f} fps)")
        
        return {
            "status": "ok",
            "output_path": output_path,
            "original_width": W,
            "original_height": H,
            "compressed_width": target_width,
            "compressed_height": target_height,
            "num_frames": n_frames,
            "elapsed_time": total_time
        }
    
    def _compress_avi_frame_by_frame(self, input_path, output_path, target_width, target_height, start_time):
        """Compress AVI frame-by-frame to avoid loading entire video into memory."""
        import cv2
        
        cap = cv2.VideoCapture(input_path)
        if not cap.isOpened():
            return {"status": "error", "message": f"Could not open AVI: {input_path}"}
        
        n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        print(f"  AVI: {W}x{H} x {n_frames} frames")
        print(f"  Processing frame-by-frame (memory-efficient mode)")
        
        resized_frames = []
        frames_processed = 0
        
        while True:
            if frames_processed % 10 == 0:
                self._check_cancelled("Video compression")
            ret, frame = cap.read()
            if not ret:
                break
            
            # Convert BGR to grayscale if needed
            if frame.ndim == 3:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # Resize
            resized_frame = cv2.resize(frame, (target_width, target_height),
                                       interpolation=cv2.INTER_LINEAR)
            resized_frames.append(resized_frame)
            frames_processed += 1
            
            # Progress update every 50 frames
            if frames_processed % 50 == 0:
                elapsed = _pc() - start_time
                fps = frames_processed / elapsed
                print(f"  Resizing frame {frames_processed}/{n_frames} ({fps:.1f} fps)")
        
        cap.release()
        
        # Stack and save as ImageJ-compatible TIFF
        print(f"  Writing ImageJ-compatible TIFF...")
        self._check_cancelled("Video compression")
        resized_stack = np.stack(resized_frames, axis=0)
        tifffile.imwrite(output_path, resized_stack, imagej=True)
        
        total_time = _pc() - start_time
        print(f"✓ Compression complete in {total_time:.2f}s ({frames_processed/total_time:.1f} fps)")
        
        return {
            "status": "ok",
            "output_path": output_path,
            "original_width": W,
            "original_height": H,
            "compressed_width": target_width,
            "compressed_height": target_height,
            "num_frames": frames_processed,
            "elapsed_time": total_time
        }

    def _store_flow_entry(self, video_path, flows, metadata=None, force_float32=False):
        """Store optical flow in cache with memory-efficient float16 compression.
        
        Args:
            video_path: Path to video file (cache key)
            flows: Flow array (T-1, H, W, 2)
            metadata: Optional metadata dict
            force_float32: If True, disable float16 compression (for testing)
        
        Returns:
            Cache entry dict
        """
        # Calculate memory size of the flow array
        flow_size_mb = np.prod(flows.shape) * 4 / (1024 * 1024)  # float32 size
        available_mb = get_available_memory_mb()
        
        # Use float16 if:
        # 1. Array is large (> FLOAT16_THRESHOLD_MB), OR
        # 2. Available memory is low (< LOW_MEMORY_THRESHOLD_MB)
        use_float16 = not force_float32 and (
            flow_size_mb > FLOAT16_THRESHOLD_MB or 
            available_mb < LOW_MEMORY_THRESHOLD_MB
        )
        
        if use_float16:
            flows_np = MemoryEfficientFlowArray(flows, use_float16=True)
            saved_mb = flows_np.memory_savings_mb()
            print(f"  💾 Using float16 storage: {flows_np.nbytes/(1024*1024):.0f} MB "
                  f"(saved {saved_mb:.0f} MB, avail RAM: {available_mb:.0f} MB)")
        else:
            # Standard float32 storage for small arrays with plenty of RAM
            flows_np = np.array(flows, dtype=np.float32, copy=True)

        entry = {
            "flows_np": flows_np,
            "timestamp": _pc(),
        }

        if metadata:
            entry["metadata"] = metadata
            self.video_metadata[video_path] = metadata

        self.flow_cache[video_path] = entry
        return entry
    
    @staticmethod
    def _normalize_corridor_width(value):
        """Normalize corridor_width parameter."""
        if value is None:
            return None
        if isinstance(value, str):
            v = value.strip().lower()
            if v in ("", "auto", "adaptive"):
                return None  # Auto/adaptive mode
            try:
                return float(v)
            except ValueError as exc:
                raise ValueError(f"Invalid corridor width: {value}") from exc
        if isinstance(value, (int, float)):
            return float(value)
        raise ValueError(f"Unsupported corridor width type: {type(value).__name__}")
    
    def _load_video_raw_and_normalized(self, video_path, target_width=None, target_height=None):
        """Load video and return both raw and normalized versions.
        
        Args:
            video_path: Path to video file (TIFF or AVI)
            target_width: Optional target width for resizing. If None or <=0, no resizing.
            target_height: Optional target height for resizing. If None or <=0, no resizing.
        
        Returns:
            tuple: (raw_video, normalized_video, is_avi, original_shape, resized_shape)
            - raw_video: uint8/float32 in source intensity range (after any resizing)
            - normalized_video: float32, either min-max normalized (TIFF) or /255 (AVI)
            - is_avi: True if AVI path
            - original_shape: shape before resize (if any)
            - resized_shape: shape after any resize applied
        """
        video_path_str = str(video_path)
        
        # Determine if resizing is requested
        should_resize = (target_width is not None and target_width > 0 and 
                         target_height is not None and target_height > 0)
        
        if video_path_str.lower().endswith('.avi'):
            print(f"Loading AVI with mediapy: {video_path_str}")
            video = media.read_video(video_path_str)  # (T, H, W, C) uint8
            orig_shape = video.shape[:3]
            print(f"Original AVI shape: {orig_shape}")

            # Ensure RGB
            if video.ndim == 3:
                video = np.stack([video, video, video], axis=-1)
            elif video.ndim == 4 and video.shape[-1] == 1:
                video = np.repeat(video, 3, axis=-1)
            elif video.ndim == 4 and video.shape[-1] != 3:
                if video.shape[-1] > 3:
                    video = video[..., :3]
                else:
                    pad_width = [(0, 0)] * 3 + [(0, 3 - video.shape[-1])]
                    video = np.pad(video, pad_width, mode='constant', constant_values=0)

            # Normalize mediapy outputs to uint8 RGB.
            # mediapy may return float frames depending on backend/codecs.
            if video.dtype != np.uint8:
                video = np.nan_to_num(video, nan=0.0, posinf=255.0, neginf=0.0)
                vmax = float(np.max(video)) if video.size else 0.0
                if np.issubdtype(video.dtype, np.floating) and vmax <= 1.0:
                    video = np.clip(video * 255.0, 0.0, 255.0).astype(np.uint8)
                else:
                    video = np.clip(video, 0.0, 255.0).astype(np.uint8)

            # Only resize if explicitly requested
            if should_resize:
                video_resized = media.resize_video(video, (target_height, target_width))
                print(f"Resized AVI to: {video_resized.shape}")
                raw_video = video_resized
            else:
                print(f"Using original AVI resolution: {video.shape}")
                raw_video = video
            
            normalized_video = raw_video.astype(np.float32) / 255.0
            resized_shape = raw_video.shape[:3]
            return raw_video, normalized_video, True, orig_shape, resized_shape

        # TIFF path
        # Some TIFFs are written as many independent pages/series; in those cases
        # tifffile.imread() may return only the first 2D page.
        with tifffile.TiffFile(video_path_str) as tif:
            self._check_cancelled("Video loading")
            vol = tif.asarray()

            if vol.ndim == 2 and len(tif.pages) > 1:
                frames = []
                for i, page in enumerate(tif.pages):
                    if i % 10 == 0:
                        self._check_cancelled("Video loading")
                    frames.append(page.asarray())
                vol = np.stack(frames, axis=0)

        print(f"  Loaded TIFF: dtype={vol.dtype}, shape={vol.shape}")
        
        # Handle RGB TIFFs (4D: T, H, W, C or H, W, T, C) - convert to grayscale
        if vol.ndim == 4:
            # Last dimension is likely channels (RGB = 3 or RGBA = 4)
            if vol.shape[-1] in (3, 4):
                # Shape is (T, H, W, C) or (H, W, T, C)
                print(f"  RGB/RGBA TIFF detected with {vol.shape[-1]} channels, converting to grayscale")
                vol = np.mean(vol[..., :3], axis=-1)  # Average RGB channels
            elif vol.shape[1] in (3, 4):
                # Shape might be (T, C, H, W) - channel-first format
                print(f"  Channel-first RGB TIFF detected, converting to grayscale")
                vol = np.mean(vol[:, :3, :, :], axis=1)
            else:
                raise ValueError(f"Unexpected 4D TIFF shape: {vol.shape}")

        # Handle different orientations for 3D volumes
        # Only reorder if the TIFF is clearly in (H, W, T) format
        # Heuristic: if last dim is small (<= 10) AND smaller than both other dims, it's likely T
        if vol.ndim == 3:
            if vol.shape[-1] <= 10 and vol.shape[-1] < vol.shape[0] and vol.shape[-1] < vol.shape[1]:
                print(f"  Reordering from (H, W, T) to (T, H, W)")
                vol = np.moveaxis(vol, -1, 0)

        vol = vol.astype(np.float32)

        if vol.ndim != 3:
            raise ValueError(f"Expected 3D volume (T,H,W) after preprocessing, got shape {vol.shape}")

        orig_shape = vol.shape
        
        # Resize TIFF if requested AND size actually differs
        if should_resize:
            T, H, W = vol.shape
            if H != target_height or W != target_width:
                import cv2
                print(f"Resizing TIFF from {H}x{W} to {target_height}x{target_width}")
                resized_vol = np.zeros((T, target_height, target_width), dtype=np.float32)
                for t in range(T):
                    if t % 5 == 0:
                        self._check_cancelled("Video resizing")
                    resized_vol[t] = cv2.resize(vol[t], (target_width, target_height), interpolation=cv2.INTER_LINEAR)
                vol = resized_vol
                print(f"Resized TIFF to: {vol.shape}")

        raw_video = vol  # keep raw intensities for DoG
        vmin, vmax = float(vol.min()), float(vol.max())
        den = max(vmax - vmin, 1e-6)
        normalized_video = (vol - vmin) / den

        T, H, W = vol.shape
        return raw_video, normalized_video, False, orig_shape, (T, H, W)

    def _load_and_normalize_video(self, video_path, target_width=None, target_height=None):
        """Backward-compatible wrapper: return only normalized video."""
        raw_video, normalized_video, is_avi, orig_shape, resized_shape = self._load_video_raw_and_normalized(
            video_path, target_width=target_width, target_height=target_height
        )
        return normalized_video, is_avi, orig_shape
    
    def _compute_flow(self, request):
        """Compute optical flow using RAFT."""
        # Check for cancellation before starting
        self._check_cancelled("RAFT flow computation")
        
        video_path = request["video_path"]
        force_recompute = request.get("force_recompute", False)
        output_path = request.get("output_path", None)
        target_width = request.get("target_width", None)
        target_height = request.get("target_height", None)
        save_to_disk = request.get("save_to_disk", True)  # Default True for backward compatibility
        
        # Ensure force_recompute is a boolean (handle string values from CLI)
        if isinstance(force_recompute, str):
            force_recompute = force_recompute.lower() in ('true', '1', 'yes')
        else:
            force_recompute = bool(force_recompute)
        
        print(f"[RAFT FLOW] force_recompute={force_recompute}")

        # Allow caller to pass output_path as a directory ("--output-dir")
        output_path = self._normalize_flow_output_path(request, output_path)
        
        # Clear old flow cache if switching to a different video
        self._clear_flow_cache_for_new_video(video_path)
        
        # Check in-memory cache first - verify method matches
        cache_entry = self.flow_cache.get(video_path)
        cached_method = cache_entry.get("metadata", {}).get("method", "raft") if cache_entry else None
        
        if cache_entry and not force_recompute and cached_method == "raft":
            print(f"Using cached RAFT flow from memory for {video_path}")
            flows = cache_entry["flows_np"]
        # Check disk cache - search by pattern since filenames include parameters
        elif output_path and not force_recompute:
            # Find existing flow file matching resolution (exact path or pattern match)
            disk_path = output_path if os.path.exists(output_path) else self._find_existing_flow_file(
                output_path, 'raft', target_width=target_width, target_height=target_height)
            
            if disk_path and os.path.exists(disk_path):
                print(f"Loading flow from disk cache: {disk_path}")
                try:
                    data = np.load(disk_path, allow_pickle=False)
                    # Memory-efficient loading with automatic float16 compression
                    flows = load_flows_memory_efficient(data['flows'])
                    
                    # Load metadata if available - check method matches
                    disk_method = str(data.get('method', 'raft')) if 'method' in data.files else 'raft'
                    if disk_method != 'raft':
                        print(f"Disk cache has method '{disk_method}', need 'raft'. Recomputing...")
                        flows = self._compute_flow_from_video(video_path, output_path, target_width, target_height)
                    else:
                        # CRITICAL: Validate resolution matches current working video
                        if 'resized_shape' in data.files:
                            cached_shape = tuple(int(x) for x in data['resized_shape'])
                            cached_H, cached_W = cached_shape[1], cached_shape[2]
                            flow_H, flow_W = flows.shape[1], flows.shape[2]
                            
                            # Check if target resolution was specified and differs from cache
                            if target_width and target_height:
                                if flow_W != target_width or flow_H != target_height:
                                    print(f"⚠ Resolution mismatch: cached flow is {flow_W}x{flow_H}, "
                                          f"need {target_width}x{target_height}. Recomputing...")
                                    flows = self._compute_flow_from_video(video_path, output_path, target_width, target_height)
                                else:
                                    if 'original_shape' in data:
                                        metadata = {
                                            'original_shape': tuple(int(x) for x in data['original_shape']),
                                            'resized_shape': cached_shape,
                                            'is_avi': bool(data['is_avi']),
                                            'method': 'raft',
                                            'upscaled_for_raft': bool(data['upscaled_for_raft']) if 'upscaled_for_raft' in data.files else False,
                                            'raft_upscale_factor': float(data['raft_upscale_factor']) if 'raft_upscale_factor' in data.files else 1.0,
                                        }
                                    else:
                                        metadata = {'method': 'raft', 'resized_shape': cached_shape}
                                    cache_entry = self._store_flow_entry(video_path, flows, metadata=metadata)
                                    flows = cache_entry["flows_np"]
                                    print(f"✓ RAFT flow loaded from disk ({flows.shape})")
                            else:
                                # No target resolution specified, accept cached flow
                                if 'original_shape' in data:
                                    metadata = {
                                        'original_shape': tuple(int(x) for x in data['original_shape']),
                                        'resized_shape': cached_shape,
                                        'is_avi': bool(data['is_avi']),
                                        'method': 'raft',
                                        'upscaled_for_raft': bool(data['upscaled_for_raft']) if 'upscaled_for_raft' in data.files else False,
                                        'raft_upscale_factor': float(data['raft_upscale_factor']) if 'raft_upscale_factor' in data.files else 1.0,
                                    }
                                else:
                                    metadata = {'method': 'raft', 'resized_shape': cached_shape}
                                cache_entry = self._store_flow_entry(video_path, flows, metadata=metadata)
                                flows = cache_entry["flows_np"]
                                print(f"✓ RAFT flow loaded from disk ({flows.shape})")
                        else:
                            # No resolution info in cache, accept but warn
                            print(f"⚠ Cached flow has no resolution metadata, using anyway")
                            metadata = {'method': 'raft'}
                            cache_entry = self._store_flow_entry(video_path, flows, metadata=metadata)
                            flows = cache_entry["flows_np"]
                            print(f"✓ RAFT flow loaded from disk ({flows.shape})")
                except Exception as e:
                    print(f"Failed to load from disk cache: {e}")
                    print("Computing RAFT flow from scratch...")
                    flows = self._compute_flow_from_video(video_path, output_path, target_width, target_height)
            else:
                if cached_method and cached_method != "raft":
                    print(f"Cache has method '{cached_method}', switching to RAFT...")
            flows = self._compute_flow_from_video(video_path, output_path if save_to_disk else None, target_width, target_height)
        else:
            if cached_method and cached_method != "raft":
                print(f"Cache has method '{cached_method}', switching to RAFT...")
            flows = self._compute_flow_from_video(video_path, output_path if save_to_disk else None, target_width, target_height)
        response = {
            "status": "ok",
            "message": "RAFT optical flow computed",
            "shape": list(flows.shape),
            "method": "raft",
        }
        
        if video_path in self.video_metadata:
            metadata = self.video_metadata[video_path]
            response["metadata"] = {
                "original_shape": list(metadata['original_shape']) if metadata['original_shape'] else None,
                "resized_shape": list(metadata['resized_shape']) if metadata['resized_shape'] else None,
                "is_avi": metadata['is_avi'],
                "method": "raft"
            }
        
        return response
    
    def _compute_flow_from_video(self, video_path, output_path=None, target_width=None, target_height=None):
        """Helper to compute RAFT flow from video file.
        
        Args:
            video_path: Path to video file
            output_path: Optional path to save computed flow
            target_width: Optional target width for resizing
            target_height: Optional target height for resizing
            
        Note: If the video resolution would exceed available GPU memory, this method
              will automatically downscale to a safe resolution.
        """
        print(f"Computing RAFT optical flow for {video_path}")
        if target_width and target_height:
            print(f"Using target resolution: {target_width}x{target_height}")
        start_time = _pc()
        
        # First, peek at video dimensions without loading full data
        video_path_str = str(video_path)
        if video_path_str.lower().endswith('.avi'):
            # For AVI, we need to load to get dimensions
            import mediapy as media
            video_peek = media.read_video(video_path_str)
            orig_T, orig_H, orig_W = video_peek.shape[:3]
            del video_peek
        else:
            # For TIFF, use tifffile to get shape without loading
            with tifffile.TiffFile(video_path_str) as tif:
                if len(tif.pages) > 0:
                    first_page = tif.pages[0]
                    orig_H, orig_W = first_page.shape[:2]
                    orig_T = len(tif.pages)
                else:
                    # Fallback to full load
                    orig_H, orig_W, orig_T = None, None, None
        
        # Determine effective resolution (after any requested resize)
        if target_width and target_height:
            eff_H, eff_W = target_height, target_width
        elif orig_H and orig_W:
            eff_H, eff_W = orig_H, orig_W
        else:
            # Will be determined after load
            eff_H, eff_W = None, None
        
        # Check GPU memory and warn if likely to OOM (but don't auto-scale)
        if eff_H and eff_W and torch.cuda.is_available():
            available_gpu = get_available_gpu_memory_gb()
            estimated_mem = estimate_raft_memory_gb(eff_H, eff_W)
            
            print(f"  GPU memory check: {eff_W}x{eff_H} estimated to need {estimated_mem:.1f} GB, "
                  f"available: {available_gpu:.1f} GB")
            
            if estimated_mem > available_gpu * 0.90:  # 90% threshold for warning
                print(f"  ⚠️ WARNING: Resolution {eff_W}x{eff_H} may exceed GPU memory!")
                print(f"     Estimated: {estimated_mem:.1f} GB, Available: {available_gpu:.1f} GB")
                print(f"     Consider compressing video to smaller resolution.")
        
        raw_video, v01, is_avi, orig_shape, resized_shape = self._load_video_raw_and_normalized(
            video_path, target_width=target_width, target_height=target_height
        )
        
        if is_avi:
            T, H, W, C = v01.shape
        else:
            T, H, W = v01.shape
        
        # RAFT minimum resolution check and auto-upscaling
        # RAFT requires at least 128x128 pixels due to the correlation pyramid architecture
        needs_upscale = H < RAFT_MINIMUM_RESOLUTION or W < RAFT_MINIMUM_RESOLUTION
        upscale_factor = 1.0
        original_raft_H, original_raft_W = H, W
        
        if needs_upscale:
            # Calculate upscale factor to reach minimum resolution
            upscale_factor = max(
                RAFT_MINIMUM_RESOLUTION / H,
                RAFT_MINIMUM_RESOLUTION / W
            )
            new_H = int(np.ceil(H * upscale_factor))
            new_W = int(np.ceil(W * upscale_factor))
            # Ensure dimensions are multiples of 8 (RAFT padding requirement)
            new_H = ((new_H + 7) // 8) * 8
            new_W = ((new_W + 7) // 8) * 8
            
            print(f"  ⚠️ Video resolution {W}x{H} is below RAFT minimum ({RAFT_MINIMUM_RESOLUTION}x{RAFT_MINIMUM_RESOLUTION})")
            print(f"  📐 Upscaling to {new_W}x{new_H} for RAFT processing (factor: {upscale_factor:.2f}x)")
            
            import cv2
            if is_avi:
                # AVI: (T, H, W, C) RGB uint8
                v01_upscaled = np.zeros((T, new_H, new_W, C), dtype=v01.dtype)
                for t in range(T):
                    if t % 20 == 0:
                        self._check_cancelled("Video upscaling")
                    v01_upscaled[t] = cv2.resize(v01[t], (new_W, new_H), interpolation=cv2.INTER_LINEAR)
                v01 = v01_upscaled
                H, W = new_H, new_W
            else:
                # TIFF: (T, H, W) float32
                v01_upscaled = np.zeros((T, new_H, new_W), dtype=v01.dtype)
                for t in range(T):
                    if t % 20 == 0:
                        self._check_cancelled("Video upscaling")
                    v01_upscaled[t] = cv2.resize(v01[t], (new_W, new_H), interpolation=cv2.INTER_LINEAR)
                v01 = v01_upscaled
                H, W = new_H, new_W
        
        metadata = {
            'original_shape': orig_shape if orig_shape else resized_shape,
            'resized_shape': resized_shape,
            'is_avi': is_avi,
            'method': 'raft',
            'upscaled_for_raft': needs_upscale,
            'raft_upscale_factor': upscale_factor if needs_upscale else 1.0,
            'raft_processing_resolution': (H, W) if needs_upscale else None,
        }
        self.video_metadata[video_path] = metadata
        
        # Compute memory estimate for error message
        available_gpu = get_available_gpu_memory_gb() if torch.cuda.is_available() else 0
        estimated_mem = estimate_raft_memory_gb(H, W)

        # Ensure RAFT model is loaded (may have been unloaded to free VRAM)
        self.model_manager.ensure_loaded()

        try:
            flows = self.model_manager.compute_optical_flow(
                v01,
                batch_pairs=1,
                is_avi=is_avi,
                cancel_check=lambda: self._check_cancelled("RAFT flow computation"),
            )
        except torch.cuda.OutOfMemoryError as e:
            # Clean up GPU memory
            cleanup_gpu_memory()
            
            # Calculate suggested max resolution
            safe_H, safe_W, _ = get_max_safe_resolution_for_raft(
                available_gpu, H, W, safety_margin=0.80
            )
            
            # Raise with special error code for Java to detect
            raise RuntimeError(
                f"GPU_OOM: Video resolution {W}x{H} requires ~{estimated_mem:.0f} GB but only "
                f"{available_gpu:.0f} GB available. Please compress the video to {safe_W}x{safe_H} or smaller."
            ) from e
        
        elapsed = _pc() - start_time
        print(f"✓ Flow computed in {elapsed:.2f}s")

        # If we upscaled the video for RAFT, downscale the flow back to original resolution
        if needs_upscale:
            print(f"  📐 Downscaling flow from {flows.shape[2]}x{flows.shape[1]} back to {original_raft_W}x{original_raft_H}")
            import cv2
            Tm1 = flows.shape[0]
            flows_downscaled = np.zeros((Tm1, original_raft_H, original_raft_W, 2), dtype=np.float32)
            
            for t in range(Tm1):
                if t % 20 == 0:
                    self._check_cancelled("Flow downscaling")
                # Downscale each flow component
                for c in range(2):
                    flows_downscaled[t, :, :, c] = cv2.resize(
                        flows[t, :, :, c], 
                        (original_raft_W, original_raft_H), 
                        interpolation=cv2.INTER_LINEAR
                    )
                # Scale the flow vectors by the inverse of upscale factor
                # since the motion magnitude is relative to the upscaled image
                flows_downscaled[t] /= upscale_factor
            
            flows = flows_downscaled
            print(f"  ✓ Flow downscaled to original resolution")

        # Cache the flow in memory
        entry = self._store_flow_entry(video_path, flows, metadata=metadata)
        flows_np = entry["flows_np"]
        
        # Save to file if requested (also save metadata including method)
        if output_path:
            # Build descriptive filename - RAFT includes size if resized
            actual_output_path = self._build_flow_filename(
                output_path, 'raft', 
                resized_shape=resized_shape,
                original_shape=orig_shape
            )
            # Get raw array for saving (handles MemoryEfficientFlowArray)
            flows_to_save = flows_np.get_raw() if hasattr(flows_np, 'get_raw') else flows_np
            # Use uncompressed savez for ~4x faster loading (8% larger files)
            # Include upscale metadata for small videos that were upscaled for RAFT
            np.savez(
                actual_output_path, 
                flows=flows_to_save,
                original_shape=metadata['original_shape'],
                resized_shape=metadata['resized_shape'],
                is_avi=is_avi,
                method='raft',
                upscaled_for_raft=metadata.get('upscaled_for_raft', False),
                raft_upscale_factor=metadata.get('raft_upscale_factor', 1.0)
            )
            print(f"✓ RAFT flow saved to {actual_output_path}")

        return flows_np
    
    def _compute_dis_flow(self, request):
        """Compute optical flow using OpenCV DISOpticalFlow.
        
        This is a fast, CPU-based optical flow method that works well for
        general motion tracking. It uses a coarse-to-fine approach with
        variational refinement.
        
        MEMORY OPTIMIZATION: Flow is computed and stored at (H/DS, W/DS) resolution,
        NOT upscaled to full resolution. This reduces memory by DS^2 factor.
        The flow values are scaled by DS so they represent displacement in original coords.
        
        Request parameters:
            video_path: Path to video file (TIFF or AVI)
            output_path: Optional path to save flow
            force_recompute: Force recomputation even if cached
            target_width: Target width for resizing (optional, from Java UI compression)
            target_height: Target height for resizing (optional, from Java UI compression)
            downsample_factor: Downsampling for DIS computation (default: 2)
                              Flow will be stored at this reduced resolution.
            save_to_disk: Whether to save computed flow to disk (default: True)
            incremental_allocation: Use incremental allocation (like jellyfish_tracker) for 
                                   low-memory systems. If not specified, auto-detected based
                                   on system RAM (enabled for systems with < 32GB RAM).
        """
        # Check for cancellation before starting
        self._check_cancelled("DIS flow computation")
        
        video_path = request["video_path"]
        force_recompute = request.get("force_recompute", False)
        output_path = request.get("output_path", None)
        target_width = request.get("target_width", None)
        target_height = request.get("target_height", None)
        downsample_factor = request.get("downsample_factor", 2)
        save_to_disk = request.get("save_to_disk", True)  # Default True for backward compatibility
        
        # DEBUG: Log all parameters received
        print(f"\n{'='*60}")
        print(f"[DIS FLOW] Request parameters:")
        print(f"  video_path: {video_path}")
        print(f"  force_recompute: {force_recompute} (type: {type(force_recompute).__name__})")
        print(f"  downsample_factor: {downsample_factor} (type: {type(downsample_factor).__name__})")
        print(f"  output_path: {output_path}")
        print(f"  target_width: {target_width}, target_height: {target_height}")
        print(f"{'='*60}\n")
        
        # Ensure force_recompute is a boolean (handle string values)
        if isinstance(force_recompute, str):
            force_recompute = force_recompute.lower() in ('true', '1', 'yes')
        else:
            force_recompute = bool(force_recompute)
        
        # Ensure downsample_factor is an integer
        if isinstance(downsample_factor, str):
            try:
                downsample_factor = int(downsample_factor)
            except ValueError:
                print(f"  WARNING: Invalid downsample_factor '{downsample_factor}', using 2")
                downsample_factor = 2
        
        # Incremental allocation mode: allocate flow arrays one at a time (like jellyfish_tracker)
        # This allows the OS to swap older arrays to disk, preventing OOM on low-memory systems
        incremental_req = request.get("incremental_allocation", None)
        use_incremental = should_use_incremental_allocation(incremental_req)

        # Allow caller to pass output_path as a directory ("--output-dir")
        output_path = self._normalize_flow_output_path(request, output_path)
        
        # Clear old flow cache if switching to a different video
        self._clear_flow_cache_for_new_video(video_path)
        
        # Validate downsample factor
        downsample_factor = max(1, int(downsample_factor))
        DS = downsample_factor
        
        # Cache key includes downsample factor
        cache_key = f"{video_path}_dis_ds{downsample_factor}"
        print(f"[DIS FLOW] Cache key: {cache_key}")
        print(f"[DIS FLOW] force_recompute (after parsing): {force_recompute}")
        
        # Check in-memory cache first
        cache_entry = self.flow_cache.get(cache_key)
        print(f"[DIS FLOW] Memory cache entry exists: {cache_entry is not None}")
        
        if cache_entry and not force_recompute:
            cached_method = cache_entry.get("metadata", {}).get("method", None)
            print(f"[DIS FLOW] Using MEMORY CACHE (method={cached_method})")
            if cached_method == "dis":
                print(f"Using cached DIS flow from memory for {video_path}")
                flows = cache_entry["flows_np"]
                
                response = {
                    "status": "ok",
                    "message": "DIS optical flow loaded from cache",
                    "shape": list(flows.shape),
                    "method": "dis",
                    "downsample_factor": downsample_factor,
                }
                if video_path in self.video_metadata:
                    metadata = self.video_metadata[video_path]
                    response["metadata"] = {
                        "original_shape": list(metadata['original_shape']) if metadata.get('original_shape') else None,
                        "resized_shape": list(metadata['resized_shape']) if metadata.get('resized_shape') else None,
                        "is_avi": metadata.get('is_avi', False),
                        "method": "dis",
                        "downsample_factor": downsample_factor,
                    }
                return response
        else:
            if force_recompute:
                print(f"[DIS FLOW] Skipping memory cache (force_recompute=True)")
            else:
                print(f"[DIS FLOW] No memory cache entry found")
        
        # Check disk cache
        print(f"[DIS FLOW] Checking disk cache: output_path={output_path}, force_recompute={force_recompute}")
        if output_path and not force_recompute:
            disk_path = self._find_existing_flow_file(
                output_path, 'dis', target_width=target_width, target_height=target_height,
                downsample_factor=downsample_factor)
            if disk_path and os.path.exists(disk_path):
                print(f"Loading DIS flow from disk cache: {disk_path}")
                try:
                    data = np.load(disk_path, allow_pickle=False)
                    # Memory-efficient loading with automatic float16 compression
                    flows = load_flows_memory_efficient(data['flows'])
                    print(f"  Loaded flow array shape: {flows.shape}, ndim: {flows.ndim}")
                    
                    # Validate flow shape - must be 4D (T-1, H, W, 2)
                    if flows.ndim != 4 or flows.shape[-1] != 2:
                        print(f"  WARNING: Invalid DIS flow shape in {disk_path}, recomputing...")
                        # Don't use this file, proceed to computation
                    else:
                        disk_method = str(data.get('method', 'dis')) if 'method' in data.files else 'dis'
                        if disk_method == 'dis':
                            # CRITICAL: Verify downsample_factor matches before using cached file
                            cached_ds = int(data['downsample_factor']) if 'downsample_factor' in data.files else 2
                            if cached_ds != downsample_factor:
                                print(f"  ⚠ Downsample factor mismatch: cached={cached_ds}, requested={downsample_factor}")
                                print(f"  Recomputing DIS flow with correct downsample factor...")
                                # Don't use this file - fall through to computation
                            elif 'original_shape' in data:
                                metadata = {
                                    'original_shape': tuple(int(x) for x in data['original_shape']),
                                    'resized_shape': tuple(int(x) for x in data['resized_shape']),
                                    'is_avi': bool(data['is_avi']) if 'is_avi' in data.files else False,
                                    'method': 'dis',
                                    'downsample_factor': int(data['downsample_factor']) if 'downsample_factor' in data.files else downsample_factor,
                                }
                                # CRITICAL: Load flow scaling metadata for DIS
                                if 'flow_shape' in data.files:
                                    metadata['flow_shape'] = tuple(int(x) for x in data['flow_shape'])
                                if 'flow_to_input_scale' in data.files:
                                    metadata['flow_to_input_scale'] = float(data['flow_to_input_scale'])
                                    print(f"  Loaded flow_to_input_scale={metadata['flow_to_input_scale']}")
                                else:
                                    # Infer from downsample_factor if not saved explicitly
                                    ds = metadata.get('downsample_factor', 1)
                                    if ds > 1:
                                        metadata['flow_to_input_scale'] = float(ds)
                                        metadata['flow_shape'] = flows.shape[:-1]  # (T-1, H, W)
                                        print(f"  Inferred flow_to_input_scale={ds} from downsample_factor")
                            else:
                                metadata = {'method': 'dis', 'downsample_factor': downsample_factor}
                            
                            self.flow_cache[cache_key] = {
                                "flows_np": flows,
                                "timestamp": _pc(),
                                "metadata": metadata,
                            }
                            self.flow_cache[video_path] = self.flow_cache[cache_key]
                            self.video_metadata[video_path] = metadata
                            print(f"✓ DIS flow loaded from disk ({flows.shape})")
                            
                            return {
                                "status": "ok",
                                "message": "DIS optical flow loaded from cache",
                                "shape": list(flows.shape),
                                "method": "dis",
                                "downsample_factor": metadata.get('downsample_factor', downsample_factor),
                        }
                except Exception as e:
                    print(f"Failed to load DIS flow from disk: {e}")
        
        # Compute DIS flow
        print(f"Computing DIS optical flow for {video_path}")
        print(f"Downsample factor: {downsample_factor}")
        if target_width and target_height:
            print(f"Using target resolution: {target_width}x{target_height}")
        start_time = _pc()
        
        video_path_str = str(video_path)
        is_avi = video_path_str.lower().endswith('.avi')
        
        # Determine if resizing is requested
        should_resize = (target_width is not None and target_width > 0 and 
                         target_height is not None and target_height > 0)
        
        # Variable for zarr store cleanup
        tiff_store = None
        
        # Use memory-efficient frame-by-frame loading for large TIFF files
        # This avoids loading the entire video into memory at once
        if is_avi:
            # For AVI, use mediapy which handles large files better
            print(f"Loading AVI with mediapy: {video_path_str}")
            video = media.read_video(video_path_str)
            orig_shape = video.shape[:3]
            T, H, W = orig_shape[0], orig_shape[1], orig_shape[2]
            
            if should_resize:
                target_H, target_W = target_height, target_width
            else:
                target_H, target_W = H, W
            
            resized_shape = (T, target_H, target_W)
            
            def get_frame_gray(t):
                """Get frame t as grayscale uint8."""
                frame = video[t]
                if frame.ndim == 3:
                    frame = np.mean(frame, axis=-1)
                if should_resize and (H != target_H or W != target_W):
                    frame = cv2.resize(frame.astype(np.float32), (target_W, target_H))
                return cv2.normalize(frame, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        else:
            # For TIFF, detect format and use appropriate loading strategy
            print(f"Loading TIFF with memory-efficient access: {video_path_str}")
            with tifffile.TiffFile(video_path_str) as tif:
                # Check pages count
                num_pages = len(tif.pages)
                print(f"TIFF has {num_pages} pages")
                
                # Check series shape (handles ImageJ hyperstack format)
                series_frames = 1
                if tif.series and len(tif.series) > 0:
                    series = tif.series[0]
                    shape = series.shape
                    print(f"TIFF series shape: {shape}")
                    if len(shape) >= 3:
                        series_frames = shape[0]
                        H = shape[1]
                        W = shape[2]
                    else:
                        H, W = shape[0], shape[1]
                else:
                    first_page = tif.pages[0].asarray()
                    H, W = first_page.shape[:2]
                
                # Determine frame count and access method
                # ImageJ TIFFs: series_frames > num_pages (e.g., 191 frames in 1 page)
                # Regular TIFFs: num_pages == series_frames (each page is a frame)
                if series_frames > num_pages:
                    # ImageJ-style: single page with multiple images
                    T = series_frames
                    use_pages = False
                    print(f"ImageJ-style TIFF: {T} frames in {num_pages} page(s)")
                else:
                    # Regular multi-page TIFF
                    T = num_pages
                    use_pages = True
                    print(f"Multi-page TIFF: {T} pages")
                
                orig_shape = (T, H, W)
                print(f"TIFF detected: {T} frames, {H}x{W}")
                
                # Verify we have multiple frames
                if T <= 1:
                    print(f"WARNING: TIFF appears to have only {T} frame(s)")
            
            if should_resize:
                target_H, target_W = target_height, target_width
            else:
                target_H, target_W = H, W
            
            resized_shape = (T, target_H, target_W)
            
            # Store original H, W for the closure
            orig_H, orig_W = H, W
            
            # For ImageJ-style TIFFs, we need different loading strategy
            if not use_pages:
                # ImageJ TIFF: frames stored in single page, need to load via zarr or full load
                loaded_via_zarr = False
                
                if HAS_ZARR:
                    print(f"Trying zarr for memory-mapped access...")
                    try:
                        tiff_store = tifffile.imread(video_path_str, aszarr=True)
                        tiff_zarr = zarr.open(tiff_store, mode='r')
                        print(f"Zarr array shape: {tiff_zarr.shape}")
                        loaded_via_zarr = True
                        
                        def get_frame_gray(t):
                            """Get frame t as grayscale uint8 from zarr store."""
                            frame = np.array(tiff_zarr[t]).astype(np.float32)
                            if should_resize and (orig_H != target_H or orig_W != target_W):
                                frame = cv2.resize(frame, (target_W, target_H))
                            return cv2.normalize(frame, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
                    except Exception as e:
                        print(f"Zarr failed: {e}")
                        loaded_via_zarr = False
                
                if not loaded_via_zarr:
                    # Fallback: load entire video into memory
                    # WARNING: This can use a lot of memory for large videos!
                    video_mem_estimate = T * orig_H * orig_W * 4 / 1e9  # float32
                    print(f"WARNING: Loading full ImageJ TIFF into memory (~{video_mem_estimate:.1f} GB)...")
                    print(f"  Consider using target_width/target_height to reduce resolution")
                    video_data = tifffile.imread(video_path_str)
                    print(f"Loaded video data shape: {video_data.shape}")
                    
                    def get_frame_gray(t):
                        """Get frame t as grayscale uint8 from memory."""
                        frame = video_data[t].astype(np.float32)
                        if should_resize and (orig_H != target_H or orig_W != target_W):
                            frame = cv2.resize(frame, (target_W, target_H))
                        return cv2.normalize(frame, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
            else:
                def get_frame_gray(t):
                    """Get frame t as grayscale uint8 from pages."""
                    with tifffile.TiffFile(video_path_str) as tif:
                        frame = tif.pages[t].asarray().astype(np.float32)
                    if should_resize and (orig_H != target_H or orig_W != target_W):
                        frame = cv2.resize(frame, (target_W, target_H))
                    return cv2.normalize(frame, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        
        # Create DIS optical flow object with MEDIUM preset
        dis = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_MEDIUM)
        
        # Compute optical flow for each frame pair
        # H, W are the frame dimensions (after any target resize from Java UI)
        H, W = target_H, target_W
        
        # CRITICAL MEMORY OPTIMIZATION: 
        # Store flow at DOWNSAMPLED resolution (H//DS, W//DS), NOT upscaled to full resolution.
        # This reduces memory by DS^2 factor (e.g., DS=2 -> 4x less memory).
        # Flow values are scaled by DS so they represent displacement in the input coordinate space.
        flow_H = H // DS
        flow_W = W // DS
        
        num_flow_frames = T - 1
        expected_mem_gb = num_flow_frames * flow_H * flow_W * 2 * 4 / 1e9
        original_mem_gb = num_flow_frames * H * W * 2 * 4 / 1e9  # What it would be without DS
        single_flow_mb = flow_H * flow_W * 2 * 4 / 1e6  # Memory for one flow frame
        
        # Report system memory info
        total_ram_gb = get_total_system_memory_gb()
        
        print(f"\n{'='*60}")
        print(f"DIS OPTICAL FLOW - MEMORY ESTIMATION")
        print(f"  Video: {T} frames at {W}x{H}")
        print(f"  Downsample factor: {DS}x")
        print(f"  Flow resolution: {flow_W}x{flow_H} (stored)")
        print(f"  Memory required: {expected_mem_gb:.2f} GB")
        if DS > 1:
            print(f"  Memory saved: {original_mem_gb:.2f} GB -> {expected_mem_gb:.2f} GB ({DS*DS}x reduction)")
        print(f"  System RAM: {total_ram_gb:.1f} GB")
        print(f"  Allocation mode: {'INCREMENTAL (swap-friendly)' if use_incremental else 'PRE-ALLOCATED (faster)'}")
        print(f"{'='*60}\n")
        
        # Check if we have enough memory before allocating (only for pre-allocated mode)
        if HAS_PSUTIL and not use_incremental:
            available_mem_gb = psutil.virtual_memory().available / 1e9
            print(f"Available system memory: {available_mem_gb:.2f} GB")
            
            if expected_mem_gb > available_mem_gb * 0.8:  # Leave 20% headroom
                # Instead of failing, suggest enabling incremental mode
                print(f"  ⚠️  Memory may be tight - consider enabling incremental allocation")
                print(f"  Switching to incremental allocation mode automatically...")
                use_incremental = True
        elif not HAS_PSUTIL:
            print("  (psutil not available - skipping memory check)")
        
        # Choose allocation strategy based on mode
        if use_incremental:
            # INCREMENTAL ALLOCATION (like jellyfish_tracker)
            # Allocate each flow array separately - allows OS to swap older arrays to disk
            print(f"Using INCREMENTAL allocation: {num_flow_frames} arrays of {single_flow_mb:.1f} MB each")
            print(f"  This allows the OS to swap older arrays to disk if needed.")
            print(f"  May run slower but prevents OOM on low-memory systems.")
            
            flow_maps = [None] * num_flow_frames  # List of None pointers initially
            
        else:
            # PRE-ALLOCATED (faster, but requires contiguous memory)
            print(f"Pre-allocating flow array: {num_flow_frames} x {flow_H} x {flow_W} x 2 ({expected_mem_gb:.2f} GB)")
            
            try:
                flows_np = np.zeros((num_flow_frames, flow_H, flow_W, 2), dtype=np.float32)
            except MemoryError:
                print(f"  ⚠️  Pre-allocation failed, falling back to incremental mode...")
                use_incremental = True
                flow_maps = [None] * num_flow_frames
        
        print(f"\n{'*'*60}")
        print(f"[DIS FLOW] COMPUTING FRESH (no cache used)")
        print(f"{'*'*60}")
        print(f"Computing DIS flow for {T-1} frame pairs...")
        print(f"  Downsample factor: {DS}")
        print(f"  Input frame size: {W}x{H}")
        print(f"  Computing at: {W//DS}x{H//DS}")
        print(f"  Storing at: {flow_W}x{flow_H}")
        
        # Pre-load first frame
        prev_frame = get_frame_gray(0)
        
        for t in range(T - 1):
            # Allow responsive cancellation during long per-frame loops
            self._check_cancelled("DIS flow computation")
            # Get current and next frame
            curr_frame = get_frame_gray(t + 1)
            
            # Downsample frames for DIS computation
            if DS > 1:
                small1 = cv2.resize(prev_frame, (W // DS, H // DS))
                small2 = cv2.resize(curr_frame, (W // DS, H // DS))
            else:
                small1, small2 = prev_frame, curr_frame
            
            # Compute flow at reduced resolution
            flow_small = dis.calc(small1, small2, None)
            
            # Scale flow values by DS so they represent displacement in input coordinates
            if use_incremental:
                # INCREMENTAL MODE: allocate new array for each frame
                flow = np.zeros((flow_H, flow_W, 2), dtype=np.float32)
                flow[:, :, 0] = flow_small[:, :, 0] * DS
                flow[:, :, 1] = flow_small[:, :, 1] * DS
                flow_maps[t] = flow
            else:
                # PRE-ALLOCATED MODE: write directly to pre-allocated array
                flows_np[t, :, :, 0] = flow_small[:, :, 0] * DS
                flows_np[t, :, :, 1] = flow_small[:, :, 1] * DS
            
            prev_frame = curr_frame  # Reuse for next iteration
            
            if (t + 1) % 50 == 0 or t == T - 2:
                print(f"  DIS flow: {t + 1}/{T - 1} frames processed")
        
        # Convert incremental flow_maps list to numpy array for compatibility
        if use_incremental:
            print(f"  Converting {num_flow_frames} flow maps to array...")
            flows_np = np.stack(flow_maps, axis=0)
            # Free the list to reduce memory during compression
            del flow_maps
            import gc
            gc.collect()
        
        print(f"  DIS flow array shape: {flows_np.shape}")
        
        if flows_np.ndim != 4:
            raise RuntimeError(f"Invalid flow shape: {flows_np.shape}, expected 4D array (T-1, H, W, 2)")
        
        elapsed = _pc() - start_time
        print(f"✓ DIS flow computed in {elapsed:.2f}s")
        
        # Update metadata with actual flow resolution
        # This is CRITICAL for tracking to know how to scale coordinates
        metadata = {
            'original_shape': orig_shape,           # Original video dimensions (T, H, W)
            'resized_shape': resized_shape,         # After Java UI compression (T, target_H, target_W)
            'flow_shape': (num_flow_frames, flow_H, flow_W, 2),  # Actual flow array dimensions
            'is_avi': is_avi,
            'method': 'dis',
            'downsample_factor': DS,
            # Scale factor to convert flow coords to input coords: multiply flow coords by this
            'flow_to_input_scale': DS,
        }
        self.video_metadata[video_path] = metadata
        
        # Clean up zarr store if used
        if tiff_store is not None:
            try:
                tiff_store.close()
            except:
                pass
        
        # Compress to float16 if beneficial (saves ~50% memory for large flows)
        flows_np = wrap_flows_memory_efficient(flows_np, "DIS")
        
        # Cache in memory
        self.flow_cache[cache_key] = {
            "flows_np": flows_np,
            "timestamp": _pc(),
            "metadata": metadata,
        }
        self.flow_cache[video_path] = self.flow_cache[cache_key]
        
        # Save to disk if requested and save_to_disk is enabled
        if output_path and save_to_disk:
            actual_output_path = self._build_flow_filename(
                output_path, 'dis',
                resized_shape=resized_shape,
                original_shape=orig_shape,
                downsample_factor=DS,
            )
            # Get raw array for saving (handles MemoryEfficientFlowArray)
            flows_to_save = flows_np.get_raw() if hasattr(flows_np, 'get_raw') else flows_np
            np.savez(
                actual_output_path,
                flows=flows_to_save,
                original_shape=metadata['original_shape'],
                resized_shape=metadata['resized_shape'],
                flow_shape=metadata['flow_shape'],
                is_avi=is_avi,
                method='dis',
                downsample_factor=DS,
                flow_to_input_scale=DS,
            )
            print(f"✓ DIS flow saved to {actual_output_path}")
        
        return {
            "status": "ok",
            "message": "DIS optical flow computed",
            "shape": list(flows_np.shape),
            "flow_resolution": f"{flow_W}x{flow_H}",
            "input_resolution": f"{W}x{H}",
            "method": "dis",
            "downsample_factor": DS,
            "memory_gb": f"{expected_mem_gb:.2f}",
            "elapsed_time": f"{elapsed:.2f}s",
        }
    
    def _propagate_with_flows(self, flows, seed_x, seed_y, seed_frame=0, flow_scale=1, input_dims=None):
        """Propagate track bidirectionally from seed point at seed_frame.
        
        Args:
            flows: (T-1, H, W, 2) optical flow array, flow[t] is t -> t+1
                   For DIS with downsample, H and W are the reduced dimensions.
            seed_x: X coordinate of seed point (in input/display coordinates)
            seed_y: Y coordinate of seed point (in input/display coordinates)
            seed_frame: Frame index where seed is placed (0-indexed)
            flow_scale: Scale factor for converting input coords to flow coords.
                        For DIS with DS=2, flow_scale=2 means flow is at half resolution.
                        Input coords / flow_scale = flow coords for lookup.
            input_dims: (H, W) of the input/display coordinate space.
                        If None, uses flow dimensions * flow_scale.
            
        Returns:
            track: (T, 2) int32 array of (x, y) positions in INPUT coordinates
        """
        Tm1, flow_H, flow_W, _ = flows.shape
        T = Tm1 + 1  # flows is (T-1, H, W, 2)
        
        # Determine input dimensions (the coordinate space for output)
        if input_dims is not None:
            input_H, input_W = input_dims
        else:
            input_H = flow_H * flow_scale
            input_W = flow_W * flow_scale
        
        # Validate seed_frame
        seed_frame = int(np.clip(seed_frame, 0, T - 1))
        
        track = np.zeros((T, 2), dtype=np.int32)
        
        # Place seed point (in input coordinates)
        x_i = int(np.clip(round(seed_x), 0, input_W - 1))
        y_i = int(np.clip(round(seed_y), 0, input_H - 1))
        track[seed_frame] = (x_i, y_i)
        
        # Forward propagation: seed_frame -> T-1
        # Position is tracked in INPUT coordinates
        pos = np.array([seed_x, seed_y], dtype=np.float32)
        for t in range(seed_frame, T):
            self._check_cancelled("Track propagation")
            # Record position in input coordinates
            x_i = int(np.clip(round(pos[0]), 0, input_W - 1))
            y_i = int(np.clip(round(pos[1]), 0, input_H - 1))
            track[t] = (x_i, y_i)
            
            if t < T - 1:
                # Convert input coords to flow coords for lookup
                flow_x = int(np.clip(round(pos[0] / flow_scale), 0, flow_W - 1))
                flow_y = int(np.clip(round(pos[1] / flow_scale), 0, flow_H - 1))
                
                # Flow values are already scaled to input coordinates
                dx, dy = flows[t, flow_y, flow_x]
                pos[0] += float(dx)
                pos[1] += float(dy)
        
        # Backward propagation: seed_frame -> 0
        # Use iterative flow inversion: flows[t] maps FROM frame t TO frame t+1
        # To go backward, we need to find pos_t where pos_t + flow[t](pos_t) = pos_{t+1}
        if seed_frame > 0:
            pos = np.array([seed_x, seed_y], dtype=np.float32)
            for t in range(seed_frame - 1, -1, -1):
                self._check_cancelled("Track propagation")
                # Iterative refinement to invert the flow field
                x_est = pos[0]
                y_est = pos[1]
                
                for iteration in range(3):  # 3 iterations usually sufficient
                    # Keep cancellation responsive even in tight inner loops
                    if iteration == 0:
                        self._check_cancelled("Track propagation")
                    # Look up flow at estimated position in frame t
                    flow_x = int(np.clip(round(x_est / flow_scale), 0, flow_W - 1))
                    flow_y = int(np.clip(round(y_est / flow_scale), 0, flow_H - 1))
                    dx, dy = flows[t, flow_y, flow_x]
                    
                    # Update estimate: pos_t = pos_{t+1} - flow[t](pos_t_estimate)
                    x_est = pos[0] - float(dx)
                    y_est = pos[1] - float(dy)
                
                # Update position
                pos[0] = x_est
                pos[1] = y_est
                
                # Record position in input coordinates
                x_i = int(np.clip(round(pos[0]), 0, input_W - 1))
                y_i = int(np.clip(round(pos[1]), 0, input_H - 1))
                track[t] = (x_i, y_i)
        
        return track

    def _propagate_with_flows_blob(self, flows, video_frames, seed_x, seed_y, seed_frame=0, 
                                    flow_scale=1, input_dims=None, search_radius=15, blob_radius=5.0):
        """Propagate track bidirectionally using optical flow + blob detection refinement.
        
        This method combines optical flow prediction with blob detection for more accurate
        tracking, similar to the AutoTrack function in jellyfish_tracker7_1.py.
        
        Args:
            flows: (T-1, H, W, 2) optical flow array, flow[t] is t -> t+1
            video_frames: (T, H, W) or (T, H, W, 3) video frames for blob detection
            seed_x: X coordinate of seed point (in input/display coordinates)
            seed_y: Y coordinate of seed point (in input/display coordinates)
            seed_frame: Frame index where seed is placed (0-indexed)
            flow_scale: Scale factor for converting input coords to flow coords.
            input_dims: (H, W) of the input/display coordinate space.
            search_radius: Radius around flow-predicted position to search for blobs
            blob_radius: Expected radius of blob for detection (in pixels)
            
        Returns:
            track: (T, 2) int32 array of (x, y) positions in INPUT coordinates
        """
        if not HAS_SKIMAGE:
            print("WARNING: skimage not available, falling back to pure flow propagation")
            return self._propagate_with_flows(flows, seed_x, seed_y, seed_frame, flow_scale, input_dims)
        
        Tm1, flow_H, flow_W, _ = flows.shape
        T = Tm1 + 1
        
        # Determine input dimensions
        if input_dims is not None:
            input_H, input_W = input_dims
        else:
            input_H = flow_H * flow_scale
            input_W = flow_W * flow_scale
        
        # Validate seed_frame
        seed_frame = int(np.clip(seed_frame, 0, T - 1))
        
        # Use video frames directly - convert to grayscale/float32 per-patch for efficiency
        # (Avoids copying entire video array which can be 10+ GB)
        frames = video_frames
        is_rgb = (video_frames.ndim == 4)
        
        # Ensure frames match expected dimensions
        if is_rgb:
            _, frame_H, frame_W, _ = frames.shape
        else:
            _, frame_H, frame_W = frames.shape
        
        # Blob detection parameters (similar to jellyfish_tracker)
        sigma_center = blob_radius / np.sqrt(2.0)
        min_sigma = max(0.5, sigma_center * 0.7)
        max_sigma = sigma_center * 1.3
        
        track = np.zeros((T, 2), dtype=np.int32)
        
        # Place seed point
        x_i = int(np.clip(round(seed_x), 0, input_W - 1))
        y_i = int(np.clip(round(seed_y), 0, input_H - 1))
        track[seed_frame] = (x_i, y_i)
        
        def find_blob_position(t_next, x_pred, y_pred):
            """Search for blob near predicted position."""
            r = int(search_radius)
            y_min = max(0, int(y_pred) - r)
            y_max = min(frame_H, int(y_pred) + r + 1)
            x_min = max(0, int(x_pred) - r)
            x_max = min(frame_W, int(x_pred) + r + 1)
            
            # Extract patch and convert to grayscale float32 (efficient per-patch conversion)
            if is_rgb:
                patch_raw = frames[t_next, y_min:y_max, x_min:x_max, :3]
                patch = np.mean(patch_raw, axis=-1).astype(np.float32)
            else:
                patch = frames[t_next, y_min:y_max, x_min:x_max].astype(np.float32)
            
            if patch.size == 0:
                return x_pred, y_pred
            
            # Normalize patch for blob detection
            pmin, pmax = float(patch.min()), float(patch.max())
            patch_norm = (patch - pmin) / (pmax - pmin) if pmax > pmin else np.zeros_like(patch)
            
            chosen = None
            y_local = y_pred - y_min
            x_local = x_pred - x_min
            
            # Try blob_log detection first
            try:
                blobs = blob_log(
                    patch_norm,
                    min_sigma=min_sigma,
                    max_sigma=max_sigma,
                    num_sigma=5,
                    threshold=0.02,
                )
                
                if len(blobs) > 0:
                    # Find closest blob to predicted position
                    dists = [np.hypot(by - y_local, bx - x_local) for (by, bx, _) in blobs]
                    by, bx, _ = blobs[int(np.argmin(dists))]
                    chosen = (y_min + by, x_min + bx)
            except Exception as e:
                print(f"  blob_log error: {e}")
            
            # Fallback to region-based detection if no blob found
            if chosen is None:
                try:
                    thresh = np.percentile(patch_norm, 70)
                    mask = patch_norm > thresh
                    if mask.any():
                        labels = measure.label(mask)
                        regions = measure.regionprops(labels)
                        candidates = []
                        for reg in regions:
                            eqr = reg.equivalent_diameter / 2
                            if abs(eqr - blob_radius) <= blob_radius * 0.5:
                                cy, cx = reg.centroid
                                candidates.append((np.hypot(cy - y_local, cx - x_local), cy, cx))
                        if candidates:
                            _, cy, cx = min(candidates)
                            chosen = (y_min + cy, x_min + cx)
                except Exception as e:
                    print(f"  region detection error: {e}")
            
            if chosen is not None:
                return chosen[1], chosen[0]  # Return (x, y)
            else:
                return x_pred, y_pred
        
        # Forward propagation: seed_frame -> T-1
        pos = np.array([seed_x, seed_y], dtype=np.float32)
        for t in range(seed_frame, T):
            # Record position
            x_i = int(np.clip(round(pos[0]), 0, input_W - 1))
            y_i = int(np.clip(round(pos[1]), 0, input_H - 1))
            track[t] = (x_i, y_i)
            
            if t < T - 1:
                # Get flow prediction
                flow_x = int(np.clip(round(pos[0] / flow_scale), 0, flow_W - 1))
                flow_y = int(np.clip(round(pos[1] / flow_scale), 0, flow_H - 1))
                dx, dy = flows[t, flow_y, flow_x]
                
                # Flow-predicted position
                x_pred = np.clip(pos[0] + float(dx), 0, input_W - 1)
                y_pred = np.clip(pos[1] + float(dy), 0, input_H - 1)
                
                # Refine with blob detection
                x_refined, y_refined = find_blob_position(t + 1, x_pred, y_pred)
                pos[0] = float(x_refined)
                pos[1] = float(y_refined)
        
        # Backward propagation: seed_frame -> 0
        # Note: flows[t] gives displacement from frame t to t+1
        # To go backward (from t+1 to t), we need to invert the flow field
        # This is an iterative process since flow[t] is defined on frame t's grid
        if seed_frame > 0:
            pos = np.array([seed_x, seed_y], dtype=np.float32)
            for t in range(seed_frame - 1, -1, -1):
                # We're at position pos in frame t+1, want to find position in frame t
                # flows[t] maps FROM frame t TO frame t+1: pos_t + flow[t] = pos_{t+1}
                # So: pos_t = pos_{t+1} - flow[t](pos_t)  <- this requires knowing pos_t!
                
                # Iterative refinement to solve for pos_t:
                # Start with initial guess: pos_t ≈ pos_{t+1}
                x_est = pos[0]
                y_est = pos[1]
                
                for iteration in range(3):  # 3 iterations usually sufficient
                    # Look up flow at estimated position in frame t
                    flow_x = int(np.clip(round(x_est / flow_scale), 0, flow_W - 1))
                    flow_y = int(np.clip(round(y_est / flow_scale), 0, flow_H - 1))
                    dx, dy = flows[t, flow_y, flow_x]
                    
                    # Update estimate: pos_t = pos_{t+1} - flow[t](pos_t_estimate)
                    x_est = np.clip(pos[0] - float(dx), 0, input_W - 1)
                    y_est = np.clip(pos[1] - float(dy), 0, input_H - 1)
                
                # Use the refined estimate as flow-predicted position
                x_pred = x_est
                y_pred = y_est
                
                # Refine with blob detection in frame t
                x_refined, y_refined = find_blob_position(t, x_pred, y_pred)
                pos[0] = float(x_refined)
                pos[1] = float(y_refined)
                
                # Record position
                x_i = int(np.clip(round(pos[0]), 0, input_W - 1))
                y_i = int(np.clip(round(pos[1]), 0, input_H - 1))
                track[t] = (x_i, y_i)
        
        return track

    def _propagate_with_flows_blob_memmap(self, flows, video_accessor, seed_x, seed_y, seed_frame=0, 
                                           flow_scale=1, input_dims=None, search_radius=15, blob_radius=5.0):
        """Propagate track using optical flow + blob detection with memory-mapped video access.
        
        This version uses MemoryMappedVideoAccessor to read only the small patches needed,
        avoiding loading the entire video into RAM. RAM usage is minimal (~few MB) regardless
        of video size.
        
        Args:
            flows: (T-1, H, W, 2) optical flow array
            video_accessor: MemoryMappedVideoAccessor instance
            seed_x, seed_y: Seed point coordinates
            seed_frame: Frame index where seed is placed (0-indexed)
            flow_scale: Scale factor for flow coordinates
            input_dims: (H, W) of input coordinate space
            search_radius: Search radius for blob detection
            blob_radius: Expected blob radius
            
        Returns:
            track: (T, 2) int32 array of (x, y) positions
        """
        if not HAS_SKIMAGE:
            print("WARNING: skimage not available, falling back to pure flow propagation")
            return self._propagate_with_flows(flows, seed_x, seed_y, seed_frame, flow_scale, input_dims)
        
        Tm1, flow_H, flow_W, _ = flows.shape
        T = Tm1 + 1
        
        # Determine input dimensions
        if input_dims is not None:
            input_H, input_W = input_dims
        else:
            input_H = flow_H * flow_scale
            input_W = flow_W * flow_scale
        
        # Get frame dimensions from video accessor
        frame_H, frame_W = video_accessor.shape[1], video_accessor.shape[2]
        
        # Validate seed_frame
        seed_frame = int(np.clip(seed_frame, 0, T - 1))
        
        # Blob detection parameters
        sigma_center = blob_radius / np.sqrt(2.0)
        min_sigma = max(0.5, sigma_center * 0.7)
        max_sigma = sigma_center * 1.3
        
        track = np.zeros((T, 2), dtype=np.int32)
        
        # Place seed point
        x_i = int(np.clip(round(seed_x), 0, input_W - 1))
        y_i = int(np.clip(round(seed_y), 0, input_H - 1))
        track[seed_frame] = (x_i, y_i)
        
        def find_blob_position(t_next, x_pred, y_pred):
            """Search for blob near predicted position using memory-mapped access."""
            self._check_cancelled("Track propagation")
            r = int(search_radius)
            y_min = max(0, int(y_pred) - r)
            y_max = min(frame_H, int(y_pred) + r + 1)
            x_min = max(0, int(x_pred) - r)
            x_max = min(frame_W, int(x_pred) + r + 1)
            
            # Get patch using memory-mapped accessor (reads from disk, not RAM)
            patch = video_accessor.get_patch(t_next, y_min, y_max, x_min, x_max)
            
            if patch.size == 0:
                return x_pred, y_pred
            
            # Normalize patch for blob detection
            pmin, pmax = float(patch.min()), float(patch.max())
            patch_norm = (patch - pmin) / (pmax - pmin) if pmax > pmin else np.zeros_like(patch)
            
            chosen = None
            y_local = y_pred - y_min
            x_local = x_pred - x_min
            
            # Try blob_log detection first
            try:
                blobs = blob_log(
                    patch_norm,
                    min_sigma=min_sigma,
                    max_sigma=max_sigma,
                    num_sigma=5,
                    threshold=0.02,
                )
                
                if len(blobs) > 0:
                    dists = [np.hypot(by - y_local, bx - x_local) for (by, bx, _) in blobs]
                    by, bx, _ = blobs[int(np.argmin(dists))]
                    chosen = (y_min + by, x_min + bx)
            except Exception as e:
                print(f"  blob_log error: {e}")
            
            # Fallback to region-based detection if no blob found
            if chosen is None:
                try:
                    thresh = np.percentile(patch_norm, 70)
                    mask = patch_norm > thresh
                    if mask.any():
                        labels = measure.label(mask)
                        regions = measure.regionprops(labels)
                        candidates = []
                        for reg in regions:
                            eqr = reg.equivalent_diameter / 2
                            if abs(eqr - blob_radius) <= blob_radius * 0.5:
                                cy, cx = reg.centroid
                                candidates.append((np.hypot(cy - y_local, cx - x_local), cy, cx))
                        if candidates:
                            _, cy, cx = min(candidates)
                            chosen = (y_min + cy, x_min + cx)
                except Exception as e:
                    print(f"  region detection error: {e}")
            
            if chosen is not None:
                return chosen[1], chosen[0]  # Return (x, y)
            else:
                return x_pred, y_pred
        
        # Forward propagation: seed_frame -> T-1
        pos = np.array([seed_x, seed_y], dtype=np.float32)
        for t in range(seed_frame, T):
            x_i = int(np.clip(round(pos[0]), 0, input_W - 1))
            y_i = int(np.clip(round(pos[1]), 0, input_H - 1))
            track[t] = (x_i, y_i)
            
            if t < T - 1:
                flow_x = int(np.clip(round(pos[0] / flow_scale), 0, flow_W - 1))
                flow_y = int(np.clip(round(pos[1] / flow_scale), 0, flow_H - 1))
                dx, dy = flows[t, flow_y, flow_x]
                
                x_pred = np.clip(pos[0] + float(dx), 0, input_W - 1)
                y_pred = np.clip(pos[1] + float(dy), 0, input_H - 1)
                
                x_refined, y_refined = find_blob_position(t + 1, x_pred, y_pred)
                pos[0] = float(x_refined)
                pos[1] = float(y_refined)
        
        # Backward propagation: seed_frame -> 0
        if seed_frame > 0:
            pos = np.array([seed_x, seed_y], dtype=np.float32)
            for t in range(seed_frame - 1, -1, -1):
                # Iterative flow inversion
                x_est = pos[0]
                y_est = pos[1]
                
                for iteration in range(3):
                    flow_x = int(np.clip(round(x_est / flow_scale), 0, flow_W - 1))
                    flow_y = int(np.clip(round(y_est / flow_scale), 0, flow_H - 1))
                    dx, dy = flows[t, flow_y, flow_x]
                    
                    x_est = np.clip(pos[0] - float(dx), 0, input_W - 1)
                    y_est = np.clip(pos[1] - float(dy), 0, input_H - 1)
                
                x_pred = x_est
                y_pred = y_est
                
                x_refined, y_refined = find_blob_position(t, x_pred, y_pred)
                pos[0] = float(x_refined)
                pos[1] = float(y_refined)
                
                x_i = int(np.clip(round(pos[0]), 0, input_W - 1))
                y_i = int(np.clip(round(pos[1]), 0, input_H - 1))
                track[t] = (x_i, y_i)
        
        return track

    def _propagate_track(self, request):
        """Propagate track bidirectionally from seed point.
        
        Supports two propagation modes:
        1. Pure optical flow (default): Fast, follows flow vectors directly
        2. Blob detection mode: Combines flow with blob detection for refined tracking
        """
        # Check for cancellation before starting
        self._check_cancelled("Track propagation")
        
        video_path = request["video_path"]
        seed_x = request["seed_x"]
        seed_y = request["seed_y"]
        seed_frame = request.get("seed_frame", 0)  # 0-indexed frame where seed is placed
        output_path = request.get("output_path")
        video_name = request.get("video_name")  # Optional video name override
        
        # If output_path is a directory, construct the full file path
        if output_path and os.path.isdir(output_path):
            if video_name:
                base_name = video_name
            else:
                base_name = os.path.splitext(os.path.basename(video_path))[0]
            output_path = os.path.join(output_path, f"{base_name}_annotations.json")
        
        # Blob detection parameters (new feature)
        use_blob_detection = request.get("use_blob_detection", False)
        blob_search_radius = request.get("blob_search_radius", 15)
        blob_radius = request.get("blob_radius", 5.0)
        cache_video = request.get("cache_video", False)  # Whether to cache video in memory
        
        # Get or compute flows
        entry = self.flow_cache.get(video_path)
        if not entry:
            self._compute_flow(request)
            entry = self.flow_cache.get(video_path)
        if not entry:
            raise RuntimeError("Optical flow unavailable after recompute")

        flows = entry["flows_np"]
        metadata = entry.get("metadata", {})
        
        # Get flow scale for DIS (flow stored at reduced resolution)
        flow_scale = metadata.get("flow_to_input_scale", 1)
        method = metadata.get("method", "raft")
        
        # Get input dimensions from metadata
        resized_shape = metadata.get("resized_shape")
        input_dims = None
        if resized_shape and len(resized_shape) >= 3:
            input_dims = (resized_shape[1], resized_shape[2])  # (H, W)
        
        print(f"Propagating track from ({seed_x}, {seed_y}) at frame {seed_frame}")
        print(f"  Method: {method}, flow_scale: {flow_scale}")
        print(f"  Blob detection: {use_blob_detection}")
        if use_blob_detection:
            print(f"  Blob params: search_radius={blob_search_radius}, blob_radius={blob_radius}")
        if input_dims:
            print(f"  Input dims: {input_dims[1]}x{input_dims[0]}")

        if use_blob_detection:
            # Use cached video accessor for blob detection (if caching enabled)
            # This avoids expensive re-initialization for AVI and mixed-memmappable TIFFs
            target_width = resized_shape[2] if resized_shape and len(resized_shape) >= 3 else None
            target_height = resized_shape[1] if resized_shape and len(resized_shape) >= 3 else None
            
            video_accessor = self._get_or_create_video_accessor(video_path, target_width, target_height, use_cache=cache_video)
            
            t_start = time.time()
            track = self._propagate_with_flows_blob_memmap(
                flows, video_accessor, seed_x, seed_y,
                seed_frame=seed_frame,
                flow_scale=flow_scale,
                input_dims=input_dims,
                search_radius=blob_search_radius,
                blob_radius=blob_radius
            )
            t_elapsed = time.time() - t_start
            print(f"  ✓ Blob propagation completed in {t_elapsed:.2f}s ({t_elapsed/len(track)*1000:.1f} ms/frame)")
            # Close accessor if caching is disabled
            if not cache_video:
                try:
                    video_accessor.close()
                except Exception:
                    pass
        else:
            track = self._propagate_with_flows(
                flows, seed_x, seed_y, 
                seed_frame=seed_frame,
                flow_scale=flow_scale,
                input_dims=input_dims
            )


        # Export to JSON with resolution metadata
        self._export_track_json(track, output_path, "Track1_RAFT_Baseline", video_path)
        
        return {
            "status": "ok",
            "message": "Track propagated",
            "output_path": output_path,
        }
    
    def _optimize_track(self, request):
        """Optimize track with anchors using flow-based segment blending.
        
        This method supports two correction methods:
        1. Full-Blend (default): Pure optical flow with forward/backward blending
        2. Blob-Assisted: Combines optical flow with blob detection for refinement
        
        The old DP-based corridor interpolation is still available via use_legacy_interpolation=True.
        """
        # Check for cancellation before starting
        self._check_cancelled("Track optimization")
        
        video_path = request["video_path"]
        anchors_json = request["anchors"]
        output_path = request.get("output_path")
        video_name = request.get("video_name")  # Optional video name override
        
        # If output_path is a directory, construct the full file path
        if output_path and os.path.isdir(output_path):
            if video_name:
                base_name = video_name
            else:
                base_name = os.path.splitext(os.path.basename(video_path))[0]
            output_path = os.path.join(output_path, f"{base_name}_annotations.json")
        
        # Option to use legacy DP-based interpolation (not recommended)
        use_legacy = request.get("use_legacy_interpolation", False)
        
        # Correction method: "full_blend" (default), "blob_assisted", or "corridor_dp"
        correction_method = request.get("correction_method", "full_blend")
        
        # Corridor DP parameters (only used for corridor_dp mode)
        corridor_width = self._normalize_corridor_width(request.get("corridor_width", None))
        
        # Blob detection parameters (only used for blob_assisted mode)
        blob_search_radius = request.get("blob_search_radius", 15)
        blob_radius = request.get("blob_radius", 5.0)
        cache_video = request.get("cache_video", False)  # Whether to cache video in memory
        
        # Linear interpolation threshold: if two anchors are within N frames, use linear interpolation
        # instead of optical flow. This helps avoid jitter from unreliable flow in short segments.
        # 0 = disabled (always use optical flow), typical values: 5-10 for noisy/jittery conditions
        linear_interp_threshold = request.get("linear_interp_threshold", 0)
        
        # Get or compute flows
        entry = self.flow_cache.get(video_path)
        if not entry:
            self._compute_flow(request)
            entry = self.flow_cache.get(video_path)
        if not entry:
            raise RuntimeError("Optical flow unavailable after recompute")

        flows = entry["flows_np"]
        metadata = entry.get("metadata", {})
        
        # Get flow dimensions
        Tm1, flow_H, flow_W, _ = flows.shape
        T = Tm1 + 1  # flows is (T-1, H, W, 2)
        
        # Get flow scaling for DIS
        flow_scale = metadata.get("flow_to_input_scale", 1)
        method = metadata.get("method", "raft")
        
        # Get input dimensions from metadata
        resized_shape = metadata.get("resized_shape")
        if resized_shape and len(resized_shape) >= 3:
            input_H, input_W = resized_shape[1], resized_shape[2]
        else:
            # Fallback: assume flow is at full resolution
            input_H, input_W = flow_H * flow_scale, flow_W * flow_scale
        
        print(f"Track optimization: method={method}, flow_scale={flow_scale}", file=sys.stderr)
        print(f"  Flow dims: {flow_W}x{flow_H}, Input dims: {input_W}x{input_H}", file=sys.stderr)
        print(f"  Correction method: {correction_method}", file=sys.stderr)
        
        # Parse anchors (expecting list of {"frame": int, "x": int, "y": int})
        anchors = []
        for anchor in anchors_json:
            frame = anchor["frame"]
            x = anchor["x"]
            y = anchor["y"]
            anchors.append((frame, x, y))
        
        # Sort anchors by frame (bidirectional propagation - anchor can start at any frame)
        anchors = sorted(anchors, key=lambda a: a[0])
        first_frame = anchors[0][0]
        last_frame = anchors[-1][0]
        
        # Build optimized track
        start_time = _pc()
        
        cancel_check = lambda: self._check_cancelled("Track optimization")

        if use_legacy:
            # Legacy DP-based corridor interpolation (not recommended for long segments)
            if corridor_width is None:
                corridor_msg = "adaptive corridor"
            elif corridor_width <= 0:
                corridor_msg = "full resolution (no corridor)"
            else:
                corridor_msg = f"corridor width={corridor_width}"
            
            print(f"Optimizing track with {len(anchors)} anchor points (LEGACY: {corridor_msg})", file=sys.stderr)
            print(f"  Anchor range: frame {first_frame} to frame {last_frame}", file=sys.stderr)
            
            # Legacy mode doesn't support flow scaling - use input dimensions
            builder = IncrementalTrackBuilder(
                T,
                input_H,
                input_W,
                flows,
                interpolate,
                corridor_width=corridor_width,
                segment_cache=self.segment_cache,
                cancel_check=cancel_check,
            )
            track = builder.build_track(anchors)
            stats = builder.get_cache_stats()
        elif correction_method == "blob_assisted":
            # Blob-assisted correction: combines flow with blob detection
            print(f"Optimizing track with {len(anchors)} anchor points (blob-assisted)", file=sys.stderr)
            print(f"  Anchor range: frame {first_frame} to frame {last_frame}", file=sys.stderr)
            print(f"  Blob params: search_radius={blob_search_radius}, blob_radius={blob_radius}", file=sys.stderr)
            
            # Use cached video accessor for blob detection (if caching enabled)
            target_width = input_W if input_W != flow_W else None
            target_height = input_H if input_H != flow_H else None
            video_accessor = self._get_or_create_video_accessor(video_path, target_width, target_height, use_cache=cache_video)
            
            # Determine if RGB
            is_rgb = video_accessor.ndim == 4 if hasattr(video_accessor, 'ndim') else video_path.lower().endswith('.avi')
            
            builder = FlowBlendBlobTrackBuilder(
                T, input_H, input_W, flows, video_accessor,
                flow_scale=flow_scale,
                input_dims=(input_H, input_W),
                search_radius=blob_search_radius,
                blob_radius=blob_radius,
                is_rgb=is_rgb,
                frame_shape=(input_H, input_W),
                cancel_check=cancel_check,
            )
            track = builder.build_track(anchors)
            stats = builder.get_cache_stats()
            # Close accessor if caching is disabled
            if not cache_video:
                try:
                    video_accessor.close()
                except Exception:
                    pass
        elif correction_method == "corridor_dp":
            # Corridor DP interpolation: original RAFT_v4 optimized DP with caching
            if corridor_width is None:
                corridor_msg = "adaptive corridor"
            elif corridor_width <= 0:
                corridor_msg = "full resolution (no corridor)"
            else:
                corridor_msg = f"corridor width={corridor_width}"
            
            print(f"Optimizing track with {len(anchors)} anchor points (corridor DP: {corridor_msg})", file=sys.stderr)
            print(f"  Anchor range: frame {first_frame} to frame {last_frame}", file=sys.stderr)
            if linear_interp_threshold > 0:
                print(f"  Linear interp threshold: {linear_interp_threshold} frames", file=sys.stderr)
            
            builder = CorridorDPTrackBuilder(
                T, input_H, input_W, flows,
                segment_cache=self.segment_cache,
                corridor_width=corridor_width,
                flow_scale=flow_scale,
                input_dims=(input_H, input_W),
                cancel_check=cancel_check,
                linear_interp_threshold=linear_interp_threshold,
            )
            track = builder.build_track(anchors)
            stats = builder.get_cache_stats()
        else:
            # Flow-based segment blending (supports DIS flow scaling)
            print(f"Optimizing track with {len(anchors)} anchor points (flow blend)", file=sys.stderr)
            print(f"  Anchor range: frame {first_frame} to frame {last_frame}", file=sys.stderr)
            if linear_interp_threshold > 0:
                print(f"  Linear interp threshold: {linear_interp_threshold} frames", file=sys.stderr)
            
            builder = FlowBlendTrackBuilder(
                T, input_H, input_W, flows,
                flow_scale=flow_scale,
                input_dims=(input_H, input_W),
                cancel_check=cancel_check,
                linear_interp_threshold=linear_interp_threshold,
            )
            track = builder.build_track(anchors)
            stats = builder.get_cache_stats()
        
        elapsed = _pc() - start_time
        
        print(f"✓ Track optimized in {elapsed:.3f}s", file=sys.stderr)
        if 'segments_cached' in stats:
            print(
                f"  Cache: {stats['segments_cached']} segments, {stats.get('cache_hit_rate', 0):.1%} hit rate",
                file=sys.stderr,
            )
        
        # Export to JSON with resolution metadata
        self._export_track_json(track, output_path, "Track1_Optimized", video_path)
        
        return {
            "status": "ok",
            "message": "Track optimized with anchors",
            "output_path": output_path,
            "elapsed_time": f"{elapsed:.3f}s",
            "cache_stats": stats,
            "correction_method": correction_method,
        }

    def _optimize_tracks(self, request):
        """Optimize multiple tracks in a single pass using flow-based segment blending."""
        # Check for cancellation before starting
        self._check_cancelled("Track optimization")

        cancel_check = lambda: self._check_cancelled("Track optimization")
        
        video_path = request["video_path"]
        tracks_spec = request.get("tracks", [])
        output_path = request.get("output_path")
        video_name = request.get("video_name")  # Optional video name override
        use_legacy = request.get("use_legacy_interpolation", False)
        corridor_width = self._normalize_corridor_width(request.get("corridor_width", None))
        
        # If output_path is a directory, construct the full file path
        if output_path and os.path.isdir(output_path):
            if video_name:
                base_name = video_name
            else:
                base_name = os.path.splitext(os.path.basename(video_path))[0]
            output_path = os.path.join(output_path, f"{base_name}_annotations.json")
        
        # Correction method: "full_blend" (default), "blob_assisted", or "corridor_dp"
        correction_method = request.get("correction_method", "full_blend")
        blob_search_radius = request.get("blob_search_radius", 15)
        blob_radius = request.get("blob_radius", 5.0)
        cache_video = request.get("cache_video", False)  # Whether to cache video in memory
        
        # Linear interpolation threshold for short anchor segments
        linear_interp_threshold = request.get("linear_interp_threshold", 0)

        if not tracks_spec:
            raise ValueError("No tracks provided for batch optimization")

        entry = self.flow_cache.get(video_path)
        if not entry:
            self._compute_flow({"video_path": video_path, "force_recompute": False})
            entry = self.flow_cache.get(video_path)
        if not entry:
            raise RuntimeError("Optical flow unavailable after recompute")

        flows = entry["flows_np"]
        metadata = entry.get("metadata", {})
        
        # Get flow dimensions
        Tm1, flow_H, flow_W, _ = flows.shape
        T = Tm1 + 1
        
        # Get flow scaling for DIS
        flow_scale = metadata.get("flow_to_input_scale", 1)
        method = metadata.get("method", "raft")
        
        # Get input dimensions from metadata
        resized_shape = metadata.get("resized_shape")
        if resized_shape and len(resized_shape) >= 3:
            input_H, input_W = resized_shape[1], resized_shape[2]
        else:
            input_H, input_W = flow_H * flow_scale, flow_W * flow_scale

        tracks_out = []
        per_track_info = []
        total_elapsed = 0.0
        
        # Create track builder based on correction method
        video_accessor = None
        flow_blend_builder = None
        flow_blob_builder = None
        corridor_dp_builder = None
        
        if use_legacy:
            # Legacy mode uses DP-based corridor interpolation (created per track)
            pass
        elif correction_method == "corridor_dp":
            # Corridor DP: original RAFT_v4 optimized DP with caching
            if corridor_width is None:
                corridor_msg = "adaptive corridor"
            elif corridor_width <= 0:
                corridor_msg = "full resolution (no corridor)"
            else:
                corridor_msg = f"corridor width={corridor_width}"
            print(f"Batch optimization using Corridor DP ({corridor_msg})", file=sys.stderr)
            if linear_interp_threshold > 0:
                print(f"  Linear interp threshold: {linear_interp_threshold} frames", file=sys.stderr)
            corridor_dp_builder = CorridorDPTrackBuilder(
                T, input_H, input_W, flows,
                segment_cache=self.segment_cache,
                corridor_width=corridor_width,
                flow_scale=flow_scale,
                input_dims=(input_H, input_W),
                cancel_check=cancel_check,
                linear_interp_threshold=linear_interp_threshold,
            )
        elif correction_method == "blob_assisted":
            # Blob-assisted: use cached video accessor for blob detection (if caching enabled)
            print(f"Batch optimization using blob-assisted correction", file=sys.stderr)
            print(f"  Blob params: search_radius={blob_search_radius}, blob_radius={blob_radius}", file=sys.stderr)
            target_width = input_W if input_W != flow_W else None
            target_height = input_H if input_H != flow_H else None
            video_accessor = self._get_or_create_video_accessor(video_path, target_width, target_height, use_cache=cache_video)
            is_rgb = video_accessor.ndim == 4 if hasattr(video_accessor, 'ndim') else video_path.lower().endswith('.avi')
            flow_blob_builder = FlowBlendBlobTrackBuilder(
                T, input_H, input_W, flows, video_accessor,
                flow_scale=flow_scale,
                input_dims=(input_H, input_W),
                search_radius=blob_search_radius,
                blob_radius=blob_radius,
                is_rgb=is_rgb,
                frame_shape=(input_H, input_W),
                cancel_check=cancel_check,
            )
            # Will close accessor after batch if caching is disabled
        else:
            # Standard flow blend (no blob detection)
            if linear_interp_threshold > 0:
                print(f"  Linear interp threshold: {linear_interp_threshold} frames", file=sys.stderr)
            flow_blend_builder = FlowBlendTrackBuilder(
                T, input_H, input_W, flows,
                flow_scale=flow_scale,
                input_dims=(input_H, input_W),
                cancel_check=cancel_check,
                linear_interp_threshold=linear_interp_threshold,
            )
        
        try:
            for idx, track_spec in enumerate(tracks_spec, 1):
                cancel_check()
                track_id = track_spec.get("track_id", f"Track{idx}")
                anchors_json = track_spec.get("anchors", [])
                seed = track_spec.get("seed")

                anchors = []
                for anchor in anchors_json:
                    anchors.append((int(anchor["frame"]), float(anchor["x"]), float(anchor["y"])))
                anchors = sorted(anchors, key=lambda a: a[0])

                if use_legacy:
                    method_name = "legacy DP"
                elif correction_method == "corridor_dp":
                    method_name = "corridor DP"
                elif correction_method == "blob_assisted":
                    method_name = "blob-assisted"
                else:
                    method_name = "flow blend"
                print(f"Batch optimizing track '{track_id}' with {len(anchors)} anchors ({method_name})", file=sys.stderr)

                start_time = _pc()
                if anchors:
                    if use_legacy:
                        # Legacy DP-based corridor interpolation
                        builder = IncrementalTrackBuilder(
                            T,
                            input_H,
                            input_W,
                            flows,
                            interpolate,
                            corridor_width=corridor_width,
                            segment_cache=self.segment_cache,
                            cancel_check=cancel_check,
                        )
                        track_np = builder.build_track(anchors)
                        stats = builder.get_cache_stats()
                    elif flow_blob_builder is not None:
                        # Blob-assisted flow blend
                        track_np = flow_blob_builder.build_track(anchors)
                        stats = flow_blob_builder.get_cache_stats()
                    elif corridor_dp_builder is not None:
                        # Corridor DP interpolation
                        track_np = corridor_dp_builder.build_track(anchors)
                        stats = corridor_dp_builder.get_cache_stats()
                    else:
                        # Standard flow-based segment blending
                        track_np = flow_blend_builder.build_track(anchors)
                        stats = flow_blend_builder.get_cache_stats()
                elif seed:
                    seed_x = float(seed.get("x"))
                    seed_y = float(seed.get("y"))
                    seed_frame = int(seed.get("frame", 0))  # 0-indexed, default to 0 for backward compatibility
                    track_np = self._propagate_with_flows(
                        flows, seed_x, seed_y, seed_frame=seed_frame,
                        flow_scale=flow_scale,
                        input_dims=(input_H, input_W)
                    )
                    stats = {
                        "mode": "propagate",
                        "segments_cached": len(self.segment_cache),
                        "cache_hit_rate": self.segment_cache.stats().get("cache_hit_rate", 0.0),
                    }
                else:
                    raise ValueError(f"Track '{track_id}' missing anchors or seed")

                elapsed = _pc() - start_time
                total_elapsed += elapsed

                print(
                    f"  ✓ Track '{track_id}' completed in {elapsed:.2f}s",
                    file=sys.stderr,
                )

                tracks_out.append((track_id, track_np))
                per_track_info.append({
                    "track_id": track_id,
                    "frames": len(track_np),
                    "duration_sec": elapsed,
                    "cache_stats": stats,
                })
        finally:
            # Close the video accessor if caching is disabled
            if video_accessor is not None and not cache_video:
                try:
                    video_accessor.close()
                except Exception:
                    pass

        if output_path:
            # Export with resolution metadata
            self._export_tracks_json(tracks_out, output_path, video_path)

        cache_overall = self.segment_cache.stats()

        return {
            "status": "ok",
            "message": f"Optimized {len(tracks_out)} track(s)",
            "correction_method": correction_method,
            "output_path": output_path,
            "tracks": [self._track_to_json(track_np, track_id) for track_id, track_np in tracks_out],
            "per_track": per_track_info,
            "cache_stats": cache_overall,
            "elapsed_time": f"{total_elapsed:.2f}s",
        }
    
    def _track_to_json(self, track, track_id, scale_x=1.0, scale_y=1.0):
        """Convert track array to JSON format.
        
        Args:
            track: Numpy array of shape (T, 2) with x,y coordinates
            track_id: String identifier for the track
            scale_x: Scale factor to convert x coordinates to original resolution
            scale_y: Scale factor to convert y coordinates to original resolution
        """
        frames_list = []
        for t in range(len(track)):
            frames_list.append({
                "frame": int(t),
                "x": int(round(track[t, 0] * scale_x)),
                "y": int(round(track[t, 1] * scale_y)),
            })
        return {
            "track_id": track_id,
            "color": {"r": 255, "g": 0, "b": 0, "a": 200},
            "frames": frames_list,
        }

    def _export_tracks_json(self, tracks, output_path, video_path=None):
        """Export tracks to JSON with coordinates in ORIGINAL resolution.
        
        All coordinates are automatically scaled to the original video resolution.
        This makes exported files portable and importable across different compression levels.
        
        Args:
            tracks: List of (track_id, track_np) tuples
            output_path: Path to save JSON file
            video_path: Optional video path to get resolution metadata for scaling
        """
        # Calculate scale factors to convert from working to original resolution
        scale_x = 1.0
        scale_y = 1.0
        
        if video_path and video_path in self.video_metadata:
            metadata = self.video_metadata[video_path]
            if 'original_shape' in metadata and 'resized_shape' in metadata:
                orig_shape = metadata['original_shape']
                resized_shape = metadata['resized_shape']
                # original_shape and resized_shape are (T, H, W) tuples
                if len(orig_shape) >= 3 and len(resized_shape) >= 3:
                    orig_H, orig_W = orig_shape[1], orig_shape[2]
                    resized_H, resized_W = resized_shape[1], resized_shape[2]
                    if resized_W > 0 and resized_H > 0:
                        scale_x = orig_W / resized_W
                        scale_y = orig_H / resized_H
        
        payload = {
            "metadata": {
                "total_frames": len(tracks[0][1]) if tracks else 0
            },
            "tracks": [self._track_to_json(track_np, track_id, scale_x, scale_y) for track_id, track_np in tracks]
        }
        
        # Store source filename for validation (only check that remains)
        if video_path:
            import os
            payload["metadata"]["source_filename"] = os.path.basename(video_path)
        
        with open(output_path, "w") as f:
            json.dump(payload, f, indent=2)

    def _export_track_json(self, track, output_path, track_id, video_path=None):
        self._export_tracks_json([(track_id, track)], output_path, video_path)
    
    def _load_flow(self, request):
        """Load a pre-computed optical flow file into the cache.
        
        This allows switching between different flow methods without recomputing.
        
        Request parameters:
            video_path: Path to the original video file
            flow_path: Path to the .npz flow file to load
            method: The flow method name (raft, locotrack, trackpy)
        """
        video_path = request["video_path"]
        flow_path = request["flow_path"]
        method = request.get("method", "raft")
        
        # Clear old flow cache if switching to a different video
        self._clear_flow_cache_for_new_video(video_path)
        
        if not os.path.exists(flow_path):
            return {
                "status": "error",
                "message": f"Flow file not found: {flow_path}"
            }
        
        print(f"Loading {method.upper()} flow from: {flow_path}")
        
        try:
            data = np.load(flow_path, allow_pickle=False)
            # Memory-efficient loading with automatic float16 compression
            flows = load_flows_memory_efficient(data['flows'])
            
            # Load metadata - handle all fields that may be saved
            metadata = {
                'method': method
            }
            if 'original_shape' in data:
                metadata['original_shape'] = tuple(int(x) for x in data['original_shape'])
            if 'resized_shape' in data:
                metadata['resized_shape'] = tuple(int(x) for x in data['resized_shape'])
            if 'is_avi' in data:
                metadata['is_avi'] = bool(data['is_avi'])
            
            # CRITICAL: Load DIS-specific metadata for proper scaling
            if 'flow_shape' in data:
                metadata['flow_shape'] = tuple(int(x) for x in data['flow_shape'])
            if 'flow_to_input_scale' in data:
                metadata['flow_to_input_scale'] = float(data['flow_to_input_scale'])
            if 'downsample_factor' in data:
                metadata['downsample_factor'] = int(data['downsample_factor'])
            
            # If flow_to_input_scale is missing but this is DIS flow at reduced resolution,
            # try to infer it from the shapes
            if 'flow_to_input_scale' not in metadata and method == 'dis':
                if 'resized_shape' in metadata and len(flows.shape) == 4:
                    input_H = metadata['resized_shape'][0] if len(metadata['resized_shape']) >= 2 else metadata['resized_shape'][1]
                    flow_H = flows.shape[1]
                    if flow_H < input_H:
                        inferred_scale = input_H / flow_H
                        metadata['flow_to_input_scale'] = inferred_scale
                        metadata['flow_shape'] = flows.shape[:-1]  # (T-1, H, W)
                        print(f"  ⚠ Inferred flow_to_input_scale={inferred_scale:.1f} from shape mismatch")
            
            # Log what we loaded
            flow_scale = metadata.get('flow_to_input_scale', 1)
            print(f"  Metadata: method={method}, flow_to_input_scale={flow_scale}")
            if 'resized_shape' in metadata:
                print(f"  Input dims: {metadata['resized_shape']}, Flow dims: {flows.shape[1:3]}")
            
            # Store in cache under the video_path key
            self.flow_cache[video_path] = {
                "flows_np": flows,
                "timestamp": _pc(),
                "metadata": metadata
            }
            self.video_metadata[video_path] = metadata
            
            print(f"✓ {method.upper()} flow loaded into cache ({flows.shape})")
            
            return {
                "status": "ok",
                "message": f"{method.upper()} optical flow loaded",
                "shape": list(flows.shape),
                "method": method,
                "flow_to_input_scale": metadata.get('flow_to_input_scale', 1),
            }
            
        except Exception as e:
            return {
                "status": "error",
                "message": f"Failed to load flow file: {e}"
            }
    
    def _visualize_flow(self, request):
        """Generate flow visualization as colored video.
        
        Uses the currently cached optical flow for the video.
        The visualization will reflect whichever method was last used to compute flow.
        """
        # Check for cancellation before starting
        self._check_cancelled("Flow visualization")
        
        video_path = request["video_path"]
        output_path = request.get("output_path")  # Can be directory or file path
        video_name = request.get("video_name")  # Optional video name override
        save_to_disk = request.get("save_to_disk", True)  # Default True - viz files are user-triggered
        
        entry = self.flow_cache.get(video_path)
        if not entry:
            # No flow cached - compute RAFT flow as default
            print(f"No cached flow found, computing RAFT flow for visualization...")
            self._compute_flow_from_video(video_path)
            entry = self.flow_cache.get(video_path)
        if not entry:
            raise RuntimeError("Optical flow unavailable")

        # Get method from cached entry for logging
        cached_method = entry.get("metadata", {}).get("method", "raft")
        
        # If output_path is a directory, construct the full file path
        if output_path and os.path.isdir(output_path):
            if video_name:
                base_name = video_name
            else:
                base_name = os.path.splitext(os.path.basename(video_path))[0]
            output_path = os.path.join(output_path, f"{base_name}_{cached_method}_flow_viz.tif")
        elif output_path and "_flow_viz" in output_path:
            # Modify output path to include method name for clarity
            # e.g., video_flow_viz.tif -> video_raft_flow_viz.tif
            output_path = output_path.replace("_flow_viz", f"_{cached_method}_flow_viz")
        
        flows = entry["flows_np"]
        
        # Validate flow shape
        print(f"Flow array shape: {flows.shape}, ndim: {flows.ndim}")
        if flows.ndim != 4:
            raise RuntimeError(f"Invalid flow shape: {flows.shape}, expected 4D array (T-1, H, W, 2). "
                             f"Method: {cached_method}. Cache entry keys: {list(entry.keys())}")
        
        T_minus_1, H, W, _ = flows.shape
        
        print(f"Generating flow visualization for {video_path} (method: {cached_method})")
        start_time = _pc()
        
        # Compute global max flow for consistent coloring
        max_flow = np.sqrt(flows[:, :, :, 0]**2 + flows[:, :, :, 1]**2).max()
        
        # Convert each flow frame to color
        flow_viz_frames = []
        for i in range(T_minus_1):
            # Responsive cancellation while generating visualization
            self._check_cancelled("Flow visualization")
            flow_rgb = flow_to_color(flows[i], max_flow=max_flow)
            flow_viz_frames.append(flow_rgb)
        
        # Add black frame at the end (no flow for last frame)
        flow_viz_frames.append(np.zeros((H, W, 3), dtype=np.uint8))
        
        # Stack the frames
        flow_viz = np.stack(flow_viz_frames)
        
        # Save as TIFF only if save_to_disk is enabled and output_path is provided
        if output_path and save_to_disk:
            tifffile.imwrite(output_path, flow_viz, photometric='rgb')
            print(f"✓ Saved to {output_path}")
        
        elapsed = _pc() - start_time
        print(f"✓ Flow visualization generated in {elapsed:.2f}s")
        
        return {
            "status": "ok",
            "message": f"Flow visualization generated (method: {cached_method})",
            "output_path": output_path if (output_path and save_to_disk) else None,
            "shape": (len(flow_viz_frames), int(H), int(W), 3),
            "method": cached_method,
        }
    
    def _physics_optimize_global(self, request):
        """Run physics-informed global optimization on tracks.
        
        Uses Motion Vector Interpolation to optimize uncompleted tracks using
        completed (user-verified) tracks as reference. The algorithm:
        1. Extracts motion vectors from completed tracks (GT motion)
        2. Interpolates these motion vectors to nearby positions using IDW
        3. Re-propagates uncompleted tracks with blended motion (90% GT, 10% flow)
        
        This approach provides 50-65% error reduction when sufficient reference
        tracks are available (validated experimentally).
        
        Request parameters (Java UI format):
            video_path: Path to video file
            completed_tracks: Array of {track_id, positions: [{frame, x, y}]}
            incomplete_tracks: Array of {track_id, positions: [{frame, x, y}]}
            correction_radius: Radius of influence for interpolation (default: 100)
            idw_power: Power for IDW interpolation (default: 2.0)
            blend_factor: How much GT motion vs flow (default: 0.9 = 90% GT)
        
        Returns:
            optimized_tracks: Array of {track_id, positions: [{frame, x, y}]}
            avg_improvement: Average correction magnitude in pixels
        """
        self._check_cancelled("Physics optimization")
        cancel_check = lambda: self._check_cancelled("Physics optimization")
        
        video_path = request.get("video_path")
        
        # Parse tracks from Java UI format
        completed_tracks_arr = request.get("completed_tracks", [])
        incomplete_tracks_arr = request.get("incomplete_tracks", [])
        
        # Optimal parameters from experimental validation:
        # - correction_radius: 100px (optimal interpolation radius)
        # - idw_power: 2.0 (inverse square weighting)
        # - blend_factor: 0.9 (90% GT motion, 10% optical flow)
        correction_radius = request.get("correction_radius", 100.0)
        idw_power = request.get("idw_power", 2.0)
        blend_factor = request.get("blend_factor", 0.9)
        
        if not completed_tracks_arr:
            return {
                "status": "error",
                "message": "No completed tracks to use as constraints"
            }
        
        if not incomplete_tracks_arr:
            return {
                "status": "ok",
                "message": "No incomplete tracks to optimize",
                "optimized_tracks": [],
                "avg_improvement": 0.0
            }
        
        # Get cached flow
        entry = self.flow_cache.get(video_path) if video_path else None
        if not entry:
            return {
                "status": "error",
                "message": "No optical flow cached for this video. Compute flow first."
            }
        
        flows = entry["flows_np"]
        metadata = entry.get("metadata", {})
        flow_scale = metadata.get("flow_to_input_scale", 1)
        
        print(f"Physics optimization: {len(completed_tracks_arr)} completed, "
              f"{len(incomplete_tracks_arr)} incomplete tracks", file=sys.stderr)
        
        # Import physics module
        try:
            from physics_informed_mesh import FlowCorrectionFieldOptimizer
        except ImportError:
            try:
                # Try alternative import path
                import importlib.util
                spec = importlib.util.spec_from_file_location(
                    "physics_informed_mesh",
                    Path(__file__).parent / "physics_informed_mesh.py"
                )
                physics_module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(physics_module)
                FlowCorrectionFieldOptimizer = physics_module.FlowCorrectionFieldOptimizer
            except Exception as e:
                return {
                    "status": "error",
                    "message": f"Failed to import physics module: {e}"
                }
        
        # Convert track format from Java UI format to internal format
        # Input: [{track_id: str, positions: [{frame: int, x: float, y: float}]}]
        # Output: {track_id: {frame_int: (x, y)}}
        def convert_track_array(track_arr):
            result = {}
            for track_obj in track_arr:
                track_id = track_obj.get("track_id")
                positions = track_obj.get("positions", [])
                if track_id:
                    converted = {}
                    for pos in positions:
                        frame = int(pos["frame"])
                        x = float(pos["x"])
                        y = float(pos["y"])
                        converted[frame] = (x, y)
                    result[track_id] = converted
            return result
        
        completed_tracks = convert_track_array(completed_tracks_arr)
        uncompleted_tracks = convert_track_array(incomplete_tracks_arr)
        
        if not uncompleted_tracks:
            return {
                "status": "ok",
                "message": "No incomplete tracks to optimize",
                "optimized_tracks": [],
                "avg_improvement": 0.0
            }
        
        start_time = _pc()
        
        # Run optimization with Motion Vector Interpolation
        optimizer = FlowCorrectionFieldOptimizer(
            flows=flows,
            correction_radius=correction_radius,
            flow_scale=flow_scale,
            power=idw_power,
            blend_factor=blend_factor,
            verbose=True
        )
        
        corrected = optimizer.optimize(
            completed_tracks=completed_tracks,
            uncompleted_tracks=uncompleted_tracks,
            cancel_check=cancel_check
        )
        
        elapsed = _pc() - start_time
        
        # Convert back to Java UI format
        # From: {track_id: {frame_int: (x, y)}}
        # To: [{track_id: str, positions: [{frame: int, x: float, y: float}]}]
        optimized_tracks = []
        total_correction = 0.0
        correction_count = 0
        
        for track_id, positions in corrected.items():
            # Calculate average correction for this track
            orig_track = uncompleted_tracks.get(track_id, {})
            positions_list = []
            for frame, (x, y) in sorted(positions.items()):
                positions_list.append({"frame": frame, "x": x, "y": y})
                
                # Calculate correction magnitude
                if frame in orig_track:
                    orig_x, orig_y = orig_track[frame]
                    dx = x - orig_x
                    dy = y - orig_y
                    total_correction += (dx*dx + dy*dy)**0.5
                    correction_count += 1
            
            optimized_tracks.append({
                "track_id": track_id,
                "positions": positions_list
            })
        
        avg_improvement = total_correction / max(correction_count, 1)
        
        print(f"✓ Physics optimization complete in {elapsed:.2f}s, "
              f"avg correction: {avg_improvement:.2f}px", file=sys.stderr)
        
        return {
            "status": "ok",
            "message": f"Optimized {len(corrected)} tracks using {len(completed_tracks)} constraints",
            "optimized_tracks": optimized_tracks,
            "avg_improvement": avg_improvement,
            "elapsed_seconds": elapsed,
            "completed_count": len(completed_tracks),
            "optimized_count": len(corrected)
        }
    
    def _get_mesh_preview(self, request):
        """Get Delaunay mesh visualization data for a specific frame.
        
        This allows the UI to display the mesh structure overlaid on the video,
        helping users understand how corrections will propagate.
        
        Request parameters:
            video_path: Path to video file
            all_tracks: Dict of track_id -> {frame: {x, y}} for all tracks
            completed_track_ids: List of completed track IDs
            frame: Frame number to get mesh for
        
        Returns:
            mesh_data: Data for rendering the mesh (points, edges, triangles)
        """
        self._check_cancelled("Mesh preview")
        
        all_tracks = request.get("all_tracks", {})
        completed_track_ids = set(request.get("completed_track_ids", []))
        frame = request.get("frame", 0)
        
        if not all_tracks:
            return {
                "status": "error",
                "message": "No tracks provided"
            }
        
        # Import physics module
        try:
            from physics_informed_mesh import FrameMesh
        except ImportError:
            try:
                import importlib.util
                spec = importlib.util.spec_from_file_location(
                    "physics_informed_mesh",
                    Path(__file__).parent / "physics_informed_mesh.py"
                )
                physics_module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(physics_module)
                FrameMesh = physics_module.FrameMesh
            except Exception as e:
                return {
                    "status": "error",
                    "message": f"Failed to import physics module: {e}"
                }
        
        # Convert track format and gather positions for this frame
        points = []
        track_ids = []
        completed_indices = []
        
        for track_id, positions in all_tracks.items():
            frame_str = str(frame)
            if frame_str in positions:
                pos = positions[frame_str]
                if isinstance(pos, dict):
                    x, y = float(pos['x']), float(pos['y'])
                else:
                    x, y = float(pos[0]), float(pos[1])
                
                points.append([x, y])
                if track_id in completed_track_ids:
                    completed_indices.append(len(track_ids))
                track_ids.append(track_id)
        
        if len(points) < 3:
            return {
                "status": "ok",
                "message": "Not enough points for mesh (need at least 3)",
                "mesh_data": None
            }
        
        import numpy as np
        points_array = np.array(points)
        
        # Build mesh
        mesh = FrameMesh(
            frame=frame,
            points=points_array,
            track_ids=track_ids
        )
        
        if not mesh.build():
            return {
                "status": "ok",
                "message": "Could not build mesh",
                "mesh_data": None
            }
        
        # Get edges (avoid duplicates)
        edges = set()
        for tri in mesh.triangles:
            for i in range(3):
                v1 = tri.vertices[i]
                v2 = tri.vertices[(i + 1) % 3]
                edge = (min(v1, v2), max(v1, v2))
                edges.add(edge)
        
        edge_coords = []
        for v1, v2 in edges:
            p1 = points[v1]
            p2 = points[v2]
            edge_coords.append({
                "x1": p1[0], "y1": p1[1],
                "x2": p2[0], "y2": p2[1]
            })
        
        # Quality metrics
        quality = mesh.compute_mesh_quality()
        
        return {
            "status": "ok",
            "message": f"Mesh built with {len(mesh.triangles)} triangles",
            "mesh_data": {
                "points": [{"x": p[0], "y": p[1], "track_id": track_ids[i]} 
                          for i, p in enumerate(points)],
                "edges": edge_coords,
                "triangles": [list(tri.vertices) for tri in mesh.triangles],
                "completed_indices": completed_indices,
                "quality": {
                    "mean_aspect_ratio": quality["mean_aspect_ratio"],
                    "min_angle": quality["min_angle"],
                    "coverage": quality["coverage"]
                },
                "frame": frame,
                "num_points": len(points),
                "num_triangles": len(mesh.triangles)
            }
        }

    def _finetune_locotrack(self, request):
        """Fine-tune LocoTrack model on user-annotated tracking data.
        
        This allows users to fine-tune the LocoTrack model on their specific
        domain (e.g., fluorescent microscopy) to improve tracking accuracy.
        
        Request parameters:
            video_path: Path to video file used for annotation
            annotations_json: JSON string of annotations with tracks and visible_segments
            base_weights: Path to base LocoTrack weights to fine-tune from
            output_weights: Path to save fine-tuned weights
            epochs: Number of training epochs (default: 100)
            batch_size: Batch size for training (default: 1)
            learning_rate: Learning rate (default: 1e-4)
            train_split: Train/test split ratio (default: 0.85)
            model_type: Model type - 'base' or 'small' (default: 'base')
            resolution: Training resolution - 'native' (256x256) or 'original' (default: 'native')
            use_turbo: Use fast LoRA-based turbo fine-tuning (default: False)
        
        Returns:
            status: 'ok' or 'error'
            message: Status message
            test_accuracy: Final test accuracy metrics
            train_loss: Final training loss
            epochs_trained: Number of epochs completed
        """
        self._check_cancelled("LocoTrack fine-tuning")
        cancel_check = lambda: self._check_cancelled("LocoTrack fine-tuning")
        
        video_path = request.get("video_path")
        annotations_json = request.get("annotations_json")
        base_weights = request.get("base_weights")
        output_weights = request.get("output_weights")
        epochs = request.get("epochs", 100)
        batch_size = request.get("batch_size", 1)
        learning_rate = request.get("learning_rate", 1e-4)
        train_split = request.get("train_split", 0.85)
        model_type = request.get("model_type", "base")
        resolution = request.get("resolution", "native")  # 'native' = 256x256, 'original' = video resolution
        use_turbo = request.get("use_turbo", False)  # Turbo mode: fast LoRA-based fine-tuning
        
        # Validate required parameters
        if not video_path:
            return {"status": "error", "message": "video_path is required"}
        if not annotations_json:
            return {"status": "error", "message": "annotations_json is required"}
        if not base_weights:
            return {"status": "error", "message": "base_weights is required"}
        if not output_weights:
            return {"status": "error", "message": "output_weights is required"}
        
        # Check paths exist
        from pathlib import Path
        video_path = Path(video_path)
        base_weights = Path(base_weights)
        output_weights = Path(output_weights)
        
        if not video_path.exists():
            return {"status": "error", "message": f"Video not found: {video_path}"}
        if not base_weights.exists():
            return {"status": "error", "message": f"Base weights not found: {base_weights}"}
        
        # Create output directory if needed
        output_weights.parent.mkdir(parents=True, exist_ok=True)
        
        print(f"Starting LocoTrack fine-tuning on {video_path}", file=sys.stderr)
        print(f"  Base weights: {base_weights}", file=sys.stderr)
        print(f"  Output weights: {output_weights}", file=sys.stderr)
        print(f"  Mode: {'TURBO' if use_turbo else 'Standard'}", file=sys.stderr)
        print(f"  Epochs/Iterations: {epochs}, Batch size: {batch_size}, LR: {learning_rate}", file=sys.stderr)
        
        try:
            # Parse annotations JSON first (needed for both modes)
            import json
            try:
                annotations = json.loads(annotations_json)
            except json.JSONDecodeError as e:
                return {"status": "error", "message": f"Invalid annotations JSON: {e}"}
            
            # Check minimum track count
            tracks = annotations.get("tracks", [])
            if len(tracks) < 4:
                return {
                    "status": "error",
                    "message": f"Need at least 4 tracks for fine-tuning, got {len(tracks)}"
                }
            
            cancel_check()
            
            # ============================================================
            # TURBO MODE: Fast LoRA-based fine-tuning (~30 seconds)
            # ============================================================
            if use_turbo:
                print("⚡ Using TURBO fine-tuning mode (LoRA adapters)", file=sys.stderr)
                
                import importlib.util
                turbo_script = Path(__file__).parent / "locotrack_turbo_finetune.py"
                
                if not turbo_script.exists():
                    return {
                        "status": "error",
                        "message": f"Turbo fine-tuning script not found: {turbo_script}"
                    }
                
                spec = importlib.util.spec_from_file_location("locotrack_turbo_finetune", turbo_script)
                turbo_module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(turbo_module)
                
                start_time = _pc()
                
                # Turbo mode saves adapter (.pth) not full weights (.ckpt)
                adapter_output = output_weights.with_suffix('.pth')
                
                result = turbo_module.turbo_finetune(
                    video_path=str(video_path),
                    annotations=annotations,
                    base_weights=str(base_weights),
                    output_adapter=str(adapter_output),
                    iterations=epochs,  # epochs = iterations in turbo mode
                    learning_rate=learning_rate,
                    batch_size=batch_size,
                )
                
                elapsed = _pc() - start_time
                
                if result.get("status") == "error":
                    return result
                
                # Calculate improvement percentage
                best_val_error = result.get("best_val_error", 0)
                final_val_error = result.get("final_val_error", 0)
                # Improvement is the reduction in error (if final < best, we compare against initial)
                # Note: losses list contains training losses, first val error isn't captured
                # We'll report improvement as 0 if we can't calculate it properly
                improvement_percent = 0
                if best_val_error > 0:
                    # Best val error is our improvement target
                    improvement_percent = 0  # Would need initial val error to calculate properly
                
                print(f"✓ Turbo fine-tuning complete in {elapsed:.1f}s", file=sys.stderr)
                print(f"  Adapter saved to: {adapter_output}", file=sys.stderr)
                
                return {
                    "status": "ok",
                    "message": f"Turbo fine-tuning complete. Adapter saved to {adapter_output}",
                    "test_accuracy": {
                        "position_error": result.get("best_val_error", 0),
                        "improvement": result.get("improvement_percent", 0),
                    },
                    "train_loss": result.get("final_val_error", 0),
                    "epochs_trained": epochs,
                    "elapsed_seconds": elapsed,
                    "num_tracks": len(tracks),
                    "output_weights": str(adapter_output),
                    "mode": "turbo"
                }
            
            # ============================================================
            # STANDARD MODE: Full fine-tuning (slower but more accurate)
            # ============================================================
            print("🔧 Using Standard fine-tuning mode", file=sys.stderr)
            
            # Import the fine-tuning module
            import importlib.util
            finetune_script = Path(__file__).parent / "locotrack_finetune.py"
            
            if not finetune_script.exists():
                return {
                    "status": "error",
                    "message": f"Fine-tuning script not found: {finetune_script}"
                }
            
            spec = importlib.util.spec_from_file_location("locotrack_finetune", finetune_script)
            finetune_module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(finetune_module)
            
            # For fine-tuning, we need to track progress but the IPC is synchronous.
            # Progress updates are printed to stderr for logging but cannot be sent
            # to Java in real-time with the current socket protocol.
            # We log them to console for debugging purposes.
            def progress_callback(progress_info):
                """Log progress updates to stderr for monitoring."""
                try:
                    phase = progress_info.get('phase', '')
                    epoch = progress_info.get('epoch', 0)
                    total_epochs = progress_info.get('total_epochs', 0)
                    message = progress_info.get('message', '')
                    train_loss = progress_info.get('train_loss', 0)
                    val_loss = progress_info.get('val_loss', 0)
                    
                    # Format progress bar for console
                    if phase == 'epoch_start':
                        print(f"\n[Fine-tune] Epoch {epoch}/{total_epochs}", file=sys.stderr, flush=True)
                    elif phase == 'train':
                        batch = progress_info.get('batch', 0)
                        total_batches = progress_info.get('total_batches', 0)
                        pct = batch * 100 // max(total_batches, 1)
                        bar = '█' * (pct // 5) + '░' * (20 - pct // 5)
                        print(f"\r  Training: [{bar}] {pct}%", end='', file=sys.stderr, flush=True)
                    elif phase == 'validate_start':
                        print(f"\n  Validating...", end='', file=sys.stderr, flush=True)
                    elif phase == 'epoch_end':
                        print(f"\n  Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}", file=sys.stderr, flush=True)
                    elif phase == 'model_saved':
                        print(f"  ✓ Saved best model (val_loss={val_loss:.4f})", file=sys.stderr, flush=True)
                    elif phase == 'early_stop':
                        print(f"\n  ⚡ Early stopping triggered", file=sys.stderr, flush=True)
                    elif phase == 'testing':
                        print(f"\n[Fine-tune] Computing final test metrics...", file=sys.stderr, flush=True)
                    elif phase == 'complete':
                        print(f"\n[Fine-tune] Training complete!", file=sys.stderr, flush=True)
                except Exception as e:
                    pass  # Silently ignore progress callback errors
            
            # Determine resize_to based on resolution setting
            # 'native' = 256x256 (LocoTrack's original training resolution)
            # 'original' = None (use video's original resolution)
            resize_to = (256, 256) if resolution == "native" else None
            print(f"  Resolution: {resolution} -> resize_to={resize_to}", file=sys.stderr)
            
            # Run fine-tuning
            start_time = _pc()
            
            result = finetune_module.finetune_locotrack(
                video_path=str(video_path),
                annotations=annotations,
                base_weights=str(base_weights),
                output_weights=str(output_weights),
                epochs=epochs,
                batch_size=batch_size,
                learning_rate=learning_rate,
                train_split=train_split,
                model_type=model_type,
                resize_to=resize_to,
                cancel_check=cancel_check,
                progress_callback=progress_callback
            )
            
            elapsed = _pc() - start_time
            
            if result.get("status") == "error":
                return result
            
            # Extract test metrics from result
            test_metrics = result.get("test_metrics", {})
            train_losses = result.get("train_losses", [])
            train_loss = train_losses[-1] if train_losses else 0.0
            
            print(f"✓ Fine-tuning complete in {elapsed:.1f}s", file=sys.stderr)
            print(f"  Test metrics: {test_metrics}", file=sys.stderr)
            print(f"  Weights saved to: {output_weights}", file=sys.stderr)
            
            return {
                "status": "ok",
                "message": f"Fine-tuning complete. Weights saved to {output_weights}",
                "test_accuracy": test_metrics,  # Java expects test_accuracy
                "train_loss": train_loss,
                "epochs_trained": result.get("epochs_trained", epochs),
                "elapsed_seconds": elapsed,
                "num_tracks": len(tracks),
                "output_weights": str(output_weights)
            }
            
        except Exception as e:
            import traceback
            print(f"Fine-tuning error: {e}", file=sys.stderr)
            traceback.print_exc(file=sys.stderr)
            return {
                "status": "error",
                "message": f"Fine-tuning failed: {str(e)}"
            }

    def _compute_locotrack_flow(self, request):
        """Compute optical flow using LocoTrack point tracking.
        
        This is designed for fluorescent microscopy images where traditional
        optical flow methods may struggle with sparse, bright point sources.
        Uses DoG detector for seed initialization and LocoTrack for tracking.
        
        Request parameters (TrackMate-style):
            video_path: Path to video file (TIFF)
            output_path: Optional path to save flow
            force_recompute: Force recomputation even if cached
            diameter: Estimated blob diameter in pixels (TrackMate UI, preferred)
            radius: Legacy radius override (pixels)
            threshold: Quality threshold, 0 = accept all (default: 0.0)
            median_filter: Pre-process with median filter (default: False)
            subpixel: Sub-pixel localization (default: True)
            min_distance: Minimum distance between seeds (default: 5)
            max_keypoints: Maximum number of seeds (default: 500)
            max_occlusion_ratio: Maximum occlusion ratio for filtering (default: 0.5)
            flow_smoothing: RBF smoothing (default: 15.0)
            gaussian_sigma: Gaussian smoothing (default: 2.0)
            temporal_smooth_factor: Temporal smoothing strength 0-1 (default: 0.1)
            seed_frames: List of frame indices to detect seeds (default: [0])
            kernel: Interpolation kernel type (default: 'gaussian_rbf'):
                - 'gaussian_rbf': Scipy Gaussian RBF (GRBF, accurate)
                - 'gaussian': GPU Gaussian kernel (fast)
                - 'idw': Inverse Distance Weighting
                - 'wendland': Wendland kernel (compact support)
                - 'thin_plate_spline': Thin-plate spline RBF
            invert: Detect dark blobs on light background (default: False)
            save_to_disk: Whether to save computed flow to disk (default: True)
        """
        # Check for cancellation before starting
        self._check_cancelled("LocoTrack flow computation")
        
        video_path = request["video_path"]
        output_path = request.get("output_path", None)
        force_recompute = request.get("force_recompute", False)
        save_to_disk = request.get("save_to_disk", True)  # Default True for backward compatibility
        
        # Ensure force_recompute is a boolean (handle string values from CLI)
        if isinstance(force_recompute, str):
            force_recompute = force_recompute.lower() in ('true', '1', 'yes')
        else:
            force_recompute = bool(force_recompute)
        
        print(f"[LOCOTRACK FLOW] force_recompute={force_recompute}")

        # Allow caller to pass output_path as a directory ("--output-dir")
        output_path = self._normalize_flow_output_path(request, output_path)

        # Optional working resolution for cache filtering
        target_width = request.get("target_width", None)
        target_height = request.get("target_height", None)

        # Allow caller to pass output_path as a directory ("--output-dir")
        output_path = self._normalize_flow_output_path(request, output_path)

        # Optional working resolution for cache filtering
        target_width = request.get("target_width", None)
        target_height = request.get("target_height", None)
        
        # Clear old flow cache if switching to a different video
        self._clear_flow_cache_for_new_video(video_path)
        
        # TrackMate-style DoG parameters (TrackMate UI uses diameter)
        diameter = request.get("diameter", request.get("diameter_pixels", None))
        radius = request.get("radius", None)
        threshold = request.get("threshold", 0.0)
        median_filter = request.get("median_filter", False)
        subpixel = request.get("subpixel", True)
        
        # Convert diameter/radius to sigma values (TrackMate formula for 2D)
        # TrackMate uses: sigma = radius / sqrt(ndims), then sigma*0.9 and sigma*1.1
        ndims = 2  # 2D image
        if diameter is not None:
            radius_px = float(diameter) / 2.0
        elif radius is not None:
            radius_px = float(radius)
        else:
            radius_px = 2.5

        sigma_base = radius_px / np.sqrt(ndims)
        sigma_low = sigma_base * 0.9
        sigma_high = sigma_base * 1.1
        
        # Legacy parameter support (for backward compatibility)
        if "sigma_low" in request:
            sigma_low = request["sigma_low"]
        if "sigma_high" in request:
            sigma_high = request["sigma_high"]
        if "dog_threshold" in request:
            threshold = request["dog_threshold"]
        
        # Other parameters - min_distance based on radius (matching TrackMate)
        min_distance = request.get("min_distance", int(np.ceil(radius_px)))
        max_keypoints = request.get("max_keypoints", 500)
        # TrackMate does not exclude a fixed border by default; keep 0 unless user overrides
        exclude_border = request.get("exclude_border", 0)
        max_occlusion_ratio = request.get("max_occlusion_ratio", 0.5)
        flow_smoothing = request.get("flow_smoothing", 15.0)
        gaussian_sigma = request.get("gaussian_sigma", 2.0)
        temporal_smooth_factor = request.get("temporal_smooth_factor", 0.1)
        seed_frames = request.get("seed_frames", None)
        kernel = request.get("kernel", "gaussian_rbf")
        invert = request.get("invert", False)
        
        # Cache key includes parameters that affect the output
        # This ensures recalculation when user changes kernel, smoothing, seed frame, etc.
        seed_frames_str = str(seed_frames) if seed_frames else "[0]"
        cache_key = (
            f"{video_path}_locotrack_"
            f"r{radius_px:.2f}_t{threshold:.3f}_"
            f"k{kernel}_fs{flow_smoothing:.1f}_ts{temporal_smooth_factor:.2f}_"
            f"md{median_filter}_sp{subpixel}_inv{invert}_sf{seed_frames_str}"
        )
        
        print(f"[LocoTrack] Parameters: radius={radius_px:.2f}, kernel={kernel}, flow_smoothing={flow_smoothing}")
        print(f"[LocoTrack] Cache key: {cache_key}")
        
        # Check cache
        cache_entry = self.flow_cache.get(cache_key)
        if cache_entry and not force_recompute:
            print(f"[LocoTrack] HIT memory cache for {video_path}")
            flows = cache_entry["flows_np"]
            info = cache_entry.get("locotrack_info", {})
        # Check disk cache - search by pattern since filenames include parameters
        elif output_path and not force_recompute:
            disk_path = self._find_existing_flow_file(
                output_path, 'locotrack', target_width=target_width, target_height=target_height,
                radius=radius_px, threshold=threshold, kernel=kernel,
                flow_smoothing=flow_smoothing, temporal_smooth_factor=temporal_smooth_factor,
                median_filter=median_filter, subpixel=subpixel, invert=invert,
                seed_frames=seed_frames if seed_frames else [0]
            )
            if disk_path and os.path.exists(disk_path):
                print(f"[LocoTrack] Loading from disk cache: {disk_path}")
                try:
                    data = np.load(disk_path, allow_pickle=False)
                    # Memory-efficient loading with automatic float16 compression
                    flows = load_flows_memory_efficient(data['flows'])
                    info = eval(str(data.get('locotrack_info', '{}'))) if 'locotrack_info' in data.files else {}
                    
                    # Store in memory cache
                    if 'original_shape' in data:
                        metadata = {
                            'original_shape': tuple(int(x) for x in data['original_shape']),
                            'resized_shape': tuple(int(x) for x in data['resized_shape']),
                            'is_avi': bool(data['is_avi']) if 'is_avi' in data.files else False,
                            'method': 'locotrack'
                        }
                    else:
                        metadata = {'method': 'locotrack'}
                    
                    self.flow_cache[cache_key] = {
                        "flows_np": flows,
                        "timestamp": _pc(),
                        "metadata": metadata,
                        "locotrack_info": info
                    }
                    self.flow_cache[video_path] = self.flow_cache[cache_key]
                    self.video_metadata[video_path] = metadata
                    print(f"✓ LocoTrack flow loaded from disk ({flows.shape})")
                    
                    # Return early - no need to compute
                    response = {
                        "status": "ok",
                        "message": "LocoTrack optical flow loaded from cache",
                        "shape": list(flows.shape),
                        "method": "locotrack",
                        "cache_key": cache_key,
                        "locotrack_info": {
                            "total_seeds": info.get("total_seeds", 0),
                            "valid_trajectories": info.get("valid_trajectories", 0),
                            "filtered_trajectories": info.get("filtered_trajectories", 0),
                        }
                    }
                    return response
                except Exception as e:
                    print(f"[LocoTrack] Failed to load from disk cache: {e}")
                    print("[LocoTrack] Computing from scratch...")
                    # Fall through to compute below
            # No matching disk file found, fall through to compute
            pass
        
        # Compute locotrack flow (either no cache hit, or cache loading failed)
        if not (cache_entry and not force_recompute):
            print(f"[LocoTrack] Computing flow. force_recompute={force_recompute}")
            # Import locotrack flow module
            try:
                # Add same directory to path to find locotrack_flow module
                script_dir = os.path.dirname(os.path.abspath(__file__))
                if script_dir not in sys.path:
                    sys.path.insert(0, script_dir)
                
                from locotrack_flow import compute_locotrack_optical_flow
            except ImportError as e:
                return {
                    "status": "error",
                    "message": f"locotrack_flow module not available: {e}"
                }
            
            print(f"Computing LocoTrack optical flow for {video_path}")
            start_time = _pc()
            
            # Load video (raw + normalized). DoG uses raw intensities; tracking uses normalized.
            raw_video, v01, is_avi, orig_shape, resized_shape = self._load_video_raw_and_normalized(video_path)

            if is_avi:
                # Convert to grayscale for model; keep raw grayscale for DoG
                T, H, W, C = raw_video.shape
                dog_video_raw = np.mean(raw_video, axis=-1).astype(np.float32)
                video_gray = np.mean(v01, axis=-1).astype(np.float32)
            else:
                dog_video_raw = raw_video
                video_gray = v01
                T, H, W = v01.shape
            
            # Get or initialize LocoTrack manager
            try:
                locotrack_manager = self._get_locotrack_manager()
            except Exception as e:
                return {
                    "status": "error",
                    "message": f"Failed to initialize LocoTrack: {e}"
                }
            
            # Compute locotrack flow
            flows, info = compute_locotrack_optical_flow(
                video_gray,
                locotrack_manager,
                sigma_low=sigma_low,
                sigma_high=sigma_high,
                dog_threshold=threshold,
                min_distance=min_distance,
                max_keypoints=max_keypoints,
                exclude_border=exclude_border,
                max_occlusion_ratio=max_occlusion_ratio,
                flow_smoothing=flow_smoothing,
                gaussian_sigma=gaussian_sigma,
                temporal_smooth_factor=temporal_smooth_factor,
                kernel=kernel,
                invert=invert,
                seed_frames=seed_frames,
                median_filter=median_filter,
                subpixel=subpixel,
                dog_video_raw=dog_video_raw,
                verbose=True,
                cancel_check=lambda: self._check_cancelled("LocoTrack flow computation"),
            )
            
            elapsed = _pc() - start_time
            print(f"✓ LocoTrack flow computed in {elapsed:.2f}s")
            
            # Compress to float16 if beneficial
            flows = wrap_flows_memory_efficient(flows, "LocoTrack")
            
            # Cache the flow
            metadata = {
                'original_shape': orig_shape if orig_shape else (T, H, W),
                'resized_shape': resized_shape if resized_shape else (T, H, W),
                'is_avi': is_avi,
                'method': 'locotrack'
            }
            self.flow_cache[cache_key] = {
                "flows_np": flows,
                "timestamp": _pc(),
                "metadata": metadata,
                "locotrack_info": info
            }
            self.video_metadata[cache_key] = metadata
            
            # ALSO cache under main video_path key so visualization/tracking use this flow
            self.flow_cache[video_path] = {
                "flows_np": flows,
                "timestamp": _pc(),
                "metadata": metadata,
                "locotrack_info": info
            }
            self.video_metadata[video_path] = metadata
            
            # Save to file if requested and save_to_disk is enabled
            if output_path and save_to_disk:
                # Build descriptive filename with all locotrack parameters
                actual_output_path = self._build_flow_filename(
                    output_path, 'locotrack',
                    resized_shape=resized_shape,
                    original_shape=orig_shape,
                    radius=radius_px,
                    threshold=threshold,
                    kernel=kernel,
                    flow_smoothing=flow_smoothing,
                    temporal_smooth_factor=temporal_smooth_factor,
                    seed_frames=seed_frames if seed_frames else [0],
                    median_filter=median_filter,
                    subpixel=subpixel,
                    invert=invert
                )
                # Get raw array for saving (handles MemoryEfficientFlowArray)
                flows_to_save = flows.get_raw() if hasattr(flows, 'get_raw') else flows
                # Use uncompressed savez for ~4x faster loading (8% larger files)
                np.savez(
                    actual_output_path,
                    flows=flows_to_save,
                    original_shape=metadata['original_shape'],
                    resized_shape=metadata['resized_shape'],
                    is_avi=is_avi,
                    method='locotrack',
                    locotrack_info=str(info)
                )
                print(f"✓ LocoTrack flow saved to {actual_output_path}")
        
        response = {
            "status": "ok",
            "message": "LocoTrack optical flow computed",
            "shape": list(flows.shape),
            "method": "locotrack",
            "cache_key": cache_key,
            "locotrack_info": {
                "total_seeds": info.get("total_seeds", 0),
                "valid_trajectories": info.get("valid_trajectories", 0),
                "filtered_trajectories": info.get("filtered_trajectories", 0),
            }
        }
        
        return response
    
    def _compute_trackpy_flow(self, request):
        """Compute optical flow using DoG detection + trackpy trajectory linking.
        
        Uses the SAME DoG detector as LocoTrack for particle detection, then uses
        trackpy's linker for trajectory building, spectral smoothing (FFT/DCT) for
        temporal refinement, and kernel interpolation for dense flow generation.
        
        Request parameters:
            video_path: Path to video file (TIFF)
            output_path: Optional path to save flow
            force_recompute: Force recomputation even if cached
            
            # DoG detection parameters (shared with LocoTrack):
            radius: Estimated object radius in pixels (default: 2.5)
            threshold: Quality threshold for detection, 0 = accept all (default: 0.0)
            median_filter: Apply 3x3 median filter before detection (default: False)
            subpixel: Use sub-pixel localization (default: True)
            
            # Trackpy linking parameters:
            search_range: Maximum particle displacement between frames (default: 15)
            memory: Frames a particle can disappear and still link (default: 5)
            require_persistent: Only use trajectories present in all frames (default: False)
            
            # Spectral smoothing:
            smooth_factor: Spectral smoothing factor 0-1 (default: 0.1)
            use_dct_smoothing: Use DCT (True) or FFT (False) smoothing (default: False)
            
            # Flow field generation:
            flow_smoothing: Interpolation bandwidth (default: 15.0)
            kernel: Interpolation kernel type (default: 'gaussian_rbf')
            gaussian_sigma: Gaussian post-smoothing (default: 2.0)
            
            # Other:
            invert: Detect dark particles on light background (default: False)
            save_to_disk: Whether to save computed flow to disk (default: True)
            
            # Legacy parameters (for backward compatibility):
            diameter: If provided, converts to radius = diameter / 2
        """
        # Check for cancellation before starting
        self._check_cancelled("TrackPy flow computation")
        
        video_path = request["video_path"]
        output_path = request.get("output_path", None)
        force_recompute = request.get("force_recompute", False)
        save_to_disk = request.get("save_to_disk", True)  # Default True for backward compatibility
        
        # Ensure force_recompute is a boolean (handle string values from CLI)
        if isinstance(force_recompute, str):
            force_recompute = force_recompute.lower() in ('true', '1', 'yes')
        else:
            force_recompute = bool(force_recompute)
        
        print(f"[TRACKPY FLOW] force_recompute={force_recompute}")

        # Optional working resolution for cache filtering
        target_width = request.get("target_width", None)
        target_height = request.get("target_height", None)
        
        # Clear old flow cache if switching to a different video
        self._clear_flow_cache_for_new_video(video_path)
        
        # DoG detection parameters (shared with LocoTrack)
        radius = request.get("radius", 2.5)
        threshold = request.get("threshold", 0.0)
        median_filter_enabled = request.get("median_filter", False)
        subpixel = request.get("subpixel", True)
        
        # Option to use legacy trackpy detection instead of DoG
        use_legacy_detection = request.get("use_legacy_detection", False)
        
        # Native trackpy detection parameters - triggers native detection mode
        diameter = None
        minmass = request.get("minmass", 0)  # Minimum integrated brightness
        if "diameter" in request:
            diameter = int(request["diameter"])
            use_legacy_detection = True
            print(f"[Trackpy] Native mode: using trackpy detection (diameter={diameter}, minmass={minmass})")
        elif use_legacy_detection:
            # Convert radius to diameter for native mode
            diameter = int(radius * 2)
            print(f"[Trackpy] Native mode: using trackpy detection (diameter={diameter} from radius={radius})")
        
        # Trackpy linking parameters
        search_range = request.get("search_range", 15)
        memory = request.get("memory", 5)
        require_persistent = request.get("require_persistent", False)
        min_trajectory_length = request.get("min_trajectory_length", None)
        
        # Spectral smoothing parameters
        smooth_factor = request.get("smooth_factor", 0.1)
        use_dct_smoothing = request.get("use_dct_smoothing", False)
        
        # Flow field generation parameters
        flow_smoothing = request.get("flow_smoothing", 15.0)
        kernel = request.get("kernel", "gaussian_rbf")
        gaussian_sigma = request.get("gaussian_sigma", 2.0)
        
        # Other parameters
        invert = request.get("invert", False)
        
        # Cache key includes parameters that affect the output
        if use_legacy_detection and diameter is not None:
            # Native trackpy detection mode - use diameter and minmass
            cache_key = (
                f"{video_path}_trackpy_"
                f"d{diameter}_mm{minmass}_sr{search_range}_m{memory}_"
                f"k{kernel}_fs{flow_smoothing:.1f}_sf{smooth_factor:.2f}"
            )
            print(f"[Trackpy] Parameters: diameter={diameter}, minmass={minmass}, kernel={kernel}")
        else:
            # DoG detection mode - use radius and threshold
            cache_key = (
                f"{video_path}_trackpy_"
                f"r{radius:.2f}_t{threshold:.3f}_sr{search_range}_m{memory}_"
                f"k{kernel}_fs{flow_smoothing:.1f}_sf{smooth_factor:.2f}"
            )
            print(f"[Trackpy] Parameters: radius={radius:.2f}, threshold={threshold}, kernel={kernel}")
        print(f"[Trackpy] Cache key: {cache_key}")
        
        # Check cache
        cache_entry = self.flow_cache.get(cache_key)
        if cache_entry and not force_recompute:
            print(f"Using cached trackpy flow from memory for {video_path}")
            flows = cache_entry["flows_np"]
            info = cache_entry.get("trackpy_info", {})
        # Check disk cache - search by pattern since filenames include parameters
        elif output_path and not force_recompute:
            disk_path = self._find_existing_flow_file(
                output_path, 'trackpy', target_width=target_width, target_height=target_height,
                radius=radius, threshold=threshold, search_range=search_range, memory=memory,
                kernel=kernel, flow_smoothing=flow_smoothing, smooth_factor=smooth_factor,
                median_filter=median_filter_enabled, subpixel=subpixel, invert=invert,
                diameter=diameter, minmass=minmass, use_legacy_detection=use_legacy_detection
            )
            if disk_path and os.path.exists(disk_path):
                print(f"[Trackpy] Loading from disk cache: {disk_path}")
                try:
                    data = np.load(disk_path, allow_pickle=False)
                    # Memory-efficient loading with automatic float16 compression
                    flows = load_flows_memory_efficient(data['flows'])
                    info = eval(str(data.get('trackpy_info', '{}'))) if 'trackpy_info' in data.files else {}
                    
                    # Store in memory cache
                    if 'original_shape' in data:
                        metadata = {
                            'original_shape': tuple(int(x) for x in data['original_shape']),
                            'resized_shape': tuple(int(x) for x in data['resized_shape']),
                            'is_avi': bool(data['is_avi']) if 'is_avi' in data.files else False,
                            'method': 'trackpy'
                        }
                    else:
                        metadata = {'method': 'trackpy'}
                    
                    self.flow_cache[cache_key] = {
                        "flows_np": flows,
                        "timestamp": _pc(),
                        "metadata": metadata,
                        "trackpy_info": info
                    }
                    self.flow_cache[video_path] = self.flow_cache[cache_key]
                    self.video_metadata[video_path] = metadata
                    print(f"✓ Trackpy flow loaded from disk ({flows.shape})")
                    
                    # Return early - no need to compute
                    response = {
                        "status": "ok",
                        "message": "Trackpy optical flow loaded from cache",
                        "shape": list(flows.shape),
                        "method": "trackpy",
                        "cache_key": cache_key,
                        "trackpy_info": {
                            "total_particles": info.get("total_particles", 0),
                            "total_trajectories": info.get("total_trajectories", 0),
                            "used_trajectories": info.get("used_trajectories", 0),
                        }
                    }
                    return response
                except Exception as e:
                    print(f"[Trackpy] Failed to load from disk cache: {e}")
                    print("[Trackpy] Computing from scratch...")
                    # Fall through to compute below
            # No matching disk file found, fall through to compute
            pass
        
        # Compute trackpy flow (either no cache hit, or cache loading failed)
        if not (cache_entry and not force_recompute):
            # Import trackpy flow module
            try:
                # Add same directory to path to find trackpy_flow module
                script_dir = os.path.dirname(os.path.abspath(__file__))
                if script_dir not in sys.path:
                    sys.path.insert(0, script_dir)
                
                from trackpy_flow import compute_trackpy_optical_flow
            except ImportError as e:
                return {
                    "status": "error",
                    "message": f"trackpy_flow module not available: {e}. Install trackpy with: pip install trackpy"
                }
            
            # Compute trackpy flow
            if use_legacy_detection:
                print(f"Computing trackpy optical flow (LEGACY trackpy detection) for {video_path}")
            else:
                print(f"Computing trackpy optical flow (DoG detection + trackpy linking) for {video_path}")
            start_time = _pc()
            
            # Load video
            raw_video, v01, is_avi, orig_shape, resized_shape = self._load_video_raw_and_normalized(video_path)
            
            if is_avi:
                # Convert RGB to grayscale
                T, H, W, C = v01.shape
                video_gray = np.mean(v01, axis=-1).astype(np.float32) / 255.0
            else:
                video_gray = v01
                T, H, W = v01.shape
            
            # Compute trackpy flow
            flows, info = compute_trackpy_optical_flow(
                video_gray,
                # DoG detection parameters (used when diameter is None)
                radius=radius,
                threshold=threshold,
                median_filter=median_filter_enabled,
                subpixel=subpixel,
                # Trackpy linking parameters
                search_range=search_range,
                memory=memory,
                require_persistent=require_persistent,
                min_trajectory_length=min_trajectory_length,
                # Spectral smoothing
                smooth_factor=smooth_factor,
                use_dct_smoothing=use_dct_smoothing,
                # Flow field generation
                flow_smoothing=flow_smoothing,
                kernel=kernel,
                gaussian_sigma=gaussian_sigma,
                # Other
                invert=invert,
                verbose=True,
                # Native trackpy parameters - when diameter is set, uses trackpy native detection
                diameter=diameter,
                minmass=minmass,
                cancel_check=lambda: self._check_cancelled("TrackPy flow computation"),
            )
            
            elapsed = _pc() - start_time
            print(f"✓ Trackpy flow computed in {elapsed:.2f}s")
            
            # Compress to float16 if beneficial
            flows = wrap_flows_memory_efficient(flows, "Trackpy")
            
            # Cache the flow
            metadata = {
                'original_shape': orig_shape if orig_shape else (T, H, W),
                'resized_shape': resized_shape if resized_shape else (T, H, W),
                'is_avi': is_avi,
                'method': 'trackpy'
            }
            self.flow_cache[cache_key] = {
                "flows_np": flows,
                "timestamp": _pc(),
                "metadata": metadata,
                "trackpy_info": info
            }
            self.video_metadata[cache_key] = metadata
            
            # Also cache under main video_path key
            self.flow_cache[video_path] = {
                "flows_np": flows,
                "timestamp": _pc(),
                "metadata": metadata,
                "trackpy_info": info
            }
            self.video_metadata[video_path] = metadata
            
            # Save to file if requested and save_to_disk is enabled
            if output_path and save_to_disk:
                # Build descriptive filename with all trackpy parameters (now using DoG detection)
                actual_output_path = self._build_flow_filename(
                    output_path, 'trackpy',
                    resized_shape=resized_shape,
                    original_shape=orig_shape,
                    radius=radius,
                    threshold=threshold,
                    search_range=search_range,
                    memory=memory,
                    kernel=kernel,
                    flow_smoothing=flow_smoothing,
                    smooth_factor=smooth_factor,
                    median_filter=median_filter_enabled,
                    subpixel=subpixel,
                    invert=invert
                )
                # Get raw array for saving (handles MemoryEfficientFlowArray)
                flows_to_save = flows.get_raw() if hasattr(flows, 'get_raw') else flows
                # Use uncompressed savez for ~4x faster loading (8% larger files)
                np.savez(
                    actual_output_path,
                    flows=flows_to_save,
                    original_shape=metadata['original_shape'],
                    resized_shape=metadata['resized_shape'],
                    is_avi=is_avi,
                    method='trackpy',
                    trackpy_info=str(info)
                )
                print(f"✓ Trackpy flow saved to {actual_output_path}")
        
        response = {
            "status": "ok",
            "message": "Trackpy optical flow computed",
            "shape": list(flows.shape),
            "method": "trackpy",
            "cache_key": cache_key,
            "trackpy_info": {
                "total_detections": info.get("total_detections", 0),
                "total_trajectories": info.get("total_trajectories", 0),
                "persistent_trajectories": info.get("persistent_trajectories", 0),
            }
        }
        
        return response
    
    def _preview_dog_detection(self, request):
        """Preview DoG (Difference of Gaussians) detection on a specified frame.
        
        Uses TrackMate-faithful implementation for exact compatibility.
        
        Request parameters (TrackMate-style):
            video_path: Path to video file
            output_path: Path to save preview image
            radius: Estimated object radius in pixels (default: 2.5)
                   Note: TrackMate UI shows "diameter" - we expect radius here
            threshold: Quality threshold, 0 = accept all (default: 0.0)
            median_filter: Pre-process with median filter (default: False)
            subpixel: Sub-pixel localization (default: True)
            frame: Frame index (0-indexed) to run detection on (default: 0)
            display_min/display_max: Display range for visualization
        """
        # Check for cancellation before starting
        self._check_cancelled("DoG detection preview")

        video_path = request["video_path"]
        output_path = request["output_path"]
        
        # TrackMate-style DoG parameters
        radius = request.get("radius", 2.5)
        threshold = request.get("threshold", 0.0)
        median_filter_enabled = request.get("median_filter", False)
        subpixel = request.get("subpixel", True)
        frame_idx = request.get("frame", 0)  # 0-indexed frame to detect on
        
        # Legacy parameter support
        if "dog_threshold" in request:
            threshold = request["dog_threshold"]
        
        # Display range for brightness adjustment (user's viewMin/viewMax from Java UI)
        display_min = request.get("display_min", None)
        display_max = request.get("display_max", None)
        
        print(f"=" * 60)
        print(f"TrackMate-faithful DoG Detection Preview")
        print(f"=" * 60)
        print(f"Video: {video_path}")
        print(f"Parameters:")
        print(f"  Radius: {radius} (diameter: {radius * 2})")
        print(f"  Threshold: {threshold}")
        print(f"  Median filter: {median_filter_enabled}")
        print(f"  Subpixel: {subpixel}")
        print(f"  Frame: {frame_idx}")
        if display_min is not None and display_max is not None:
            print(f"  Display range: {display_min} - {display_max}")
        
        start_time = _pc()
        
        try:
            # Add same directory to path to find trackmate_dog module
            script_dir = os.path.dirname(os.path.abspath(__file__))
            if script_dir not in sys.path:
                sys.path.insert(0, script_dir)
            
            from trackmate_dog import TrackMateDoGDetector
        except ImportError as e:
            return {
                "status": "error",
                "message": f"trackmate_dog module not available: {e}"
            }
        
        # Load video - RAW pixel values (no normalization, just like TrackMate)
        video_path_str = str(video_path)
        
        if video_path_str.lower().endswith('.avi'):
            # Load AVI using mediapy
            raw_video, v01, is_avi, orig_shape, resized_shape = self._load_video_raw_and_normalized(video_path)
            T, H, W, C = raw_video.shape
            # Clamp frame_idx to valid range
            frame_idx = int(np.clip(frame_idx, 0, T - 1))
            # For AVI, convert the specified frame to grayscale float (0-255 range)
            frame_for_dog = np.mean(raw_video[frame_idx], axis=-1).astype(np.float64)
            frame_rgb = raw_video[frame_idx].copy()
            # AVI files don't have calibration metadata - use pixels
            calibration = (1.0, 1.0)
        else:
            # Load TIFF - use RAW pixel values (exactly like TrackMate)
            import tifffile
            
            # Read TIFF metadata and only the requested frame (avoid loading full stacks).
            with tifffile.TiffFile(video_path_str) as tif:
                self._check_cancelled("DoG detection preview")

                # Extract calibration from ImageJ metadata (TrackMate reads this)
                # Default to 1.0 (pixel units) if not available
                pixel_size_x = 1.0
                pixel_size_y = 1.0

                if tif.imagej_metadata and 'scales' in tif.imagej_metadata:
                    scales = tif.imagej_metadata['scales']
                    # scales format is typically "X,Y,Z" or similar
                    if isinstance(scales, str):
                        scale_parts = [float(s) for s in scales.split(',') if s.strip()]
                        if len(scale_parts) >= 2:
                            pixel_size_x = scale_parts[0]
                            pixel_size_y = scale_parts[1]
                    elif hasattr(scales, '__iter__'):
                        scale_list = list(scales)
                        if len(scale_list) >= 2:
                            pixel_size_x = float(scale_list[0])
                            pixel_size_y = float(scale_list[1])

                # Also check resolution tags as fallback
                if pixel_size_x == 1.0 and pixel_size_y == 1.0:
                    page0 = tif.pages[0]
                    if page0.resolution and page0.resolutionunit == 3:  # 3 = centimeters
                        # resolution is pixels per cm, convert to microns per pixel
                        if page0.resolution[0] > 0 and page0.resolution[1] > 0:
                            # 1 cm = 10000 microns
                            pixel_size_x = 10000.0 / page0.resolution[0]
                            pixel_size_y = 10000.0 / page0.resolution[1]

                # Load only the requested frame. Most stacks are multi-page TIFF.
                n_pages = len(tif.pages)
                if n_pages > 1:
                    T = n_pages
                    frame_idx = int(np.clip(frame_idx, 0, T - 1))
                    frame_for_dog = tif.pages[frame_idx].asarray()
                else:
                    # Single-page TIFF could still be a 3D stack; fall back to full read.
                    raw_vol = tif.asarray()
                    # Handle RGB and orientation and then slice requested frame.
                    if raw_vol.ndim == 4:
                        if raw_vol.shape[-1] in (3, 4):
                            raw_vol = np.mean(raw_vol[..., :3], axis=-1)
                        elif raw_vol.shape[1] in (3, 4):
                            raw_vol = np.mean(raw_vol[:, :3, :, :], axis=1)
                        else:
                            raise ValueError(f"Unexpected 4D TIFF shape: {raw_vol.shape}")

                    if raw_vol.ndim == 3:
                        if raw_vol.shape[-1] <= 10 and raw_vol.shape[-1] < raw_vol.shape[0] and raw_vol.shape[-1] < raw_vol.shape[1]:
                            raw_vol = np.moveaxis(raw_vol, -1, 0)
                        T = raw_vol.shape[0]
                        frame_idx = int(np.clip(frame_idx, 0, T - 1))
                        frame_for_dog = raw_vol[frame_idx]
                    elif raw_vol.ndim == 2:
                        T = 1
                        frame_idx = 0
                        frame_for_dog = raw_vol
                    else:
                        raise ValueError(f"Unexpected TIFF array shape: {raw_vol.shape}")
            
            calibration = (pixel_size_y, pixel_size_x)  # (y, x) order for consistency
            print(f"  Image calibration: {pixel_size_x:.4f} x {pixel_size_y:.4f} units/pixel")
            
            # Normalize/convert to grayscale if needed
            if isinstance(frame_for_dog, np.ndarray) and frame_for_dog.ndim == 3 and frame_for_dog.shape[-1] in (3, 4):
                frame_for_dog = np.mean(frame_for_dog[..., :3], axis=-1)
            frame_for_dog = np.asarray(frame_for_dog).astype(np.float64)
            if frame_for_dog.ndim != 2:
                raise ValueError(f"Expected 2D frame for DoG preview, got shape {frame_for_dog.shape}")

            H, W = frame_for_dog.shape
            print(f"  Frame {frame_idx} stats: min={frame_for_dog.min():.2f}, max={frame_for_dog.max():.2f}, mean={frame_for_dog.mean():.2f}")
            
            # Convert grayscale frame to RGB for visualization
            if display_min is not None and display_max is not None and display_max > display_min:
                fg_min = display_min
                fg_max = display_max
                frame_normalized = np.clip((frame_for_dog - fg_min) / (fg_max - fg_min) * 255, 0, 255).astype(np.uint8)
            else:
                fg_min = frame_for_dog.min()
                fg_max = frame_for_dog.max()
                if fg_max > fg_min:
                    frame_normalized = ((frame_for_dog - fg_min) / (fg_max - fg_min) * 255).astype(np.uint8)
                else:
                    frame_normalized = (frame_for_dog * 255).astype(np.uint8)
            frame_rgb = np.stack([frame_normalized, frame_normalized, frame_normalized], axis=-1)
        
        # Create TrackMate-faithful detector
        # Use calibration from TIFF metadata (or 1.0 for AVI)
        detector = TrackMateDoGDetector(
            radius=radius,
            threshold=threshold,
            do_median_filter=median_filter_enabled,
            do_subpixel=subpixel,
            calibration=calibration,  # From TIFF metadata or (1.0, 1.0) for AVI
            match_trackmate_exactly=True  # Apply small threshold correction for exact match
        )
        
        # Run detection
        self._check_cancelled("DoG detection preview")
        spots = detector.process(frame_for_dog)
        n_keypoints = len(spots)
        
        if n_keypoints == 0:
            print(f"  Detected 0 spots on frame {frame_idx} (still writing preview)")
        else:
            print(f"  Detected {n_keypoints} spots on frame {frame_idx}")
        
        # Draw keypoints on preview
        import cv2
        np.random.seed(42)
        
        preview_frame = frame_rgb.copy()
        for i, spot in enumerate(spots):
            if i % 200 == 0:
                self._check_cancelled("DoG detection preview")
            x_int = int(round(spot['x_pixel']))
            y_int = int(round(spot['y_pixel']))
            
            # Generate color based on index
            hue = (i * 37) % 180
            color = cv2.cvtColor(np.array([[[hue, 255, 255]]], dtype=np.uint8), cv2.COLOR_HSV2RGB)[0, 0]
            color = tuple(int(c) for c in color)
            
            # Draw thin circle (ring only, no filled center)
            cv2.circle(preview_frame, (x_int, y_int), 6, color, 1)
        
        # Add text overlay with calibration info
        # Determine physical units display
        has_calibration = calibration[0] != 1.0 or calibration[1] != 1.0
        pixel_size = calibration[0]  # Assume square pixels, use Y calibration
        diameter_um = radius * 2 * pixel_size if has_calibration else None
        
        cv2.putText(preview_frame, f"Frame {frame_idx} - TrackMate DoG Detection", (10, 25),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(preview_frame, f"Spots detected: {n_keypoints}", (10, 50),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 255, 100), 1)
        
        # Show diameter in both pixels and physical units if calibration available
        if has_calibration:
            cv2.putText(preview_frame, f"Diameter: {radius * 2:.2f} px ({diameter_um:.2f} um)", (10, 70),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
            cv2.putText(preview_frame, f"Calibration: {pixel_size:.4f} um/px", (10, 90),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
            cv2.putText(preview_frame, f"Threshold: {threshold}, median: {median_filter_enabled}, subpixel: {subpixel}", (10, 110),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
        else:
            cv2.putText(preview_frame, f"Diameter: {radius * 2:.1f} px, threshold: {threshold}", (10, 70),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
            cv2.putText(preview_frame, f"median_filter: {median_filter_enabled}, subpixel: {subpixel}", (10, 90),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
        
        # Preview image is NOT saved to disk - data is returned to client for overlay
        
        elapsed = _pc() - start_time
        print(f"✓ Preview generated in {elapsed:.2f}s")
        
        # Compute sigma values for stats
        ndim = 2
        sigma_base = radius / np.sqrt(ndim)
        sigma_low = sigma_base * 0.9
        sigma_high = sigma_base * 1.1
        
        # Build spot list for overlay in main window
        spot_list = []
        for spot in spots:
            spot_list.append({
                "x": float(spot['x_pixel']),
                "y": float(spot['y_pixel']),
                "quality": float(spot.get('quality', 0.0))
            })
        
        return {
            "status": "ok",
            "message": "TrackMate-faithful DoG detection preview generated",
            "output_path": output_path,
            "spots": spot_list,  # Return spots for overlay in main window
            "statistics": {
                "total_frames": int(T),
                "total_keypoints": int(n_keypoints),
                "sigma_low": sigma_low,
                "sigma_high": sigma_high,
                "threshold": threshold,
                "image_width": int(W),
                "image_height": int(H),
                "calibration_x": float(calibration[1]),
                "calibration_y": float(calibration[0]),
            }
        }
    
    def _preview_trackpy_detection(self, request):
        """Preview DoG particle detection for trackpy flow computation.
        
        Uses the SAME DoG detector as LocoTrack for consistent particle detection.
        This ensures that both flow methods detect the same particles.
        
        Request parameters (same as LocoTrack DoG preview):
            video_path: Path to video file
            output_path: Path to save preview image
            radius: Estimated object radius in pixels (default: 2.5)
            threshold: Quality threshold, 0 = accept all (default: 0.0)
            median_filter: Pre-process with median filter (default: False)
            subpixel: Sub-pixel localization (default: True)
            frame: Frame index (0-indexed) to run detection on (default: 0)
            display_min/display_max: Display range for visualization
            invert: Detect dark particles (default: False)
            
            # Legacy parameters (for backward compatibility):
            diameter: If provided, converts to radius = diameter / 2
        """
        # Check for cancellation before starting
        self._check_cancelled("Trackpy DoG detection preview")

        video_path = request["video_path"]
        output_path = request["output_path"]
        
        # DoG detection parameters (shared with LocoTrack)
        radius = request.get("radius", 2.5)
        threshold = request.get("threshold", 0.0)
        median_filter_enabled = request.get("median_filter", False)
        subpixel = request.get("subpixel", True)
        frame_idx = request.get("frame", 0)
        invert = request.get("invert", False)
        
        # Display range for brightness adjustment
        display_min = request.get("display_min", None)
        display_max = request.get("display_max", None)
        
        # Legacy diameter parameter support
        if "diameter" in request and "radius" not in request:
            radius = float(request["diameter"]) / 2.0
            print(f"[TrackpyPreview] Converting legacy diameter={request['diameter']} to radius={radius}")
        
        print(f"=" * 60)
        print(f"Trackpy DoG Detection Preview (TrackMate-style)")
        print(f"=" * 60)
        print(f"Video: {video_path}")
        print(f"Parameters:")
        print(f"  Radius: {radius} (diameter: {radius * 2})")
        print(f"  Threshold: {threshold}")
        print(f"  Median filter: {median_filter_enabled}")
        print(f"  Subpixel: {subpixel}")
        print(f"  Frame: {frame_idx}")
        
        start_time = _pc()
        
        try:
            # Add same directory to path to find trackmate_dog module
            script_dir = os.path.dirname(os.path.abspath(__file__))
            if script_dir not in sys.path:
                sys.path.insert(0, script_dir)
            
            from trackmate_dog import TrackMateDoGDetector
        except ImportError as e:
            return {
                "status": "error",
                "message": f"trackmate_dog module not available: {e}"
            }
        
        # Load video - RAW pixel values (no normalization, just like TrackMate)
        video_path_str = str(video_path)
        
        if video_path_str.lower().endswith('.avi'):
            # Load AVI using mediapy
            raw_video, v01, is_avi, orig_shape, resized_shape = self._load_video_raw_and_normalized(video_path)
            T, H, W, C = raw_video.shape
            frame_idx = int(np.clip(frame_idx, 0, T - 1))
            frame_for_dog = np.mean(raw_video[frame_idx], axis=-1).astype(np.float64)
            frame_rgb = raw_video[frame_idx].copy()
            calibration = (1.0, 1.0)
        else:
            # Load TIFF - use RAW pixel values, but only load the requested frame.
            import tifffile
            with tifffile.TiffFile(video_path_str) as tif:
                self._check_cancelled("Trackpy DoG detection preview")
                n_pages = len(tif.pages)
                if n_pages > 1:
                    T = n_pages
                    frame_idx = int(np.clip(frame_idx, 0, T - 1))
                    frame_for_dog = tif.pages[frame_idx].asarray()
                else:
                    raw_vol = tif.asarray()
                    if raw_vol.ndim == 4:
                        if raw_vol.shape[-1] in (3, 4):
                            raw_vol = np.mean(raw_vol[..., :3], axis=-1)
                        elif raw_vol.shape[1] in (3, 4):
                            raw_vol = np.mean(raw_vol[:, :3, :, :], axis=1)
                        else:
                            raise ValueError(f"Unexpected 4D TIFF shape: {raw_vol.shape}")

                    if raw_vol.ndim == 3:
                        if raw_vol.shape[-1] <= 10 and raw_vol.shape[-1] < raw_vol.shape[0] and raw_vol.shape[-1] < raw_vol.shape[1]:
                            raw_vol = np.moveaxis(raw_vol, -1, 0)
                        T = raw_vol.shape[0]
                        frame_idx = int(np.clip(frame_idx, 0, T - 1))
                        frame_for_dog = raw_vol[frame_idx]
                    elif raw_vol.ndim == 2:
                        T = 1
                        frame_idx = 0
                        frame_for_dog = raw_vol
                    else:
                        raise ValueError(f"Unexpected TIFF array shape: {raw_vol.shape}")

            if isinstance(frame_for_dog, np.ndarray) and frame_for_dog.ndim == 3 and frame_for_dog.shape[-1] in (3, 4):
                frame_for_dog = np.mean(frame_for_dog[..., :3], axis=-1)

            frame_for_dog = np.asarray(frame_for_dog).astype(np.float64)
            if frame_for_dog.ndim != 2:
                raise ValueError(f"Expected 2D frame after preprocessing, got shape {frame_for_dog.shape}")

            H, W = frame_for_dog.shape
            calibration = (1.0, 1.0)
            
            # Convert grayscale frame to RGB for visualization
            if display_min is not None and display_max is not None and display_max > display_min:
                frame_normalized = np.clip((frame_for_dog - display_min) / (display_max - display_min) * 255, 0, 255).astype(np.uint8)
            else:
                fg_min = frame_for_dog.min()
                fg_max = frame_for_dog.max()
                if fg_max > fg_min:
                    frame_normalized = ((frame_for_dog - fg_min) / (fg_max - fg_min) * 255).astype(np.uint8)
                else:
                    frame_normalized = (frame_for_dog * 255).astype(np.uint8)
            frame_rgb = np.stack([frame_normalized, frame_normalized, frame_normalized], axis=-1)
        
        # Create TrackMate-faithful detector (SAME as LocoTrack)
        detector = TrackMateDoGDetector(
            radius=radius,
            threshold=threshold,
            do_median_filter=median_filter_enabled,
            do_subpixel=subpixel,
            calibration=calibration,
            match_trackmate_exactly=True
        )
        
        # Run detection
        self._check_cancelled("Trackpy DoG detection preview")
        spots = detector.process(frame_for_dog)
        n_keypoints = len(spots)
        
        print(f"  Detected {n_keypoints} spots on frame {frame_idx}")
        
        # Draw keypoints on preview
        import cv2
        np.random.seed(42)
        
        preview_frame = frame_rgb.copy()
        for i, spot in enumerate(spots):
            if i % 200 == 0:
                self._check_cancelled("Trackpy DoG detection preview")
            x_int = int(round(spot['x_pixel']))
            y_int = int(round(spot['y_pixel']))
            
            # Generate color based on index
            hue = (i * 37) % 180
            color = cv2.cvtColor(np.array([[[hue, 255, 255]]], dtype=np.uint8), cv2.COLOR_HSV2RGB)[0, 0]
            color = tuple(int(c) for c in color)
            
            # Draw thin circle (ring only, no filled center)
            cv2.circle(preview_frame, (x_int, y_int), 6, color, 1)
        
        # Add text overlay
        cv2.putText(preview_frame, f"Frame {frame_idx} - Trackpy DoG Detection", (10, 25),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(preview_frame, f"Spots detected: {n_keypoints}", (10, 50),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 255, 100), 1)
        cv2.putText(preview_frame, f"Diameter: {radius * 2:.1f} px, threshold: {threshold}", (10, 70),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
        cv2.putText(preview_frame, f"median_filter: {median_filter_enabled}, subpixel: {subpixel}", (10, 90),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
        
        # Preview image is NOT saved to disk - data is returned to client for overlay
        
        elapsed = _pc() - start_time
        print(f"✓ Preview generated in {elapsed:.2f}s")
        
        # Compute sigma values for stats
        ndim = 2
        sigma_base = radius / np.sqrt(ndim)
        sigma_low = sigma_base * 0.9
        sigma_high = sigma_base * 1.1
        
        # Build spot list for overlay in main window
        spot_list = []
        for spot in spots:
            spot_list.append({
                "x": float(spot['x_pixel']),
                "y": float(spot['y_pixel']),
                "quality": float(spot.get('quality', 0.0))
            })
        
        return {
            "status": "ok",
            "message": "Trackpy DoG detection preview generated",
            "output_path": output_path,
            "spots": spot_list,
            "statistics": {
                "total_frames": int(T),
                "total_keypoints": int(n_keypoints),
                "sigma_low": sigma_low,
                "sigma_high": sigma_high,
                "threshold": threshold,
                "image_width": int(W),
                "image_height": int(H),
            }
        }
    
    def _preview_trackpy_trajectories(self, request):
        """Preview trackpy particle detection AND trajectory linking.
        
        Unlike DoG preview which only shows detections on one frame, this:
        1. Runs trackpy native detection on ALL frames
        2. Links detections into trajectories  
        3. Returns full trajectory data so Java can overlay on any frame
        
        This allows the user to freely navigate frames while seeing trajectories.
        
        Request parameters:
            video_path: Path to video file
            diameter: Particle diameter in pixels (must be odd)
            minmass: Minimum integrated brightness (0 = auto)
            search_range: Maximum displacement between frames
            memory: Frames particle can disappear
            percentile: Brightness percentile threshold (default: 64)
            max_frames: Maximum frames to process for preview (default: all, 0 = all)
        
        Returns:
            trajectories: List of trajectory dicts with frame-by-frame positions
            detections_per_frame: Dict mapping frame -> list of (x, y) detections
            statistics: Summary statistics
        """
        import trackpy as tp
        tp.quiet()
        
        video_path = request["video_path"]
        
        # Trackpy detection parameters
        diameter = int(request.get("diameter", 11))
        if diameter % 2 == 0:
            diameter += 1  # Must be odd
        minmass = request.get("minmass", 0)
        search_range = float(request.get("search_range", 15))
        memory = int(request.get("memory", 5))
        percentile = float(request.get("percentile", 64))
        max_frames = int(request.get("max_frames", 0))  # 0 = all frames
        require_persistent = request.get("require_persistent", False)
        if isinstance(require_persistent, str):
            require_persistent = require_persistent.lower() in ('true', '1', 'yes')
        
        print(f"=" * 60)
        print(f"Trackpy Trajectory Preview")
        print(f"=" * 60)
        print(f"Video: {video_path}")
        print(f"Parameters:")
        print(f"  Diameter: {diameter}")
        print(f"  Minmass: {minmass}")
        print(f"  Search range: {search_range}")
        print(f"  Memory: {memory}")
        print(f"  Percentile: {percentile}")
        print(f"  Require persistent: {require_persistent}")

        # Responsive cancellation
        self._check_cancelled("Trackpy trajectory preview")
        
        start_time = _pc()
        
        # Load video
        video_path_str = str(video_path)

        self._check_cancelled("Trackpy trajectory preview")
        
        if video_path_str.lower().endswith('.avi'):
            raw_video, v01, is_avi, orig_shape, resized_shape = self._load_video_raw_and_normalized(video_path)
            T, H, W, C = raw_video.shape
            # Convert to grayscale for trackpy
            video_gray = np.mean(raw_video, axis=-1).astype(np.float64)
        else:
            # Use TiffFile to reliably load multi-page stacks.
            import tifffile
            with tifffile.TiffFile(video_path_str) as tif:
                self._check_cancelled("Trackpy trajectory preview")
                raw_vol = tif.asarray()

                # Some TIFFs are written as many independent pages/series; in that
                # case tif.asarray() may return just the first 2D page.
                if raw_vol.ndim == 2 and len(tif.pages) > 1:
                    n_pages = len(tif.pages)
                    n = n_pages
                    if max_frames > 0:
                        n = min(n, max_frames)

                    frames = []
                    for i in range(n):
                        if i % 10 == 0:
                            self._check_cancelled("Trackpy trajectory preview")
                        frames.append(tif.pages[i].asarray())
                    raw_vol = np.stack(frames, axis=0)

            # Handle RGB stacks (T, H, W, C) -> grayscale
            if raw_vol.ndim == 4:
                if raw_vol.shape[-1] in (3, 4):
                    raw_vol = np.mean(raw_vol[..., :3], axis=-1)
                elif raw_vol.shape[1] in (3, 4):
                    raw_vol = np.mean(raw_vol[:, :3, :, :], axis=1)
                else:
                    raise ValueError(f"Unexpected 4D TIFF shape: {raw_vol.shape}")

            # Normalize orientation if stored as (H, W, T)
            if raw_vol.ndim == 3 and raw_vol.shape[-1] <= 10 and raw_vol.shape[-1] < raw_vol.shape[0] and raw_vol.shape[-1] < raw_vol.shape[1]:
                raw_vol = np.moveaxis(raw_vol, -1, 0)

            if raw_vol.ndim == 2:
                raw_vol = raw_vol[None, ...]

            video_gray = raw_vol.astype(np.float64)
            T, H, W = video_gray.shape
        
        # Limit frames for faster preview if requested
        if max_frames > 0 and max_frames < T:
            video_gray = video_gray[:max_frames]
            T = max_frames
            print(f"  Limited to first {max_frames} frames for preview")
        
        print(f"  Video: {T} frames, {H}x{W} pixels")
        
        # Step 1: Detect particles using trackpy's native detection
        print(f"[TrackpyPreview] Detecting particles...")

        self._check_cancelled("Trackpy trajectory preview")
        
        # Auto-compute minmass if 0
        if minmass == 0:
            sample_frame = video_gray[T // 2]
            minmass = np.percentile(sample_frame, 90) * (diameter ** 2) * 0.1
            print(f"  Auto minmass: {minmass:.1f}")
        
        features = tp.batch(
            video_gray,
            diameter=diameter,
            minmass=minmass,
            percentile=percentile,
            processes='auto'
        )

        self._check_cancelled("Trackpy trajectory preview")
        
        n_detections = len(features)
        print(f"  Total detections: {n_detections} across {T} frames")
        
        if n_detections == 0:
            return {
                "status": "ok",
                "message": "No particles detected",
                "trajectories": [],
                "detections_per_frame": {},
                "statistics": {
                    "total_frames": T,
                    "total_detections": 0,
                    "total_trajectories": 0,
                    "image_width": W,
                    "image_height": H
                }
            }
        
        # Step 2: Link trajectories
        print(f"[TrackpyPreview] Linking trajectories...")

        self._check_cancelled("Trackpy trajectory preview")
        
        # Use adaptive linking to handle dense regions
        adaptive_stop = max(1.0, search_range / 5.0)
        
        try:
            trajectories = tp.link(
                features,
                search_range=search_range,
                memory=memory,
                adaptive_stop=adaptive_stop,
                adaptive_step=0.95
            )
        except Exception as e:
            if "Subnetwork" in str(e):
                # Reduce search range
                reduced_range = search_range / 2.0
                print(f"  Network too complex, reducing search_range to {reduced_range}")
                trajectories = tp.link(
                    features,
                    search_range=reduced_range,
                    memory=memory,
                    adaptive_stop=max(1.0, reduced_range / 3.0),
                    adaptive_step=0.9
                )
            else:
                raise
        
        n_trajectories = trajectories['particle'].nunique()
        print(f"  Total trajectories: {n_trajectories}")

        self._check_cancelled("Trackpy trajectory preview")
        
        # Filter to persistent trajectories if requested
        n_persistent = 0
        if require_persistent:
            # Keep only trajectories that appear in all frames
            traj_lengths = trajectories.groupby('particle')['frame'].count()
            persistent_particles = traj_lengths[traj_lengths == T].index
            n_persistent = len(persistent_particles)
            print(f"  Persistent trajectories (spanning all {T} frames): {n_persistent}")
            trajectories = trajectories[trajectories['particle'].isin(persistent_particles)]
            if len(trajectories) == 0:
                print(f"  Warning: No persistent trajectories found!")
        
        # Build trajectory data structure for Java
        # Format: list of trajectories, each with frame -> (x, y) mapping
        trajectory_list = []
        particle_ids = trajectories['particle'].unique()
        
        for pid in particle_ids:
            # Avoid UI feeling "stuck" on large trajectory sets
            if len(trajectory_list) % 50 == 0:
                self._check_cancelled("Trackpy trajectory preview")
            traj_data = trajectories[trajectories['particle'] == pid]
            frames = traj_data['frame'].values.astype(int).tolist()
            x_vals = traj_data['x'].values.tolist()
            y_vals = traj_data['y'].values.tolist()
            
            # Build frame -> position mapping
            positions = {}
            for f, x, y in zip(frames, x_vals, y_vals):
                positions[str(f)] = {"x": x, "y": y}
            
            trajectory_list.append({
                "id": int(pid),
                "positions": positions,
                "length": len(frames),
                "start_frame": min(frames),
                "end_frame": max(frames)
            })
        
        # Also build detections per frame for quick lookup
        detections_per_frame = {}
        for f in range(T):
            if f % 10 == 0:
                self._check_cancelled("Trackpy trajectory preview")
            frame_data = features[features['frame'] == f]
            detections_per_frame[str(f)] = [
                {"x": row['x'], "y": row['y'], "mass": row.get('mass', 0)}
                for _, row in frame_data.iterrows()
            ]
        
        # Compute trajectory length statistics
        traj_lengths = trajectories.groupby('particle')['frame'].count()
        
        elapsed = _pc() - start_time
        print(f"✓ Trajectory preview generated in {elapsed:.2f}s")
        
        return {
            "status": "ok",
            "message": f"Detected {n_detections} particles, linked into {n_trajectories} trajectories",
            "trajectories": trajectory_list,
            "detections_per_frame": detections_per_frame,
            "statistics": {
                "total_frames": T,
                "total_detections": n_detections,
                "total_trajectories": n_trajectories,
                "avg_detections_per_frame": n_detections / T,
                "trajectory_length_min": int(traj_lengths.min()),
                "trajectory_length_max": int(traj_lengths.max()),
                "trajectory_length_median": float(traj_lengths.median()),
                "image_width": W,
                "image_height": H,
                "diameter": diameter,
                "minmass": minmass,
                "search_range": search_range,
                "memory": memory,
                "require_persistent": require_persistent,
                "persistent_trajectories": n_persistent if require_persistent else n_trajectories
            }
        }
    
    def stop(self):
        """Stop the server."""
        self.running = False
        if self.server_socket:
            self.server_socket.close()
        # Only try to unlink socket file for Unix socket mode
        if not self.use_tcp and self.socket_path and os.path.exists(self.socket_path):
            os.unlink(self.socket_path)


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Persistent RAFT Tracking Server")
    parser.add_argument(
        "--socket",
        default="/tmp/ripple-env.sock",
        help="Unix socket path (for Linux/Mac)"
    )
    parser.add_argument(
        "--tcp-host",
        default=None,
        help="TCP host to bind to (for Windows). If specified, TCP mode is used instead of Unix socket."
    )
    parser.add_argument(
        "--tcp-port",
        type=int,
        default=9876,
        help="TCP port to bind to (default: 9876, only used with --tcp-host)"
    )
    parser.add_argument(
        "--model",
        choices=["large", "small"],
        default="large",
        help="RAFT model size"
    )
    parser.add_argument(
        "--device",
        choices=["cuda", "cpu", "auto"],
        default="auto",
        help="Device to use"
    )
    parser.add_argument(
        "--mode",
        choices=["gpu", "cpu", "auto"],
        default="auto",
        help="Execution mode: gpu (full features), cpu (DIS flow only), auto (detect)"
    )
    parser.add_argument(
        "--gpu",
        type=int,
        default=None,
        help="GPU index to use (0, 1, 2, etc.). Sets CUDA_VISIBLE_DEVICES. If not specified, uses all available GPUs."
    )
    
    args = parser.parse_args()
    
    # Note: GPU selection via --gpu is handled in _early_gpu_selection() at module load time
    # (CUDA_VISIBLE_DEVICES must be set before `import torch`)
    
    # Determine execution mode
    execution_mode = args.mode
    if execution_mode == "auto":
        execution_mode = "gpu" if torch.cuda.is_available() else "cpu"
    
    # Setup device based on execution mode
    if args.device == "auto":
        if execution_mode == "gpu" and torch.cuda.is_available():
            device = torch.device("cuda")
        else:
            device = torch.device("cpu")
    else:
        device = torch.device(args.device)
    
    print("=" * 60)
    print("PERSISTENT TRACKING SERVER")
    print("=" * 60)
    print(f"Execution Mode: {execution_mode.upper()}")
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA version: {torch.version.cuda}")
        print(f"CUDA device count: {torch.cuda.device_count()}")
        print(f"CUDA device name: {torch.cuda.get_device_name(0)}")
        print(f"CUDA memory allocated: {torch.cuda.memory_allocated(0) / 1024**2:.1f} MB")
        if _SELECTED_GPU is not None:
            print(f"Selected GPU: {_SELECTED_GPU} (via --gpu argument)")
    
    if execution_mode == "cpu":
        print("")
        print("CPU MODE: Using DIS optical flow and TrackPy")
        print("  - RAFT/LocoTrack disabled (requires GPU)")
        print("  - TrackMate-style DoG detection available")
        print("  - DIS optical flow available (fast but less accurate)")
    else:
        print("")
        print("GPU MODE: Full functionality available")
        print("  - RAFT optical flow")
        print("  - LocoTrack point tracking")
        print("  - TrackMate-style DoG detection")
        print("  - DIS optical flow fallback")
    
    print(f"Selected device: {device}")
    if execution_mode == "gpu":
        print(f"Model: RAFT {args.model}")
    
    # Determine if using TCP mode (Windows) or Unix socket mode (Linux/Mac)
    use_tcp = args.tcp_host is not None
    if use_tcp:
        print(f"Socket: TCP {args.tcp_host}:{args.tcp_port}")
    else:
        print(f"Socket: {args.socket}")
    print("=" * 60)
    
    # Create model manager based on execution mode
    # In CPU mode, we create a dummy manager that uses DIS flow
    if execution_mode == "gpu":
        model_manager = RAFTModelManager(device=device, model_size=args.model)
    else:
        # CPU mode: Create a lightweight manager for DIS-only operation
        model_manager = CPUModelManager(device=device)
    
    # Create and start server
    if use_tcp:
        server = TrackingServer(None, model_manager, tcp_host=args.tcp_host, tcp_port=args.tcp_port)
    else:
        server = TrackingServer(args.socket, model_manager)
    server.execution_mode = execution_mode  # Store mode for request handling
    
    # Handle shutdown gracefully with proper GPU cleanup
    def signal_handler(sig, frame):
        print("\nShutting down server...")
        server.stop()
        
        # Clean up GPU memory before exit
        print("Cleaning up GPU memory...")
        try:
            # Delete model to free GPU memory
            if hasattr(model_manager, 'model'):
                del model_manager.model
            cleanup_gpu_memory()
            if torch.cuda.is_available():
                final_mem = torch.cuda.memory_allocated(0) / 1024**2
                print(f"  GPU memory after cleanup: {final_mem:.1f} MB")
        except Exception as e:
            print(f"  Warning: GPU cleanup error: {e}")
        
        sys.exit(0)
    
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    # Register atexit handler as a fallback for cleanup on any exit
    def atexit_cleanup():
        """Emergency cleanup - ensures GPU memory is freed even on crashes."""
        try:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                gc.collect()
        except Exception:
            pass
    
    atexit.register(atexit_cleanup)
    
    # Start server (blocks)
    server.start()


if __name__ == "__main__":
    main()