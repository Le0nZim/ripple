#!/usr/bin/env python3
"""Trackpy-based Optical Flow Generation for Fluorescent Microscopy.

This module provides an alternative optical flow computation method based on 
DoG (Difference of Gaussians) particle detection and trackpy trajectory linking,
designed specifically for fluorescent microscopy images where traditional optical 
flow methods (like RAFT) may struggle due to:
- Sparse, bright point sources
- High noise backgrounds
- Photobleaching and intensity fluctuations
- Intermittent particle visibility

Key Features:
1. DoG-based particle detection (shared with LocoTrack for consistency)
2. Trajectory linking with trackpy's robust linker (search radius and memory)
3. Filtering of trajectories to keep only persistent (omni-present) tracks
4. Spectral interpolation for smooth trajectory reconstruction
5. Conversion from sparse trajectories to dense optical flow field

The approach:
1. Use DoG detector to find particles in each frame (same as LocoTrack)
2. Link detections across frames into trajectories using trackpy
3. Filter out trajectories that don't span the entire video (persistent only)
4. Interpolate trajectories spectrally for smooth temporal reconstruction
5. Generate dense optical flow field via spatial interpolation (RBF/IDW)

Dependencies:
    - trackpy: Trajectory linking (tp.link)
    - scipy: DoG detection, interpolation, and spatial algorithms
    - numpy: Numerical operations
    - torch: PyTorch for GPU-accelerated interpolation (optional)
"""

import numpy as np
from typing import Tuple, Optional, Dict, List, Union, Callable
from functools import lru_cache
import warnings
import os
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import multiprocessing

# Get number of CPUs for parallel processing
NUM_CPUS = multiprocessing.cpu_count()


def _maybe_cancel(cancel_check: Optional[Callable[[], None]]) -> None:
    if cancel_check is not None:
        cancel_check()

try:
    import trackpy as tp
    TRACKPY_AVAILABLE = True
    # Check if numba is available for trackpy acceleration
    try:
        import numba
        NUMBA_AVAILABLE = True
    except ImportError:
        NUMBA_AVAILABLE = False
        warnings.warn("numba not installed. trackpy will run slower. Install with: pip install numba")
except ImportError:
    TRACKPY_AVAILABLE = False
    NUMBA_AVAILABLE = False
    warnings.warn("trackpy not installed. Install with: pip install trackpy")

try:
    import pandas as pd
    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False
    warnings.warn("pandas not installed. Install with: pip install pandas")

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

from scipy.interpolate import RBFInterpolator
from scipy.fft import fft, ifft, fftfreq, dct, idct
from scipy.ndimage import gaussian_filter, maximum_filter, median_filter as scipy_median_filter


# =============================================================================
# DoG (DIFFERENCE OF GAUSSIANS) PARTICLE DETECTION
# Shared implementation with LocoTrack for consistent detection
# =============================================================================

def find_local_maxima_trackmate_style(dog: np.ndarray, threshold: float, exclude_border: bool = False) -> np.ndarray:
    """
    Find local maxima in TrackMate style - no min_distance suppression.
    
    TrackMate uses RectangleShape(1, true) which creates a 3x3 neighborhood
    and finds pixels that are greater than or equal to ALL 8 neighbors AND above threshold.
    
    Args:
        dog: 2D DoG-filtered image
        threshold: Minimum value for a pixel to be considered a maximum
        exclude_border: If True, exclude 1-pixel border (default: False to match TrackMate)
        
    Returns:
        (N, 2) array of (y, x) coordinates
    """
    # Find all pixels that are local maxima using maximum_filter with 3x3 footprint
    neighborhood = np.ones((3, 3), dtype=bool)
    neighborhood[1, 1] = False  # Exclude center
    
    # Get the maximum value in each 3x3 neighborhood (excluding center)
    local_max_values = maximum_filter(dog, footprint=neighborhood, mode='mirror')
    
    # A pixel is a local maximum if it's >= all neighbors AND >= threshold
    is_local_max = (dog >= local_max_values) & (dog >= threshold)
    
    if exclude_border:
        is_local_max[0, :] = False
        is_local_max[-1, :] = False
        is_local_max[:, 0] = False
        is_local_max[:, -1] = False
    
    y_coords, x_coords = np.where(is_local_max)
    
    if len(y_coords) == 0:
        return np.zeros((0, 2), dtype=np.int64)
    
    return np.stack([y_coords, x_coords], axis=-1)


class DoGDetector:
    """
    Difference of Gaussians detector matching TrackMate's implementation.
    
    This is the SAME detector used by LocoTrack, ensuring consistent particle
    detection across both flow computation methods.
    """
    
    def __init__(self, radius: float, threshold: float,
                 do_median_filter: bool = False, do_subpixel: bool = True):
        """
        Args:
            radius: Estimated object radius in pixels
            threshold: Quality threshold for detection (0 = accept all)
            do_median_filter: Apply 3x3 median filter before detection
            do_subpixel: Refine positions with quadratic fitting
        """
        self.radius = radius
        self.threshold = threshold
        self.do_median_filter = do_median_filter
        self.do_subpixel = do_subpixel
        
        # TrackMate's sigma calculation for 2D images
        ndim = 2
        sigma1_nominal = self.radius / np.sqrt(ndim) * 0.9
        sigma2_nominal = self.radius / np.sqrt(ndim) * 1.1
        
        # TrackMate applies imageSigma correction (assumes inherent blur of 0.5 pixels)
        imageSigma = 0.5
        self.sigma1 = np.sqrt(max(sigma1_nominal**2 - imageSigma**2, 0.01))
        self.sigma2 = np.sqrt(max(sigma2_nominal**2 - imageSigma**2, 0.01))
    
    def apply_median_filter(self, image: np.ndarray) -> np.ndarray:
        """Apply 3x3 median filter (matching TrackMate's implementation)."""
        return scipy_median_filter(image, size=3, mode='constant', cval=0)
    
    def compute_dog(self, image: np.ndarray) -> np.ndarray:
        """Compute Difference of Gaussians: dog = gauss(sigma1) - gauss(sigma2)"""
        gauss1 = gaussian_filter(image.astype(np.float64), sigma=self.sigma1)
        gauss2 = gaussian_filter(image.astype(np.float64), sigma=self.sigma2)
        return gauss1 - gauss2
    
    def refine_subpixel(self, dog: np.ndarray, coordinates: np.ndarray) -> np.ndarray:
        """Refine spot positions using quadratic fitting."""
        refined = []
        
        for (y, x) in coordinates:
            if y < 1 or y >= dog.shape[0] - 1 or x < 1 or x >= dog.shape[1] - 1:
                refined.append([float(y), float(x)])
                continue
            
            neighborhood = dog[y-1:y+2, x-1:x+2].astype(np.float64)
            
            # Quadratic fitting for sub-pixel localization
            dy = 0.0
            denom_y = 2 * neighborhood[1, 1] - neighborhood[0, 1] - neighborhood[2, 1]
            if abs(denom_y) > 1e-10:
                dy = (neighborhood[0, 1] - neighborhood[2, 1]) / (2 * denom_y)
                dy = np.clip(dy, -0.5, 0.5)
            
            dx = 0.0
            denom_x = 2 * neighborhood[1, 1] - neighborhood[1, 0] - neighborhood[1, 2]
            if abs(denom_x) > 1e-10:
                dx = (neighborhood[1, 0] - neighborhood[1, 2]) / (2 * denom_x)
                dx = np.clip(dx, -0.5, 0.5)
            
            refined.append([float(y) + dy, float(x) + dx])
        
        return np.array(refined) if refined else np.array([]).reshape(0, 2)
    
    def detect(self, image: np.ndarray, exclude_border: bool = False, 
               max_keypoints: Optional[int] = None, invert: bool = False) -> np.ndarray:
        """
        Detect spots in image using TrackMate-style DoG detection.
        
        Args:
            image: 2D grayscale image
            exclude_border: Exclude keypoints at image border
            max_keypoints: Maximum number of keypoints (by quality)
            invert: If True, detect dark blobs on light background
        
        Returns:
            coordinates: (N, 2) array of (y, x) positions (sub-pixel if enabled)
        """
        preprocessed = image.astype(np.float64).copy()
        
        if invert:
            preprocessed = 1.0 - preprocessed if preprocessed.max() <= 1.0 else 255.0 - preprocessed
        
        if self.do_median_filter:
            preprocessed = self.apply_median_filter(preprocessed)
        
        dog = self.compute_dog(preprocessed)
        
        # Find maxima using TrackMate-style detection
        coordinates = find_local_maxima_trackmate_style(dog, self.threshold, exclude_border)
        
        # If max_keypoints is set, sort by quality and take top N
        if max_keypoints is not None and len(coordinates) > max_keypoints:
            qualities = dog[coordinates[:, 0], coordinates[:, 1]]
            top_indices = np.argsort(qualities)[::-1][:max_keypoints]
            coordinates = coordinates[top_indices]
        
        # Sub-pixel refinement
        if self.do_subpixel and len(coordinates) > 0:
            coordinates = self.refine_subpixel(dog, coordinates)
        else:
            coordinates = coordinates.astype(np.float64)
        
        return coordinates


def detect_particles_dog_batch(
    video: np.ndarray,
    radius: float = 2.5,
    threshold: float = 0.0,
    median_filter: bool = False,
    subpixel: bool = True,
    exclude_border: bool = False,
    max_keypoints_per_frame: Optional[int] = None,
    invert: bool = False,
    verbose: bool = True,
    cancel_check: Optional[Callable[[], None]] = None,
) -> "pd.DataFrame":
    """Detect particles in all frames using DoG detector.
    
    This replaces trackpy's tp.batch with DoG-based detection,
    providing the same output format (DataFrame with x, y, frame columns)
    for compatibility with trackpy's linker.
    
    Args:
        video: 3D video array (T, H, W)
        radius: Estimated object radius in pixels
        threshold: Quality threshold (0 = accept all)
        median_filter: Apply median filter before detection
        subpixel: Use sub-pixel localization
        exclude_border: Exclude detections at image border
        max_keypoints_per_frame: Maximum detections per frame
        invert: Detect dark blobs on light background
        verbose: Print progress
        
    Returns:
        DataFrame with columns: x, y, frame (compatible with trackpy linker)
    """
    if not PANDAS_AVAILABLE:
        raise ImportError("pandas is required")
    
    T, H, W = video.shape
    
    detector = DoGDetector(
        radius=radius,
        threshold=threshold,
        do_median_filter=median_filter,
        do_subpixel=subpixel
    )
    
    all_features = []
    
    for t in range(T):
        if t % 10 == 0:
            _maybe_cancel(cancel_check)
        frame = video[t]
        
        # Detect spots in this frame
        coords = detector.detect(frame, exclude_border=exclude_border, 
                                 max_keypoints=max_keypoints_per_frame, invert=invert)
        
        # Convert to DataFrame format (x, y, frame) - matching trackpy's output format
        for i, (y, x) in enumerate(coords):
            all_features.append({
                'x': float(x),
                'y': float(y),
                'frame': int(t),  # Must be integer for trackpy linker
                'mass': 1.0,  # Placeholder for compatibility
            })
        
        if verbose and (t + 1) % 50 == 0:
            print(f"  [DoG] Processed frame {t+1}/{T}, detected {len(coords)} particles")
    
    if verbose:
        print(f"  [DoG] Total detections: {len(all_features)} across {T} frames")
    
    # Create DataFrame with proper dtypes matching trackpy's output
    if len(all_features) == 0:
        return pd.DataFrame(columns=['x', 'y', 'frame', 'mass'])
    
    df = pd.DataFrame(all_features)
    df['frame'] = df['frame'].astype(int)  # Ensure integer frame column
    return df


# =============================================================================
# LEGACY TRACKPY DETECTION (kept for backward compatibility)
# =============================================================================

def detect_particles_single_frame(
    frame: np.ndarray,
    diameter: int = 11,
    minmass: Optional[float] = None,
    separation: Optional[int] = None,
    percentile: float = 64,
    invert: bool = False,
    **kwargs
) -> "pd.DataFrame":
    """Detect particles in a single frame using trackpy.
    
    Args:
        frame: 2D grayscale image (H, W)
        diameter: Estimated particle diameter in pixels (must be odd)
        minmass: Minimum integrated brightness. If None, auto-computed.
        separation: Minimum separation between features. Default: diameter + 1
        percentile: Features must have a peak brighter than this percentile.
        invert: Set True if particles are dark on light background.
        **kwargs: Additional arguments passed to tp.locate
        
    Returns:
        DataFrame with columns: x, y, mass, size, ecc, signal, raw_mass, ep, frame
    """
    if not TRACKPY_AVAILABLE:
        raise ImportError("trackpy is required for particle detection")
    
    # Ensure diameter is odd
    if diameter % 2 == 0:
        diameter += 1
    
    # Set defaults
    if separation is None:
        separation = diameter + 1
    
    # Auto-compute minmass if not provided
    if minmass is None:
        # Use a percentile-based threshold
        minmass = np.percentile(frame, 90) * (diameter ** 2) * 0.1
    
    # Detect particles
    features = tp.locate(
        frame,
        diameter=diameter,
        minmass=minmass,
        separation=separation,
        percentile=percentile,
        invert=invert,
        **kwargs
    )
    
    return features


def detect_particles_batch(
    video: np.ndarray,
    diameter: int = 11,
    minmass: Optional[float] = None,
    separation: Optional[int] = None,
    percentile: float = 64,
    invert: bool = False,
    processes: Union[int, str] = 'auto',
    **kwargs
) -> "pd.DataFrame":
    """Detect particles in all frames of a video using trackpy.
    
    Args:
        video: 3D video array (T, H, W)
        diameter: Estimated particle diameter in pixels (must be odd)
        minmass: Minimum integrated brightness
        separation: Minimum separation between features
        percentile: Brightness percentile threshold
        invert: Set True if particles are dark on light background
        processes: Number of parallel processes ('auto' for all CPUs, 1 for serial)
        **kwargs: Additional arguments passed to tp.batch
        
    Returns:
        DataFrame with particle detections from all frames
    """
    if not TRACKPY_AVAILABLE:
        raise ImportError("trackpy is required for particle detection")
    
    # Ensure diameter is odd
    if diameter % 2 == 0:
        diameter += 1
    
    # Set defaults
    if separation is None:
        separation = diameter + 1
    
    # Auto-compute minmass if not provided
    if minmass is None:
        sample_frame = video[len(video) // 2]  # Use middle frame
        minmass = np.percentile(sample_frame, 90) * (diameter ** 2) * 0.1
    
    # Suppress trackpy progress output
    tp.quiet()
    
    # Batch detect with parallel processing
    features = tp.batch(
        video,
        diameter=diameter,
        minmass=minmass,
        separation=separation,
        percentile=percentile,
        invert=invert,
        processes=processes,
        **kwargs
    )
    
    return features


def link_trajectories(
    features: "pd.DataFrame",
    search_range: float = 10,
    memory: int = 3,
    adaptive_stop: Optional[float] = None,
    adaptive_step: float = 0.95,
    **kwargs
) -> "pd.DataFrame":
    """Link particle detections into trajectories.
    
    Args:
        features: DataFrame from detect_particles_batch
        search_range: Maximum distance particles can move between frames
        memory: Number of frames a particle can disappear and still be linked
        adaptive_stop: Minimum search range for adaptive search (auto-set if None)
        adaptive_step: Factor to reduce search range in adaptive mode
        **kwargs: Additional arguments passed to tp.link
        
    Returns:
        DataFrame with added 'particle' column for trajectory ID
    """
    if not TRACKPY_AVAILABLE:
        raise ImportError("trackpy is required for trajectory linking")
    
    # Suppress progress output
    tp.quiet()
    
    # Auto-set adaptive_stop to prevent "Subnetwork too large" errors
    if adaptive_stop is None:
        adaptive_stop = max(1.0, search_range / 5.0)
    
    # Try linking with adaptive search to handle dense regions
    try:
        trajectories = tp.link(
            features,
            search_range=search_range,
            memory=memory,
            adaptive_stop=adaptive_stop,
            adaptive_step=adaptive_step,
            **kwargs
        )
    except Exception as e:
        error_msg = str(e)
        if "Subnetwork" in error_msg or "subnet" in error_msg.lower():
            # Reduce search range and try again
            reduced_range = search_range / 2.0
            print(f"[link_trajectories] Network too complex, reducing search_range from {search_range} to {reduced_range}")
            trajectories = tp.link(
                features,
                search_range=reduced_range,
                memory=memory,
                adaptive_stop=max(1.0, reduced_range / 3.0),
                adaptive_step=0.9,
                **kwargs
            )
        else:
            raise
    
    return trajectories


def filter_persistent_trajectories(
    trajectories: "pd.DataFrame",
    min_frames: Optional[int] = None,
    require_all_frames: bool = True
) -> "pd.DataFrame":
    """Filter trajectories to keep only persistent ones.
    
    This is analogous to the remove_inconsistent_labels function in 
    pixel_connectivity.py - we only keep trajectories that are present
    throughout the video.
    
    Args:
        trajectories: DataFrame with 'particle' and 'frame' columns
        min_frames: Minimum number of frames a trajectory must span.
                   If None and require_all_frames=True, uses total frames.
        require_all_frames: If True, only keep trajectories present in ALL frames.
        
    Returns:
        Filtered DataFrame containing only persistent trajectories
    """
    if not PANDAS_AVAILABLE:
        raise ImportError("pandas is required for trajectory filtering")
    
    # Get total number of frames
    total_frames = trajectories['frame'].nunique()
    
    if require_all_frames:
        # Find particles that appear in every frame
        # Group by particle and count unique frames
        frame_counts = trajectories.groupby('particle')['frame'].nunique()
        
        # Only keep particles present in all frames
        valid_particles = frame_counts[frame_counts == total_frames].index
        
        print(f"[filter_persistent] Total trajectories: {trajectories['particle'].nunique()}")
        print(f"[filter_persistent] Persistent trajectories (all {total_frames} frames): {len(valid_particles)}")
        
        filtered = trajectories[trajectories['particle'].isin(valid_particles)]
    else:
        # Use threshold-based filtering
        if min_frames is None:
            min_frames = 3  # Minimum 3 frames (need at least 2 for velocity estimation)
        
        # Use trackpy's built-in stub filter
        filtered = tp.filter_stubs(trajectories, threshold=min_frames)
        
        print(f"[filter_persistent] Total trajectories: {trajectories['particle'].nunique()}")
        print(f"[filter_persistent] Trajectories with >= {min_frames} frames: {filtered['particle'].nunique()}")
    
    return filtered


def get_trajectory_array(
    trajectories: "pd.DataFrame",
    num_frames: int,
    cancel_check: Optional[Callable[[], None]] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Convert trajectory DataFrame to numpy arrays.
    
    Args:
        trajectories: Filtered DataFrame with persistent trajectories
        num_frames: Total number of frames in video
        
    Returns:
        Tuple of:
        - positions: (N_particles, T, 2) array of (x, y) positions
        - valid_mask: (N_particles, T) boolean mask for valid detections
        - particle_ids: (N_particles,) array of particle IDs
    """
    if not PANDAS_AVAILABLE:
        raise ImportError("pandas is required")
    
    particle_ids = trajectories['particle'].unique()
    n_particles = len(particle_ids)
    
    # Initialize output arrays
    positions = np.full((n_particles, num_frames, 2), np.nan, dtype=np.float64)
    valid_mask = np.zeros((n_particles, num_frames), dtype=bool)
    
    # Fill in positions for each particle
    for i, pid in enumerate(particle_ids):
        if i % 50 == 0:
            _maybe_cancel(cancel_check)
        particle_data = trajectories[trajectories['particle'] == pid]
        frames = particle_data['frame'].values.astype(int)
        x_vals = particle_data['x'].values
        y_vals = particle_data['y'].values
        
        positions[i, frames, 0] = x_vals
        positions[i, frames, 1] = y_vals
        valid_mask[i, frames] = True
    
    return positions, valid_mask, particle_ids


# =============================================================================
# SPECTRAL INTERPOLATION
# =============================================================================

def spectral_interpolate_trajectory(
    positions: np.ndarray,
    valid_mask: np.ndarray,
    smooth_factor: float = 0.0
) -> np.ndarray:
    """Interpolate missing positions in a trajectory using spectral methods.
    
    Uses Fourier-based interpolation which provides smooth, periodic-like
    interpolation that's particularly suitable for oscillatory motion.
    
    For non-periodic trajectories, we use a modified approach:
    1. Detrend the trajectory (remove linear component)
    2. Apply Fourier interpolation to residuals
    3. Add back the linear trend
    
    Args:
        positions: (T, 2) array of (x, y) positions (may contain NaN)
        valid_mask: (T,) boolean mask for valid positions
        smooth_factor: Amount of high-frequency smoothing (0-1)
        
    Returns:
        (T, 2) array with interpolated positions
    """
    T = len(positions)
    result = positions.copy()
    
    # Process x and y separately
    for dim in range(2):
        signal = positions[:, dim]
        mask = valid_mask & ~np.isnan(signal)
        
        if mask.sum() < 2:
            # Not enough points, use constant interpolation
            if mask.any():
                result[:, dim] = signal[mask][0]
            continue
        
        # Extract valid points
        valid_times = np.where(mask)[0]
        valid_values = signal[mask]
        
        # Detrend: fit linear component
        coeffs = np.polyfit(valid_times, valid_values, 1)
        trend = np.polyval(coeffs, np.arange(T))
        detrended = signal - trend
        detrended_valid = valid_values - np.polyval(coeffs, valid_times)
        
        # If we have all points, just smooth
        if mask.all():
            if smooth_factor > 0:
                # Apply Fourier smoothing
                freqs = fftfreq(T)
                spectrum = fft(detrended)
                # Smooth high frequencies
                cutoff = (1 - smooth_factor) * 0.5
                filter_mask = np.abs(freqs) < cutoff
                spectrum[~filter_mask] *= np.exp(-((np.abs(freqs[~filter_mask]) - cutoff) / 0.1) ** 2)
                result[:, dim] = np.real(ifft(spectrum)) + trend
            continue
        
        # Spectral interpolation for missing points
        # Use iterative algorithm: start with linear interp, refine with FFT
        
        # Initial guess: linear interpolation
        interpolated = np.interp(np.arange(T), valid_times, detrended_valid)
        
        # Iterative refinement (Gerchberg-Papoulis style)
        n_iters = 10
        for _ in range(n_iters):
            # Forward FFT
            spectrum = fft(interpolated)
            
            # Optional: smooth high frequencies
            if smooth_factor > 0:
                freqs = fftfreq(T)
                cutoff = (1 - smooth_factor) * 0.5
                attenuation = np.ones_like(freqs)
                high_freq = np.abs(freqs) > cutoff
                attenuation[high_freq] = np.exp(-((np.abs(freqs[high_freq]) - cutoff) / 0.1) ** 2)
                spectrum *= attenuation
            
            # Inverse FFT
            interpolated = np.real(ifft(spectrum))
            
            # Enforce known values
            interpolated[mask] = detrended_valid
        
        # Add back trend
        result[:, dim] = interpolated + trend
    
    return result


def spectral_interpolate_all_trajectories(
    positions: np.ndarray,
    valid_mask: np.ndarray,
    smooth_factor: float = 0.0,
    verbose: bool = True,
    cancel_check: Optional[Callable[[], None]] = None,
) -> np.ndarray:
    """Apply spectral interpolation to all trajectories.
    
    Args:
        positions: (N_particles, T, 2) array of positions
        valid_mask: (N_particles, T) boolean mask
        smooth_factor: Smoothing factor for high frequencies
        verbose: Print progress
        
    Returns:
        (N_particles, T, 2) array with interpolated positions
    """
    n_particles, T, _ = positions.shape
    result = np.zeros_like(positions)
    
    for i in range(n_particles):
        if i % 25 == 0:
            _maybe_cancel(cancel_check)
        result[i] = spectral_interpolate_trajectory(
            positions[i], 
            valid_mask[i], 
            smooth_factor
        )
        
        if verbose and (i + 1) % 100 == 0:
            print(f"[spectral_interpolate] Processed {i + 1}/{n_particles} trajectories")
    
    if verbose:
        print(f"[spectral_interpolate] Completed {n_particles} trajectories")
    
    return result


def spectral_smooth_trajectory_dct(
    positions: np.ndarray,
    valid_mask: np.ndarray,
    smooth_factor: float = 0.1
) -> np.ndarray:
    """Apply DCT-based spectral smoothing to a trajectory (like LocoTrack).
    
    This uses Discrete Cosine Transform for smoothing, which handles
    non-periodic signals better than FFT.
    
    Args:
        positions: (T, 2) array of (x, y) positions
        valid_mask: (T,) boolean mask for valid positions
        smooth_factor: Amount of high-frequency suppression (0 = none, 1 = heavy)
        
    Returns:
        (T, 2) smoothed trajectory
    """
    T = len(positions)
    result = positions.copy()
    
    if smooth_factor <= 0 or T < 4:
        return result
    
    for dim in range(2):
        signal = positions[:, dim].copy()
        mask = valid_mask & ~np.isnan(signal)
        
        if mask.sum() < 2:
            continue
        
        # For missing frames, interpolate linearly first
        if not mask.all() and mask.sum() >= 2:
            valid_indices = np.where(mask)[0]
            invalid_indices = np.where(~mask)[0]
            signal[invalid_indices] = np.interp(
                invalid_indices, valid_indices, signal[valid_indices]
            )
        
        # Apply DCT
        coeffs = dct(signal, type=2, norm='ortho')
        
        # Create frequency-dependent suppression
        freqs = np.arange(T)
        cutoff = int(T * (1 - smooth_factor))
        suppression = np.ones(T)
        if cutoff < T:
            rolloff_start = max(1, cutoff // 2)
            suppression[rolloff_start:] = np.exp(
                -((freqs[rolloff_start:] - rolloff_start) / max(1, T - rolloff_start)) ** 2 * 3
            )
        
        coeffs *= suppression
        
        # Inverse DCT
        result[:, dim] = idct(coeffs, type=2, norm='ortho')
    
    return result


def _smooth_single_trajectory_dct(args):
    """Helper function for parallel DCT smoothing."""
    i, pos, mask, smooth_factor = args
    return i, spectral_smooth_trajectory_dct(pos, mask, smooth_factor)


def spectral_smooth_all_trajectories_dct(
    positions: np.ndarray,
    valid_mask: np.ndarray,
    smooth_factor: float = 0.1,
    verbose: bool = True,
    n_workers: Optional[int] = None,
    cancel_check: Optional[Callable[[], None]] = None,
) -> np.ndarray:
    """Apply DCT-based spectral smoothing to all trajectories.
    
    Args:
        positions: (N_particles, T, 2) array of positions
        valid_mask: (N_particles, T) boolean mask
        smooth_factor: Smoothing factor (0-1)
        verbose: Print progress
        n_workers: Number of parallel workers (unused, kept for API compatibility)
        
    Returns:
        (N_particles, T, 2) smoothed positions
    """
    n_particles, T, _ = positions.shape
    result = np.zeros_like(positions)
    
    # Sequential processing - more reliable than parallel for numpy operations
    for i in range(n_particles):
        if i % 25 == 0:
            _maybe_cancel(cancel_check)
        result[i] = spectral_smooth_trajectory_dct(
            positions[i], valid_mask[i], smooth_factor
        )
    
    if verbose:
        # Compute smoothing statistics
        orig_velocities = np.diff(positions, axis=1)
        smooth_velocities = np.diff(result, axis=1)
        orig_jitter = np.nanstd(np.diff(orig_velocities, axis=1))
        smooth_jitter = np.nanstd(np.diff(smooth_velocities, axis=1))
        reduction = (1 - smooth_jitter / max(orig_jitter, 1e-8)) * 100
        print(f"[spectral_smooth_dct] Velocity jitter: {orig_jitter:.4f} -> {smooth_jitter:.4f} "
              f"({reduction:.1f}% reduction)")
    
    return result


# =============================================================================
# DENSE OPTICAL FLOW GENERATION
# =============================================================================

def compute_trajectory_velocities(
    positions: np.ndarray
) -> np.ndarray:
    """Compute velocities from interpolated positions.
    
    Args:
        positions: (N_particles, T, 2) array of interpolated positions
        
    Returns:
        (N_particles, T-1, 2) array of velocities (dx, dy)
    """
    # Velocity = position difference between consecutive frames
    velocities = np.diff(positions, axis=1)
    return velocities


def generate_dense_flow_gpu(
    positions: np.ndarray,
    velocities: np.ndarray,
    H: int,
    W: int,
    frame_idx: int,
    device: "torch.device",
    smoothing: float = 15.0,
    kernel: str = 'gaussian',
    *,
    cancel_check: Optional[Callable[[], None]] = None,
    max_chunk_elements: int = 10_000_000,
) -> np.ndarray:
    """Generate dense optical flow field using GPU-accelerated interpolation.
    
    Uses row-chunking to avoid allocating massive (H, W, N) tensors that can
    hang the GPU for large images. This ensures bounded memory usage.
    
    Args:
        positions: (N, T, 2) array of particle positions (x, y)
        velocities: (N, T-1, 2) array of particle velocities
        H, W: Height and width of output flow field
        frame_idx: Frame index (0 to T-2)
        device: torch device for GPU acceleration
        smoothing: Smoothing/bandwidth parameter
        kernel: Interpolation kernel ('gaussian', 'idw', 'wendland')
        cancel_check: Optional callback to check for cancellation
        max_chunk_elements: Maximum elements per chunk to control memory usage
        
    Returns:
        (H, W, 2) dense optical flow field
    """
    if not TORCH_AVAILABLE:
        raise ImportError("PyTorch required for GPU acceleration")
    
    # Get positions and velocities at this frame
    pts = positions[:, frame_idx, :]  # (N, 2) - x, y
    vels = velocities[:, frame_idx, :]  # (N, 2) - dx, dy
    
    # Filter out invalid points
    valid = ~(np.isnan(pts).any(axis=1) | np.isnan(vels).any(axis=1))
    pts_valid = pts[valid]
    vels_valid = vels[valid]
    
    if len(pts_valid) < 3:
        return np.zeros((H, W, 2), dtype=np.float32)
    
    # Filter velocity outliers
    vel_magnitudes = np.linalg.norm(vels_valid, axis=1)
    if len(vel_magnitudes) > 10:
        vel_median = np.median(vel_magnitudes)
        vel_mad = np.median(np.abs(vel_magnitudes - vel_median))
        vel_threshold = vel_median + 5 * max(vel_mad, 1.0)
        inlier_mask = vel_magnitudes < vel_threshold
        if inlier_mask.sum() >= 3:
            pts_valid = pts_valid[inlier_mask]
            vels_valid = vels_valid[inlier_mask]
    
    # Move to GPU
    pts_t = torch.from_numpy(pts_valid).float().to(device)
    vels_t = torch.from_numpy(vels_valid).float().to(device)
    
    n_pts = int(pts_t.shape[0])
    if n_pts <= 0:
        return np.zeros((H, W, 2), dtype=np.float32)
    
    # Compute chunk height to keep (chunkH * W * N) bounded in memory
    # This prevents GPU OOM and hanging for large images
    denom = max(1, int(W) * n_pts)
    chunk_h = int(max(1, min(H, max_chunk_elements // denom)))
    
    # Precompute broadcastable tensors
    xx = torch.arange(W, device=device, dtype=torch.float32).view(1, W, 1)  # (1, W, 1)
    pts_x = pts_t[:, 0].view(1, 1, -1)  # (1, 1, N)
    pts_y = pts_t[:, 1].view(1, 1, -1)  # (1, 1, N)
    vels_x = vels_t[:, 0].view(1, 1, -1)  # (1, 1, N)
    vels_y = vels_t[:, 1].view(1, 1, -1)  # (1, 1, N)
    
    flow = torch.empty((H, W, 2), device=device, dtype=torch.float32)
    
    for y0 in range(0, H, chunk_h):
        if cancel_check is not None:
            cancel_check()
        y1 = min(H, y0 + chunk_h)
        yy = torch.arange(y0, y1, device=device, dtype=torch.float32).view(-1, 1, 1)  # (chunkH, 1, 1)
        
        # dist_sq: (chunkH, W, N)
        dist_sq = (xx - pts_x) ** 2 + (yy - pts_y) ** 2
        
        # Compute weights based on kernel type
        if kernel == 'gaussian':
            sigma = float(max(smoothing, 1e-6))
            weights = torch.exp(-dist_sq / (2.0 * sigma * sigma))
        elif kernel == 'wendland':
            r = float(max(smoothing, 1e-6)) * 3.0
            dist = torch.sqrt(dist_sq + 1e-8)
            d_norm = dist / r
            weights = torch.clamp(1 - d_norm, min=0) ** 4 * (4 * d_norm + 1)
        else:  # 'idw'
            s2 = float(max(smoothing, 1e-6)) ** 2
            weights = 1.0 / (dist_sq + s2)
        
        # Normalize weights
        weights_sum = weights.sum(dim=-1, keepdim=True) + 1e-8
        weights_norm = weights / weights_sum
        
        # Interpolate velocities
        fx = (weights_norm * vels_x).sum(dim=-1)  # (chunkH, W)
        fy = (weights_norm * vels_y).sum(dim=-1)  # (chunkH, W)
        flow[y0:y1, :, 0] = fx
        flow[y0:y1, :, 1] = fy
        
        # Help GC/allocator between chunks
        del dist_sq, weights, weights_sum, weights_norm, fx, fy
    
    return flow.cpu().numpy().astype(np.float32)


def generate_dense_flow_rbf(
    positions: np.ndarray,
    velocities: np.ndarray,
    H: int,
    W: int,
    frame_idx: int,
    smoothing: float = 10.0,
    kernel: str = 'thin_plate_spline',
    *,
    query_pts: Optional[np.ndarray] = None,
    epsilon: Optional[float] = None,
    neighbors: Optional[int] = None,
) -> np.ndarray:
    """Generate dense optical flow field using RBF interpolation.
    
    Args:
        positions: (N_particles, T, 2) array of particle positions
        velocities: (N_particles, T-1, 2) array of particle velocities
        H, W: Height and width of output flow field
        frame_idx: Frame index (0 to T-2)
        smoothing: RBF smoothing parameter
        kernel: RBF kernel type
        query_pts: Optional pre-computed query grid (H*W, 2) to avoid re-allocation
        epsilon: Optional epsilon for kernels that require it
        neighbors: Optional limit to local neighbors for speed on large grids
        
    Returns:
        (H, W, 2) dense optical flow field
    """
    n_particles = positions.shape[0]
    
    # Get positions and velocities at this frame
    pts = positions[:, frame_idx, :]  # (N, 2) - x, y
    vels = velocities[:, frame_idx, :]  # (N, 2) - dx, dy
    
    # Filter out invalid points (NaN)
    valid = ~(np.isnan(pts).any(axis=1) | np.isnan(vels).any(axis=1))
    pts_valid = pts[valid]
    vels_valid = vels[valid]
    
    if len(pts_valid) < 3:
        # Not enough points, return zero flow
        return np.zeros((H, W, 2), dtype=np.float32)
    
    # Create grid of query points (optionally provided/cached by caller)
    if query_pts is None:
        yy, xx = np.mgrid[0:H, 0:W]
        query_pts = np.stack([xx.ravel(), yy.ravel()], axis=1)  # (H*W, 2)
    
    # Some kernels require epsilon in SciPy's RBFInterpolator
    kernels_requiring_epsilon = {
        'gaussian',
        'multiquadric',
        'inverse_multiquadric',
        'inverse_quadratic',
    }
    if kernel in kernels_requiring_epsilon and epsilon is None:
        epsilon = float(max(smoothing, 1e-6))
    
    # Build RBF interpolators for dx and dy
    try:
        # Limit to local neighbors for speed on large grids
        neighbors_eff = None
        if neighbors is not None:
            neighbors_eff = int(min(max(neighbors, 3), len(pts_valid)))
        
        rbf_kwargs = {
            'smoothing': smoothing,
            'kernel': kernel,
        }
        if kernel in kernels_requiring_epsilon:
            rbf_kwargs['epsilon'] = epsilon
        if neighbors_eff is not None:
            rbf_kwargs['neighbors'] = neighbors_eff
        
        rbf_dx = RBFInterpolator(pts_valid, vels_valid[:, 0], **rbf_kwargs)
        rbf_dy = RBFInterpolator(pts_valid, vels_valid[:, 1], **rbf_kwargs)
        
        # Interpolate
        dx = rbf_dx(query_pts).reshape(H, W)
        dy = rbf_dy(query_pts).reshape(H, W)
    except Exception as e:
        warnings.warn(f"RBF interpolation failed: {e}. Using IDW fallback.")
        dx, dy = _idw_interpolate(pts_valid, vels_valid, H, W)
    
    flow = np.stack([dx, dy], axis=-1).astype(np.float32)
    return flow


def _idw_interpolate(
    pts: np.ndarray,
    vels: np.ndarray,
    H: int,
    W: int,
    power: float = 2.0,
    epsilon: float = 1e-6,
    cancel_check: Optional[Callable[[], None]] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Inverse Distance Weighted interpolation fallback.
    
    Args:
        pts: (N, 2) control points
        vels: (N, 2) velocities at control points
        H, W: Output dimensions
        power: Distance weighting power
        epsilon: Small value to prevent division by zero
        
    Returns:
        dx, dy arrays of shape (H, W)
    """
    yy, xx = np.mgrid[0:H, 0:W]
    
    dx = np.zeros((H, W), dtype=np.float64)
    dy = np.zeros((H, W), dtype=np.float64)
    weights_sum = np.zeros((H, W), dtype=np.float64)
    
    for i in range(len(pts)):
        if i % 25 == 0:
            _maybe_cancel(cancel_check)
        # Distance from this control point to all grid points
        dist = np.sqrt((xx - pts[i, 0])**2 + (yy - pts[i, 1])**2) + epsilon
        weight = 1.0 / (dist ** power)
        
        dx += weight * vels[i, 0]
        dy += weight * vels[i, 1]
        weights_sum += weight
    
    dx /= weights_sum
    dy /= weights_sum
    
    return dx.astype(np.float32), dy.astype(np.float32)


def generate_dense_flow_all_frames(
    positions: np.ndarray,
    H: int,
    W: int,
    smoothing: float = 15.0,
    kernel: str = 'gaussian',
    gaussian_sigma: Optional[float] = None,
    verbose: bool = True,
    use_gpu: bool = True,
    cancel_check: Optional[Callable[[], None]] = None,
) -> np.ndarray:
    """Generate dense optical flow field for all frames.
    
    Args:
        positions: (N_particles, T, 2) interpolated trajectory positions
        H, W: Height and width of output flow field
        smoothing: Interpolation smoothing parameter
        kernel: Interpolation kernel ('gaussian', 'idw', 'wendland', 'thin_plate_spline')
        gaussian_sigma: Optional Gaussian smoothing of output flow
        verbose: Print progress
        use_gpu: Use GPU acceleration if available
        
    Returns:
        (T-1, H, W, 2) dense optical flow field
    """
    n_particles, T, _ = positions.shape
    
    # Compute velocities
    velocities = compute_trajectory_velocities(positions)
    
    # Determine device for GPU acceleration (CUDA only)
    device = None
    if use_gpu and TORCH_AVAILABLE:
        if torch.cuda.is_available():
            device = torch.device('cuda')
    
    # Map gaussian_rbf to scipy's gaussian kernel name (for scipy fallback)
    scipy_kernel = 'gaussian' if kernel == 'gaussian_rbf' else kernel
    
    # Map gaussian_rbf to gaussian for GPU path (they use the same Gaussian weighting)
    gpu_kernel = 'gaussian' if kernel == 'gaussian_rbf' else kernel
    
    # GPU-supported kernels (gaussian_rbf is mapped to gaussian for GPU acceleration)
    gpu_kernels = ['gaussian', 'gaussian_rbf', 'idw', 'wendland']
    
    if verbose:
        if device is not None and kernel in gpu_kernels:
            print(f"[generate_flow] Using GPU-accelerated interpolation on {device} (kernel={kernel})")
        else:
            print(f"[generate_flow] Using scipy RBF interpolation (kernel={scipy_kernel})")
    
    # Precompute query grid once (used by scipy path)
    yy, xx = np.mgrid[0:H, 0:W]
    query_pts = np.stack([xx.ravel(), yy.ravel()], axis=1)
    
    # Generate flow for each frame
    flows = np.zeros((T - 1, H, W, 2), dtype=np.float32)
    
    for t in range(T - 1):
        if t % 5 == 0:
            _maybe_cancel(cancel_check)
        # Use GPU for supported kernels, otherwise fall back to RBF
        if device is not None and kernel in gpu_kernels:
            flows[t] = generate_dense_flow_gpu(
                positions, velocities, H, W, t, device,
                smoothing=smoothing, kernel=gpu_kernel,
                cancel_check=cancel_check
            )
        else:
            # Only use neighbors for large point sets (>100) where KDTree helps.
            # For small sets, the KDTree overhead actually slows things down.
            neighbors = 32 if n_particles > 100 else None
            flows[t] = generate_dense_flow_rbf(
                positions, velocities, H, W, t,
                smoothing=smoothing, kernel=scipy_kernel,
                query_pts=query_pts,
                epsilon=smoothing,
                neighbors=neighbors,
            )
        
        # Optional Gaussian smoothing
        if gaussian_sigma is not None and gaussian_sigma > 0:
            flows[t, :, :, 0] = gaussian_filter(flows[t, :, :, 0], sigma=gaussian_sigma)
            flows[t, :, :, 1] = gaussian_filter(flows[t, :, :, 1], sigma=gaussian_sigma)
        
        if verbose and (t + 1) % 10 == 0:
            print(f"[generate_flow] Processed {t + 1}/{T - 1} frames")
    
    if verbose:
        print(f"[generate_flow] Completed {T - 1} frames, shape: {flows.shape}")
    
    return flows


# =============================================================================
# HIGH-LEVEL API
# =============================================================================

def compute_trackpy_optical_flow(
    video: np.ndarray,
    # DoG detection parameters (shared with LocoTrack)
    radius: float = 2.5,
    threshold: float = 0.0,
    median_filter: bool = False,
    subpixel: bool = True,
    # Trackpy linking parameters  
    search_range: float = 15,
    memory: int = 5,
    require_persistent: bool = True,
    min_trajectory_length: Optional[int] = None,
    # Spectral smoothing parameters
    smooth_factor: float = 0.1,
    use_dct_smoothing: bool = False,
    # Flow field generation parameters
    flow_smoothing: float = 15.0,
    kernel: str = 'gaussian',
    gaussian_sigma: Optional[float] = 2.0,
    # Other parameters
    invert: bool = False,
    verbose: bool = True,
    # Legacy parameters (for backward compatibility)
    diameter: Optional[int] = None,
    minmass: Optional[float] = None,
    separation: Optional[int] = None,
    percentile: float = 64,
    n_workers: Optional[int] = None,
    cancel_check: Optional[Callable[[], None]] = None,
) -> Tuple[np.ndarray, Dict]:
    """High-level function to compute optical flow using DoG detection + trackpy linking.
    
    Complete pipeline:
    1. Detect particles in all frames using DoG detector (same as LocoTrack)
    2. Link detections into trajectories using trackpy's linker
    3. Filter to keep only persistent/long trajectories
    4. Apply spectral smoothing (DCT or FFT) to reduce jitter
    5. Generate dense optical flow field via kernel interpolation
    
    Key features:
    - Uses SAME DoG detection as LocoTrack for consistent particle detection
    - Uses trackpy's robust linker with search radius and memory
    - Spectral smoothing for smooth trajectory reconstruction
    
    Args:
        video: (T, H, W) video array (normalized to [0, 1] or uint8)
        
        # DoG detection parameters (shared with LocoTrack):
        radius: Estimated object radius in pixels (default: 2.5)
        threshold: Quality threshold for detection, 0 = accept all (default: 0.0)
        median_filter: Apply 3x3 median filter before detection (default: False)
        subpixel: Use sub-pixel localization (default: True)
        
        # Trackpy linking parameters:
        search_range: Maximum particle displacement between frames (default: 15)
        memory: Frames a particle can disappear and still be linked (default: 5)
        require_persistent: If True, only use trajectories present in all frames (default: True)
        min_trajectory_length: Minimum frames for a trajectory (if not require_persistent)
        
        # Spectral smoothing:
        smooth_factor: Spectral smoothing factor (0-1, higher = smoother) (default: 0.1)
        use_dct_smoothing: Use DCT (True) or FFT (False) for spectral smoothing (default: False)
        
        # Flow field generation:
        flow_smoothing: Interpolation bandwidth for flow field generation (default: 15.0)
        kernel: Interpolation kernel ('gaussian', 'idw', 'wendland', 'thin_plate_spline', 'gaussian_rbf')
        gaussian_sigma: Gaussian post-smoothing of output flow (default: 2.0)
        
        # Other:
        invert: Set True if particles are dark on light background (default: False)
        verbose: Print progress information (default: True)
        
        # Legacy parameters (for backward compatibility with old trackpy API):
        diameter: If provided, converts to radius = diameter / 2
        minmass, separation, percentile, n_workers: Ignored (for backward compat)
        
    Returns:
        Tuple of:
        - flows: (T-1, H, W, 2) dense optical flow field
        - info: Dictionary with tracking statistics
    """
    if not TRACKPY_AVAILABLE:
        raise ImportError("trackpy is required for trajectory linking. Install with: pip install trackpy")
    
    if not PANDAS_AVAILABLE:
        raise ImportError("pandas is required. Install with: pip install pandas")
    
    T, H, W = video.shape

    _maybe_cancel(cancel_check)
    
    # Handle legacy diameter parameter - if provided, use legacy trackpy detection
    use_legacy_detection = (diameter is not None)
    if diameter is not None:
        radius = diameter / 2.0
        if verbose:
            print(f"[trackpy_flow] Legacy mode: diameter={diameter} provided, using trackpy native detection")
    
    # Normalize video if needed
    if video.max() > 1.0:
        video_norm = video.astype(np.float32) / 255.0
    else:
        video_norm = video
    
    if verbose:
        print(f"[trackpy_flow] Processing video: {T} frames, {H}x{W} pixels")
    
    # Step 1: Detect particles in all frames
    if use_legacy_detection:
        # Use legacy trackpy detection (tp.batch)
        if verbose:
            print(f"[trackpy_flow] Step 1: Detecting particles using trackpy native (diameter={diameter})...")
        
        # For legacy mode, use the original video (not normalized to 0-1)
        video_for_detect = video if video.max() > 1.0 else (video * 255).astype(np.uint8)
        
        features = detect_particles_batch(
            video_for_detect,
            diameter=diameter,
            minmass=minmass,
            separation=separation,
            percentile=percentile,
            invert=invert,
            processes=n_workers if n_workers else 'auto'
        )
    else:
        # Use DoG detection (same as LocoTrack)
        if verbose:
            print(f"[trackpy_flow] Step 1: Detecting particles using DoG (radius={radius}, threshold={threshold})...")
        
        # Normalize threshold if video is normalized to 0-1
        # DoG threshold should be relative to image intensity range
        effective_threshold = threshold
        if video_norm.max() <= 1.0 and threshold > 1.0:
            # Threshold was set for 0-255 range, convert to 0-1 range
            effective_threshold = threshold / 255.0
            if verbose:
                print(f"[trackpy_flow]   Note: Converting threshold {threshold} -> {effective_threshold:.4f} for normalized video")
        
        features = detect_particles_dog_batch(
            video_norm, 
            radius=radius,
            threshold=effective_threshold,
            median_filter=median_filter,
            subpixel=subpixel,
            invert=invert,
            verbose=verbose,
            cancel_check=cancel_check,
        )

    _maybe_cancel(cancel_check)
    
    n_detections = len(features)
    if verbose:
        print(f"[trackpy_flow]   Total detections: {n_detections} across {T} frames")
        print(f"[trackpy_flow]   Average detections per frame: {n_detections / T:.1f}")
        # Show per-frame breakdown
        if n_detections > 0:
            frame_counts = features.groupby('frame').size()
            print(f"[trackpy_flow]   Frames with detections: {len(frame_counts)}/{T}")
            print(f"[trackpy_flow]   Detections range: {frame_counts.min()} - {frame_counts.max()} per frame")
    
    if n_detections == 0:
        warnings.warn("No particles detected! Check radius and threshold parameters.")
        return np.zeros((T - 1, H, W, 2), dtype=np.float32), {"error": "no_particles"}
    
    # Step 2: Link trajectories using trackpy
    if verbose:
        print(f"[trackpy_flow] Step 2: Linking trajectories (search_range={search_range}, memory={memory})...")
    trajectories = link_trajectories(
        features, search_range=search_range, memory=memory
    )

    _maybe_cancel(cancel_check)
    n_trajectories = trajectories['particle'].nunique()
    if verbose:
        print(f"[trackpy_flow]   Total trajectories linked: {n_trajectories}")
        if n_trajectories > 0:
            # Show trajectory length distribution
            traj_lengths = trajectories.groupby('particle')['frame'].count()
            print(f"[trackpy_flow]   Trajectory lengths: min={traj_lengths.min()}, max={traj_lengths.max()}, median={traj_lengths.median():.0f}")
    
    # Step 3: Filter trajectories
    if verbose:
        print("[trackpy_flow] Step 3: Filtering trajectories...")
    if require_persistent:
        if verbose:
            print(f"[trackpy_flow]   Mode: require_persistent=True (need trajectory in ALL {T} frames)")
        filtered = filter_persistent_trajectories(
            trajectories, require_all_frames=True
        )
    else:
        # Use a much more lenient default: minimum 3 frames (need at least 2 for velocity)
        min_len = min_trajectory_length if min_trajectory_length else 3
        if verbose:
            print(f"[trackpy_flow]   Mode: require_persistent=False, min_frames={min_len}")
        filtered = filter_persistent_trajectories(
            trajectories, min_frames=min_len, require_all_frames=False
        )

    _maybe_cancel(cancel_check)
    
    n_persistent = filtered['particle'].nunique()
    if verbose:
        print(f"[trackpy_flow]   Filtered trajectories: {n_persistent}")
    
    if n_persistent == 0:
        warnings.warn("No persistent trajectories found! Try increasing memory or search_range.")
        return np.zeros((T - 1, H, W, 2), dtype=np.float32), {"error": "no_persistent"}
    
    # Step 4: Convert to arrays and apply spectral smoothing
    if verbose:
        print(f"[trackpy_flow] Step 4: Spectral smoothing (DCT={use_dct_smoothing}, factor={smooth_factor})...")
    positions, valid_mask, particle_ids = get_trajectory_array(filtered, T, cancel_check=cancel_check)
    
    if smooth_factor > 0:
        if use_dct_smoothing:
            interpolated = spectral_smooth_all_trajectories_dct(
                positions, valid_mask, smooth_factor=smooth_factor, verbose=verbose,
                n_workers=n_workers,
                cancel_check=cancel_check,
            )
        else:
            interpolated = spectral_interpolate_all_trajectories(
                positions, valid_mask, smooth_factor=smooth_factor, verbose=verbose,
                cancel_check=cancel_check,
            )
    else:
        # Just interpolate missing values without smoothing
        interpolated = spectral_interpolate_all_trajectories(
            positions, valid_mask, smooth_factor=0.0, verbose=verbose,
            cancel_check=cancel_check,
        )

    _maybe_cancel(cancel_check)
    
    # Step 5: Generate dense flow
    if verbose:
        print(f"[trackpy_flow] Step 5: Generating dense optical flow (kernel={kernel})...")
    flows = generate_dense_flow_all_frames(
        interpolated, H, W,
        smoothing=flow_smoothing,
        kernel=kernel,
        gaussian_sigma=gaussian_sigma,
        verbose=verbose,
        use_gpu=True,
        cancel_check=cancel_check,
    )
    
    # Compile statistics
    info = {
        "total_detections": n_detections,
        "total_trajectories": n_trajectories,
        "persistent_trajectories": n_persistent,
        "frames": T,
        "height": H,
        "width": W,
        "detection_method": "dog",  # Mark that we used DoG detection
        "parameters": {
            "radius": radius,
            "threshold": threshold,
            "median_filter": median_filter,
            "subpixel": subpixel,
            "search_range": search_range,
            "memory": memory,
            "require_persistent": require_persistent,
            "min_trajectory_length": min_trajectory_length,
            "smooth_factor": smooth_factor,
            "use_dct_smoothing": use_dct_smoothing,
            "flow_smoothing": flow_smoothing,
            "kernel": kernel,
            "gaussian_sigma": gaussian_sigma,
            "invert": invert,
        }
    }
    
    if verbose:
        print(f"[trackpy_flow] Complete! Flow shape: {flows.shape}")
    
    return flows, info


# =============================================================================
# MAIN - EXAMPLE USAGE
# =============================================================================

if __name__ == "__main__":
    import tifffile
    
    print("="*60)
    print("TRACKPY-BASED OPTICAL FLOW GENERATION")
    print("="*60)
    
    # Example usage (modify paths as needed)
    TIF_PATH = "./data/example_video.tif"
    
    try:
        # Load video
        print(f"Loading video: {TIF_PATH}")
        vol = tifffile.imread(TIF_PATH)
        if vol.ndim == 3 and vol.shape[-1] < vol.shape[0]:
            vol = np.moveaxis(vol, -1, 0)
        
        T, H, W = vol.shape
        print(f"Video shape: {T} frames, {H}x{W} pixels")
        
        # Normalize
        vol = vol.astype(np.float32)
        vol = (vol - vol.min()) / (vol.max() - vol.min() + 1e-6)
        
        # Compute trackpy-based optical flow
        flows, info = compute_trackpy_optical_flow(
            vol,
            diameter=11,
            minmass=None,  # Auto-compute
            search_range=15,
            memory=5,
            require_persistent=True,
            smooth_factor=0.1,
            flow_smoothing=15.0,
            gaussian_sigma=2.0,
            verbose=True
        )
        
        print("\n" + "="*60)
        print("RESULTS")
        print("="*60)
        print(f"Flow shape: {flows.shape}")
        print(f"Total detections: {info['total_detections']}")
        print(f"Total trajectories: {info['total_trajectories']}")
        print(f"Persistent trajectories: {info['persistent_trajectories']}")
        
        # Save flows
        np.save("./data/trackpy_flows.npy", flows)
        print(f"Saved to: ./data/trackpy_flows.npy")
        
    except FileNotFoundError:
        print(f"Example video not found at {TIF_PATH}")
        print("This is expected - modify the path to point to your video.")
    except Exception as e:
        print(f"Error: {e}")
        raise
