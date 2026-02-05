#!/usr/bin/env python3
"""LocoTrack-based Optical Flow Generation.

This module provides optical flow computation based on LocoTrack point tracking,
designed as a replacement for trackpy-based flow for fluorescent microscopy images.

Key Features:
1. DoG (Difference of Gaussians) based seed detection on the first frame
2. LocoTrack-based point tracking throughout the video
3. Filtering of trajectories based on occlusion predictions
4. Conversion from sparse trajectories to dense optical flow field (RBF interpolation)

The approach:
1. Use DoG detector to find particles/features on the first frame
2. Initialize LocoTrack with these seeds at frame 0
3. Track all seeds throughout the video using LocoTrack
4. Filter out trajectories with high occlusion percentage
5. Generate dense optical flow field via spatial interpolation (RBF/IDW)

Dependencies:
    - torch: PyTorch for LocoTrack model
    - scipy: DoG detection, interpolation, and spatial algorithms
    - numpy: Numerical operations
"""

import os
import sys
from pathlib import Path
import numpy as np
from typing import Callable, Tuple, Optional, Dict, List, Union
import warnings

import torch
import torch.nn.functional as F
from scipy.ndimage import gaussian_filter, gaussian_laplace
from scipy.interpolate import RBFInterpolator
from scipy.ndimage import maximum_filter, minimum_filter, median_filter as scipy_median_filter
from scipy.fft import dct, idct


def find_local_maxima_trackmate_style(dog: np.ndarray, threshold: float, exclude_border: bool = False) -> np.ndarray:
    """
    Find local maxima in TrackMate style - no min_distance suppression.
    
    TrackMate uses RectangleShape(1, true) which creates a 3x3 neighborhood
    and finds pixels that are greater than or equal to ALL 8 neighbors AND above threshold.
    
    This is fundamentally different from skimage.peak_local_max which applies
    min_distance-based suppression.
    
    TrackMate extends the image by 1 pixel with mirror padding before finding maxima,
    which effectively allows detection at all pixels including edges. We replicate
    this by using mode='mirror' in maximum_filter and not excluding the border by default.
    
    Args:
        dog: 2D DoG-filtered image
        threshold: Minimum value for a pixel to be considered a maximum
        exclude_border: If True, exclude 1-pixel border (default: False to match TrackMate)
        
    Returns:
        (N, 2) array of (y, x) coordinates
    """
    # Find all pixels that are local maxima using maximum_filter with 3x3 footprint
    # A pixel is a local maximum if it equals the maximum in its 3x3 neighborhood
    neighborhood = np.ones((3, 3), dtype=bool)
    neighborhood[1, 1] = False  # Exclude center (match TrackMate's RectangleShape(1, true))
    
    # Get the maximum value in each 3x3 neighborhood (excluding center)
    # TrackMate uses Views.extendMirrorSingle which is symmetric reflection without edge repeat
    local_max_values = maximum_filter(dog, footprint=neighborhood, mode='mirror')
    
    # A pixel is a local maximum if it's greater than or equal to all its neighbors
    # AND above or equal to the threshold (TrackMate uses non-strict comparison)
    is_local_max = (dog >= local_max_values) & (dog >= threshold)
    
    # TrackMate extends image by 1 pixel with mirror padding, so it can detect at edges
    # Only exclude border if explicitly requested (not TrackMate's default behavior)
    if exclude_border:
        is_local_max[0, :] = False
        is_local_max[-1, :] = False
        is_local_max[:, 0] = False
        is_local_max[:, -1] = False
    
    # Get coordinates of local maxima
    y_coords, x_coords = np.where(is_local_max)
    
    if len(y_coords) == 0:
        return np.zeros((0, 2), dtype=np.int64)
    
    return np.stack([y_coords, x_coords], axis=-1)


# Add locotrack_pytorch to path
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# Find RIPPLE root robustly by searching for repository markers.
# We support running from:
# 1) <repo>/src/main/python (development)
# 2) <repo>/target/classes/python (after Maven build)
def find_ripple_root(start_dir: str) -> str:
    """Find RIPPLE repository root by searching parent directories."""
    current = Path(start_dir).resolve()
    for _ in range(12):
        # Maven repo marker
        if (current / "pom.xml").is_file():
            return str(current)
        # LocoTrack weights marker (canonical location)
        if (current / "locotrack_pytorch" / "weights").is_dir():
            return str(current)

        parent = current.parent
        if parent == current:
            break
        current = parent

    # Fallback: best-effort (keeps older behavior but avoids breaking)
    return str(Path(start_dir).resolve().parent)

RIPPLE_ROOT = find_ripple_root(SCRIPT_DIR)

# Add locotrack_pytorch to path for model imports
# This is the canonical location for all LocoTrack-related code
LOCOTRACK_PYTORCH_PATH = os.path.join(RIPPLE_ROOT, "locotrack_pytorch")
if os.path.isdir(LOCOTRACK_PYTORCH_PATH) and LOCOTRACK_PYTORCH_PATH not in sys.path:
    sys.path.insert(0, LOCOTRACK_PYTORCH_PATH)

# LocoTrack model will be imported when needed to avoid import errors at module load time


# =============================================================================
# DIFFERENCE OF GAUSSIANS (DoG) DETECTOR
# =============================================================================

class DoGDetector:
    """
    Difference of Gaussians detector matching TrackMate's implementation.
    
    TrackMate DoG implementation:
    - sigma1 = radius / sqrt(ndim) * 0.9
    - sigma2 = radius / sqrt(ndim) * 1.1
    - DoG = Gaussian(sigma1) - Gaussian(sigma2)
    - Find local maxima above threshold
    
    Optional features:
    - Median filter pre-processing (3x3)
    - Sub-pixel localization via quadratic fitting
    """
    
    def __init__(self, diameter_pixels: float, threshold: float,
                 do_median_filter: bool = False, do_subpixel: bool = False):
        """
        Args:
            diameter_pixels: Estimated object diameter in pixels
            threshold: Quality threshold for detection
            do_median_filter: Apply 3x3 median filter before detection
            do_subpixel: Refine positions with quadratic fitting
        """
        self.diameter = diameter_pixels
        self.radius = diameter_pixels / 2.0
        self.threshold = threshold
        self.do_median_filter = do_median_filter
        self.do_subpixel = do_subpixel
        
        # TrackMate's sigma calculation for 2D images
        ndim = 2
        sigma1_nominal = self.radius / np.sqrt(ndim) * 0.9
        sigma2_nominal = self.radius / np.sqrt(ndim) * 1.1
        
        # TrackMate applies imageSigma correction (assumes inherent blur of 0.5 pixels)
        # This is done in DifferenceOfGaussian.computeSigmas()
        # sigmas_actual = sqrt(sigma^2 - imageSigma^2)
        imageSigma = 0.5
        self.sigma1 = np.sqrt(max(sigma1_nominal**2 - imageSigma**2, 0.01))
        self.sigma2 = np.sqrt(max(sigma2_nominal**2 - imageSigma**2, 0.01))
        
        print(f"DoG Detector initialized:")
        print(f"  Diameter: {self.diameter} pixels")
        print(f"  Radius: {self.radius} pixels")
        print(f"  Sigma1 (smaller): {self.sigma1:.3f}")
        print(f"  Sigma2 (larger): {self.sigma2:.3f}")
        print(f"  Threshold: {self.threshold}")
        print(f"  Median filter: {self.do_median_filter}")
        print(f"  Sub-pixel localization: {self.do_subpixel}")
    
    def apply_median_filter(self, image: np.ndarray) -> np.ndarray:
        """Apply 3x3 median filter (matching TrackMate's implementation).
        
        TrackMate uses Views.extendZero for boundary handling (zero-padding).
        """
        from scipy.ndimage import median_filter as scipy_median_filter
        return scipy_median_filter(image, size=3, mode='constant', cval=0)
    
    def compute_dog(self, image: np.ndarray) -> np.ndarray:
        """
        Compute Difference of Gaussians.
        
        Following TrackMate: dog = gauss(sigma1) - gauss(sigma2)
        where sigma1 < sigma2
        """
        # Apply Gaussian filters
        gauss1 = gaussian_filter(image.astype(np.float64), sigma=self.sigma1)
        gauss2 = gaussian_filter(image.astype(np.float64), sigma=self.sigma2)
        
        # DoG: smaller sigma - larger sigma (blob detection)
        dog = gauss1 - gauss2
        
        return dog
    
    def find_local_maxima_legacy(self, dog: np.ndarray, min_distance: int = None, 
                          exclude_border: bool = True, max_keypoints: Optional[int] = None) -> np.ndarray:
        """
        Find local maxima above threshold using skimage.peak_local_max.
        
        NOTE: This is the LEGACY method that uses min_distance suppression.
        TrackMate does NOT use min_distance - it finds ALL local maxima.
        Use find_local_maxima() for TrackMate-compatible behavior.
        
        Args:
            dog: DoG-filtered image
            min_distance: Minimum distance between keypoints (default: ceil(radius))
            exclude_border: Exclude keypoints at image border
            max_keypoints: Maximum number of keypoints to return
        
        Returns array of (y, x) coordinates.
        """
        from skimage.feature import peak_local_max
        
        # Use minimum separation based on radius if not specified
        if min_distance is None:
            min_distance = int(np.ceil(self.radius))
        
        # Build kwargs for peak_local_max
        peak_kwargs = {
            'min_distance': min_distance,
            'threshold_abs': self.threshold,
            'exclude_border': exclude_border,
        }
        
        # num_peaks parameter (limit number of peaks)
        if max_keypoints is not None:
            peak_kwargs['num_peaks'] = max_keypoints
        
        # Find local maxima
        coordinates = peak_local_max(dog, **peak_kwargs)
        
        return coordinates
    
    def find_local_maxima(self, dog: np.ndarray, exclude_border: bool = False, 
                          max_keypoints: Optional[int] = None) -> np.ndarray:
        """
        Find local maxima in TrackMate style - no min_distance suppression.
        
        This matches TrackMate's approach exactly:
        - Uses 3x3 neighborhood (RectangleShape(1, true))
        - Finds pixels that are strictly greater than ALL 8 neighbors
        - Applies threshold filter
        - No min_distance-based suppression
        - Extends image with mirror padding to allow edge detection (exclude_border=False)
        
        Args:
            dog: DoG-filtered image
            exclude_border: Exclude keypoints at image border (default: False to match TrackMate)
            max_keypoints: Maximum number of keypoints to return (by quality)
        
        Returns array of (y, x) coordinates.
        """
        # Use TrackMate-style local maxima detection
        coordinates = find_local_maxima_trackmate_style(dog, self.threshold, exclude_border)
        
        # If max_keypoints is set, sort by quality (DoG value) and take top N
        if max_keypoints is not None and len(coordinates) > max_keypoints:
            # Get DoG values at each coordinate
            qualities = dog[coordinates[:, 0], coordinates[:, 1]]
            # Sort by quality descending and take top max_keypoints
            top_indices = np.argsort(qualities)[::-1][:max_keypoints]
            coordinates = coordinates[top_indices]
        
        return coordinates
    
    def refine_subpixel(self, dog: np.ndarray, coordinates: np.ndarray) -> np.ndarray:
        """
        Refine spot positions using quadratic fitting.
        
        Fits a 2D quadratic to the 3x3 neighborhood around each maximum
        and finds the sub-pixel peak location.
        
        Returns refined (y, x) coordinates as floats.
        """
        refined = []
        
        for (y, x) in coordinates:
            # Skip if too close to border for 3x3 neighborhood
            if y < 1 or y >= dog.shape[0] - 1 or x < 1 or x >= dog.shape[1] - 1:
                refined.append([float(y), float(x)])
                continue
            
            # Extract 3x3 neighborhood
            neighborhood = dog[y-1:y+2, x-1:x+2].astype(np.float64)
            
            # Quadratic fitting for sub-pixel localization
            # Using the method from TrackMate/ImgLib2
            # For each dimension, fit parabola through 3 points
            
            # Y direction: fit to column 1 (center column)
            dy = 0.0
            denom_y = 2 * neighborhood[1, 1] - neighborhood[0, 1] - neighborhood[2, 1]
            if abs(denom_y) > 1e-10:
                dy = (neighborhood[0, 1] - neighborhood[2, 1]) / (2 * denom_y)
                dy = np.clip(dy, -0.5, 0.5)  # Limit to within pixel
            
            # X direction: fit to row 1 (center row)
            dx = 0.0
            denom_x = 2 * neighborhood[1, 1] - neighborhood[1, 0] - neighborhood[1, 2]
            if abs(denom_x) > 1e-10:
                dx = (neighborhood[1, 0] - neighborhood[1, 2]) / (2 * denom_x)
                dx = np.clip(dx, -0.5, 0.5)  # Limit to within pixel
            
            refined.append([float(y) + dy, float(x) + dx])
        
        return np.array(refined) if refined else np.array([]).reshape(0, 2)
    
    def detect(self, image: np.ndarray, 
               exclude_border: bool = True, max_keypoints: Optional[int] = None,
               invert: bool = False) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Detect spots in image using TrackMate-style local maxima detection.
        
        This matches TrackMate's DoG detector:
        - Computes DoG = Gauss(sigma1) - Gauss(sigma2)
        - Finds ALL pixels that are local maxima in 3x3 neighborhood
        - NO min_distance suppression (unlike skimage.peak_local_max)
        
        Args:
            image: 2D grayscale image
            exclude_border: Exclude keypoints at image border (default: True)
            max_keypoints: Maximum number of keypoints (by quality, None = all)
            invert: If True, detect dark blobs on light background
        
        Returns:
            coordinates: (N, 2) array of (y, x) positions (sub-pixel if enabled)
            dog: DoG-filtered image
            preprocessed: Image after preprocessing (median filter if enabled)
        """
        # Work with a copy
        preprocessed = image.astype(np.float64).copy()
        
        # Debug: Print input image stats
        print(f"  DoG Input image stats: min={preprocessed.min():.4f}, max={preprocessed.max():.4f}, mean={preprocessed.mean():.4f}")
        
        # Optionally invert the image
        if invert:
            preprocessed = 1.0 - preprocessed if preprocessed.max() <= 1.0 else 255.0 - preprocessed
        
        # Preprocessing
        if self.do_median_filter:
            preprocessed = self.apply_median_filter(preprocessed)
        
        # DoG computation
        dog = self.compute_dog(preprocessed)
        
        # Debug: Print DoG response stats
        print(f"  DoG response stats: min={dog.min():.4f}, max={dog.max():.4f}, mean={dog.mean():.4f}")
        print(f"  Threshold: {self.threshold}, DoG values above threshold: {np.sum(dog > self.threshold)}")
        
        # Find maxima using TrackMate-style detection (no min_distance)
        coordinates = self.find_local_maxima(dog, exclude_border, max_keypoints)
        print(f"  Found {len(coordinates)} local maxima")
        
        # Sub-pixel refinement
        if self.do_subpixel and len(coordinates) > 0:
            coordinates = self.refine_subpixel(dog, coordinates)
        else:
            coordinates = coordinates.astype(np.float64)
        
        return coordinates, dog, preprocessed


def detect_dog_keypoints(
    frame: np.ndarray,
    sigma_low: float = 1.0,
    sigma_high: float = 2.0,
    threshold: float = 0.01,
    min_distance: int = 5,
    max_keypoints: Optional[int] = None,
    exclude_border: bool = True,  # Changed to bool to match DoGDetector default behavior
    invert: bool = False,
    median_filter: bool = False,
    subpixel: bool = False,
) -> np.ndarray:
    """Detect keypoints using Difference of Gaussians (DoG).
    
    This is a wrapper function that uses the DoGDetector class internally.
    The sigma_low and sigma_high parameters are used to compute the diameter.
    
    Args:
        frame: 2D grayscale image (H, W), values in [0, 1] or [0, 255]
        sigma_low: Standard deviation for the smaller Gaussian (sigma1 in TrackMate)
        sigma_high: Standard deviation for the larger Gaussian (sigma2 in TrackMate)
        threshold: Minimum DoG response to consider a keypoint (threshold_abs)
        min_distance: Minimum distance between detected keypoints
        max_keypoints: Maximum number of keypoints to return (None for all)
        exclude_border: Exclude keypoints at image borders (True to match DoGDetector behavior)
        invert: If True, detect dark blobs on light background
        median_filter: If True, apply 3x3 median filter before detection
        subpixel: If True, refine positions with quadratic fitting
        
    Returns:
        (N, 2) array of keypoint coordinates (x, y)
    """
    from skimage.feature import peak_local_max
    from scipy.ndimage import median_filter as scipy_median_filter
    
    print(f"[detect_dog_keypoints] sigma_low={sigma_low:.4f}, sigma_high={sigma_high:.4f}, threshold={threshold}")
    print(f"[detect_dog_keypoints] Frame shape: {frame.shape}, dtype: {frame.dtype}")
    print(f"[detect_dog_keypoints] Frame stats: min={frame.min():.4f}, max={frame.max():.4f}, mean={frame.mean():.4f}")
    
    # Work with a copy as float64
    img = frame.astype(np.float64).copy()
    
    # Optionally invert
    if invert:
        if img.max() <= 1.0:
            img = 1.0 - img
        else:
            img = img.max() - img
    
    # Optional median filter
    if median_filter:
        img = scipy_median_filter(img, size=3)
    
    # Compute DoG directly with the given sigmas (matching TrackMate exactly)
    gauss1 = gaussian_filter(img, sigma=sigma_low)
    gauss2 = gaussian_filter(img, sigma=sigma_high)
    dog = gauss1 - gauss2
    
    print(f"[detect_dog_keypoints] DoG response: min={dog.min():.6f}, max={dog.max():.6f}")
    print(f"[detect_dog_keypoints] Pixels above threshold: {np.sum(dog > threshold)}")
    
    # Find local maxima using peak_local_max
    peak_kwargs = {
        'min_distance': min_distance,
        'threshold_abs': threshold,
        'exclude_border': exclude_border,
    }
    if max_keypoints is not None:
        peak_kwargs['num_peaks'] = max_keypoints
    
    coordinates = peak_local_max(dog, **peak_kwargs)
    print(f"[detect_dog_keypoints] Found {len(coordinates)} peaks")
    
    if len(coordinates) == 0:
        return np.zeros((0, 2), dtype=np.float32)
    
    # Extract y, x coordinates
    y_coords = coordinates[:, 0].astype(np.float64)
    x_coords = coordinates[:, 1].astype(np.float64)
    
    # Optional sub-pixel refinement
    if subpixel:
        H, W = dog.shape
        x_refined = []
        y_refined = []
        for i in range(len(x_coords)):
            x_int = int(x_coords[i])
            y_int = int(y_coords[i])
            
            if y_int < 1 or y_int >= H - 1 or x_int < 1 or x_int >= W - 1:
                x_refined.append(x_coords[i])
                y_refined.append(y_coords[i])
                continue
            
            neighborhood = dog[y_int-1:y_int+2, x_int-1:x_int+2].astype(np.float64)
            
            # Y direction
            dy = 0.0
            denom_y = 2 * neighborhood[1, 1] - neighborhood[0, 1] - neighborhood[2, 1]
            if abs(denom_y) > 1e-10:
                dy = (neighborhood[0, 1] - neighborhood[2, 1]) / (2 * denom_y)
                dy = np.clip(dy, -0.5, 0.5)
            
            # X direction
            dx = 0.0
            denom_x = 2 * neighborhood[1, 1] - neighborhood[1, 0] - neighborhood[1, 2]
            if abs(denom_x) > 1e-10:
                dx = (neighborhood[1, 0] - neighborhood[1, 2]) / (2 * denom_x)
                dx = np.clip(dx, -0.5, 0.5)
            
            x_refined.append(x_coords[i] + dx)
            y_refined.append(y_coords[i] + dy)
        
        x_coords = np.array(x_refined)
        y_coords = np.array(y_refined)
    
    # Return as (x, y) coordinates
    keypoints = np.stack([x_coords, y_coords], axis=-1).astype(np.float32)
    
    return keypoints


def detect_dog_multiscale(
    frame: np.ndarray,
    sigma_range: Tuple[float, float] = (1.0, 4.0),
    num_scales: int = 4,
    threshold: float = 0.01,
    min_distance: int = 5,
    max_keypoints: Optional[int] = None,
    exclude_border: int = 5,
    invert: bool = False,
) -> np.ndarray:
    """Detect keypoints using multi-scale DoG.
    
    Args:
        frame: 2D grayscale image (H, W), normalized to [0, 1]
        sigma_range: (min_sigma, max_sigma) range for scale space
        num_scales: Number of scales to compute
        threshold: Minimum DoG response threshold
        min_distance: Minimum distance between keypoints
        max_keypoints: Maximum number of keypoints
        exclude_border: Border exclusion width
        invert: Detect dark blobs on light background
        
    Returns:
        (N, 2) array of keypoint coordinates (x, y)
    """
    H, W = frame.shape
    
    if invert:
        frame = 1.0 - frame
    
    # Generate sigma values
    sigma_low_vals = np.linspace(sigma_range[0], sigma_range[1], num_scales)
    sigma_ratio = 1.6  # Common ratio between consecutive sigmas
    
    all_keypoints = []
    all_responses = []
    
    for sigma_low in sigma_low_vals:
        sigma_high = sigma_low * sigma_ratio
        
        # Compute DoG
        blur_low = gaussian_filter(frame, sigma=sigma_low)
        blur_high = gaussian_filter(frame, sigma=sigma_high)
        dog = blur_low - blur_high
        dog_abs = np.abs(dog)
        
        # Find local maxima
        data_max = maximum_filter(dog_abs, size=min_distance)
        local_max_mask = (dog_abs == data_max) & (dog_abs > threshold)
        
        # Exclude border
        if exclude_border > 0:
            local_max_mask[:exclude_border, :] = False
            local_max_mask[-exclude_border:, :] = False
            local_max_mask[:, :exclude_border] = False
            local_max_mask[:, -exclude_border:] = False
        
        y_coords, x_coords = np.where(local_max_mask)
        responses = dog_abs[y_coords, x_coords]
        
        for x, y, r in zip(x_coords, y_coords, responses):
            all_keypoints.append([x, y])
            all_responses.append(r)
    
    if len(all_keypoints) == 0:
        return np.zeros((0, 2), dtype=np.float32)
    
    all_keypoints = np.array(all_keypoints)
    all_responses = np.array(all_responses)
    
    # Sort by response
    sorted_indices = np.argsort(all_responses)[::-1]
    all_keypoints = all_keypoints[sorted_indices]
    all_responses = all_responses[sorted_indices]
    
    # Non-maximum suppression
    keep = []
    for i in range(len(all_keypoints)):
        if max_keypoints is not None and len(keep) >= max_keypoints:
            break
        
        is_valid = True
        for j in keep:
            dist = np.linalg.norm(all_keypoints[i] - all_keypoints[j])
            if dist < min_distance:
                is_valid = False
                break
        
        if is_valid:
            keep.append(i)
    
    if len(keep) == 0:
        return np.zeros((0, 2), dtype=np.float32)
    
    return all_keypoints[keep].astype(np.float32)


# =============================================================================
# LOCOTRACK MODEL MANAGER
# =============================================================================

class LocoTrackModelManager:
    """Manages persistent LocoTrack model instance for inference."""
    
    def __init__(self, device=None, model_size="base", weights_path=None, adapter_path=None):
        """Initialize LocoTrack model.
        
        Args:
            device: torch.device to use (None for auto-detect)
            model_size: "base" or "small"
            weights_path: Path to model weights (None to auto-detect)
            adapter_path: Optional path to LoRA adapter (.pth file) for fine-tuned models
        """
        self.device = device if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model_size = model_size
        self.model = None
        self.adapter = None  # LoRA adapter (if loaded)
        self.adapter_path = adapter_path
        
        # Auto-detect weights path
        if weights_path is None:
            weights_file = f"locotrack_{model_size}.ckpt"
            candidates = [
                os.path.join(RIPPLE_ROOT, "locotrack_pytorch", "weights", weights_file),
                os.path.join(RIPPLE_ROOT, "models", "weights", weights_file),
                os.path.join(RIPPLE_ROOT, "locotrack", "weights", weights_file),
                os.path.join(RIPPLE_ROOT, "src", "main", "locotrack", "weights", weights_file),
            ]
            weights_path = next((p for p in candidates if os.path.isfile(p)), candidates[0])
        self.weights_path = weights_path
        
        self._load_model()
        
        # Load adapter if specified
        if adapter_path is not None:
            self.load_adapter(adapter_path)
    
    def _load_model(self):
        """Load LocoTrack model once and keep in memory."""
        print(f"Loading LocoTrack {self.model_size} model on {self.device}...")
        print(f"  CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"  CUDA device: {torch.cuda.get_device_name(0)}")
        
        try:
            from models.locotrack_model import load_model
            
            self.model = load_model(self.weights_path, model_size=self.model_size)
            self.model = self.model.to(self.device)
            # Ensure model is in float32 to avoid dtype mismatches
            self.model = self.model.float()
            self.model.eval()
            
            # Verify model is on the expected device
            model_device = next(self.model.parameters()).device
            print(f"  Model parameters are on: {model_device}")
            if model_device.type != self.device.type:
                print(f"  WARNING: Model is on {model_device} but expected {self.device}!")
            
            if self.device.type == "cuda":
                mem_after = torch.cuda.memory_allocated(0) / 1024**2
                print(f"  GPU memory after model load: {mem_after:.1f} MB")
            
            print(f"✓ LocoTrack model loaded successfully from {self.weights_path}")
        except Exception as e:
            print(f"ERROR: Failed to load LocoTrack model: {e}", file=sys.stderr)
            raise

    def load_adapter(self, adapter_path: str) -> bool:
        """Load a LoRA adapter for fine-tuned inference.
        
        Args:
            adapter_path: Path to the LoRA adapter (.pth file)
            
        Returns:
            True if adapter loaded successfully, False otherwise
        """
        if self.model is None:
            print("ERROR: Cannot load adapter - model not loaded", file=sys.stderr)
            return False
        
        adapter_path = Path(adapter_path) if isinstance(adapter_path, str) else adapter_path
        
        if not adapter_path.exists():
            print(f"WARNING: Adapter file not found: {adapter_path}", file=sys.stderr)
            return False
        
        try:
            # Import LoRA adapter
            import sys
            script_dir = Path(__file__).parent
            if str(script_dir) not in sys.path:
                sys.path.insert(0, str(script_dir))
            
            from locotrack_lora import LoRAAdapter
            
            # Create adapter wrapper and load weights
            # First, read the adapter config to get the rank
            adapter_data = torch.load(adapter_path, map_location='cpu')
            config = adapter_data.get('config', {})
            rank = config.get('rank', 4)
            alpha = config.get('alpha', float(rank))
            
            # Create adapter
            self.adapter = LoRAAdapter(self.model, rank=rank, alpha=alpha)
            
            # Load the adapter weights
            self.adapter.load(adapter_path)
            
            # Move adapter to device
            self.adapter.to(self.device)
            self.adapter.eval()
            
            self.adapter_path = str(adapter_path)
            print(f"✓ LoRA adapter loaded from {adapter_path}")
            
            return True
            
        except Exception as e:
            print(f"ERROR: Failed to load adapter: {e}", file=sys.stderr)
            import traceback
            traceback.print_exc(file=sys.stderr)
            self.adapter = None
            return False

    def unload_adapter(self):
        """Unload the current LoRA adapter."""
        if self.adapter is not None:
            # Need to reload the model to remove LoRA modifications
            self._load_model()
            self.adapter = None
            self.adapter_path = None
            print("✓ LoRA adapter unloaded")

    def ensure_loaded(self):
        """Ensure the model is loaded (reload if previously unloaded)."""
        if self.model is None:
            self._load_model()
            # Reload adapter if it was previously loaded
            if self.adapter_path is not None:
                self.load_adapter(self.adapter_path)

    def unload_model(self) -> bool:
        """Unload LocoTrack model to free VRAM. Returns True if unloaded."""
        if self.model is None:
            print("  LocoTrack model already unloaded")
            return False

        try:
            if self.device.type == "cuda" and torch.cuda.is_available():
                mem_before = torch.cuda.memory_allocated(0) / 1024**2
            else:
                mem_before = 0.0

            try:
                self.model = self.model.cpu()
            except Exception:
                pass

            del self.model
            self.model = None
            self.adapter = None  # Clear adapter reference too

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            if self.device.type == "cuda" and torch.cuda.is_available():
                mem_after = torch.cuda.memory_allocated(0) / 1024**2
                print(f"✓ LocoTrack model unloaded, freed {mem_before - mem_after:.1f} MB")
            else:
                print("✓ LocoTrack model unloaded")

            return True
        except Exception as e:
            print(f"WARNING: Failed to unload LocoTrack model cleanly: {e}", file=sys.stderr)
            # Best-effort: ensure pointer cleared
            self.model = None
            return True
    
    def track_points(
        self,
        video: np.ndarray,
        query_points: np.ndarray,
        query_chunk_size: int = 64,
        cancel_check: Optional[Callable[[], None]] = None,
    ) -> Dict[str, np.ndarray]:
        """Track query points through the video.
        
        Args:
            video: (T, H, W) or (T, H, W, 3) video array, values in [0, 1] or [0, 255]
            query_points: (N, 3) array of query points as (frame, y, x)
            query_chunk_size: Chunk size for processing queries
            
        Returns:
            Dict with:
                - 'tracks': (N, T, 2) array of tracked positions (x, y)
                - 'occlusion': (N, T) boolean array of occlusion predictions
        """
        # Model may have been unloaded by server clear_memory(). Reload lazily.
        self.ensure_loaded()

        # The older, known-good integration uses the model's `inference()` helper,
        # which handles resizing and internal preprocessing consistently.
        # Some model builds can be dramatically slower (or appear to stall) when
        # calling `forward()` directly.
        if hasattr(self.model, "inference"):
            if cancel_check is not None:
                cancel_check()

            # Prepare video: ensure (T, H, W, 3)
            if video.ndim == 3:
                # Grayscale (T,H,W) -> (T,H,W,3)
                video_rgb = np.stack([video, video, video], axis=-1)
            else:
                # Already RGB or multi-channel; take first 3 channels if needed
                if video.shape[-1] == 3:
                    video_rgb = video
                else:
                    video_rgb = video[..., :3]

            T, H, W, _ = video_rgb.shape

            # Normalize to [0, 1]
            if video_rgb.max() > 1.0:
                video_rgb = video_rgb.astype(np.float32) / 255.0
            else:
                video_rgb = video_rgb.astype(np.float32)

            # Add batch dimension: (1, T, H, W, 3)
            video_batch = video_rgb[np.newaxis, ...]

            # Query points tensor in (t, y, x) format
            query_tensor = torch.from_numpy(query_points.astype(np.float32)).unsqueeze(0)

            print(f"[LocoTrack] Running inference: video shape {video_batch.shape}, {len(query_points)} query points")
            with torch.no_grad():
                output = self.model.inference(
                    video_batch,
                    query_tensor,
                    query_chunk_size=query_chunk_size,
                    resolution=(256, 256),
                    query_format='tyx',
                )

            tracks_t = output['tracks'][0]
            occ_t = output['occlusion'][0]

            tracks_np = tracks_t.cpu().numpy() if hasattr(tracks_t, 'cpu') else np.asarray(tracks_t)
            occlusion_np = occ_t.cpu().numpy() if hasattr(occ_t, 'cpu') else np.asarray(occ_t)

            # Explicitly delete and clear cache
            del output, query_tensor, video_batch, video_rgb
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            print(f"[LocoTrack] Inference complete: tracks shape {tracks_np.shape}")
            occlusion_prob = occlusion_np.astype(np.float32)
            return {
                'tracks': tracks_np.astype(np.float32),
                'occlusion': occlusion_np,
                'occlusion_prob': occlusion_prob,
            }

        # Fallback: direct forward() path (kept for compatibility if inference() is unavailable)
        # Get original dimensions before any transformation
        if video.ndim == 3:
            T, H, W = video.shape
        else:
            T, H, W, _ = video.shape

        target_resolution = (256, 256)

        # Normalize to [0, 1]
        if video.max() > 1.0:
            video = video.astype(np.float32) / 255.0
        else:
            video = video.astype(np.float32)

        if video.ndim == 3:
            video_tensor = torch.from_numpy(video).unsqueeze(1).to(self.device)
            if (H, W) != target_resolution:
                video_tensor = F.interpolate(
                    video_tensor, target_resolution,
                    mode='bilinear', align_corners=False
                )
            video_tensor = video_tensor.expand(-1, 3, -1, -1).contiguous()
        else:
            video_tensor = torch.from_numpy(video).permute(0, 3, 1, 2).to(self.device)
            if (H, W) != target_resolution:
                video_tensor = F.interpolate(
                    video_tensor, target_resolution,
                    mode='bilinear', align_corners=False
                )

        video_tensor = video_tensor.permute(0, 2, 3, 1).unsqueeze(0)
        video_tensor = video_tensor * 2 - 1

        query_points_scaled = query_points.copy()
        query_points_scaled[:, 1] = query_points_scaled[:, 1] / H * target_resolution[0]
        query_points_scaled[:, 2] = query_points_scaled[:, 2] / W * target_resolution[1]
        query_tensor = torch.from_numpy(query_points_scaled.astype(np.float32)).unsqueeze(0).to(self.device)

        print(f"[LocoTrack] Running forward(): video {T}x{H}x{W} -> {target_resolution}, {len(query_points)} query points")
        if cancel_check is not None:
            cancel_check()

        with torch.no_grad():
            output = self.model.forward(
                video_tensor,
                query_tensor,
                query_chunk_size=query_chunk_size,
                cancel_check=cancel_check,
            )

        tracks = output['tracks']
        occlusion_logits = output['occlusion']
        expected_dist = output['expected_dist']

        tracks = tracks * torch.tensor([W / target_resolution[1], H / target_resolution[0]], device=tracks.device)
        pred_occ = torch.sigmoid(occlusion_logits)
        pred_occ = 1 - (1 - pred_occ) * (1 - torch.sigmoid(expected_dist))
        pred_occ = pred_occ > 0.5

        tracks_np = tracks[0].cpu().numpy()
        occlusion_np = pred_occ[0].cpu().numpy()

        del output, query_tensor, video_tensor, tracks, occlusion_logits, expected_dist, pred_occ
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        print(f"[LocoTrack] Forward complete: tracks shape {tracks_np.shape}")
        occlusion_prob = occlusion_np.astype(np.float32)
        return {
            'tracks': tracks_np.astype(np.float32),
            'occlusion': occlusion_np,
            'occlusion_prob': occlusion_prob,
        }


# =============================================================================
# TRAJECTORY FILTERING
# =============================================================================

def filter_trajectories_by_occlusion(
    tracks: np.ndarray,
    occlusion: np.ndarray,
    max_occlusion_ratio: float = 0.3,
    verbose: bool = True,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Filter out trajectories with high occlusion percentage.
    
    Args:
        tracks: (N, T, 2) array of tracked positions
        occlusion: (N, T) boolean array of occlusion predictions
        max_occlusion_ratio: Maximum fraction of frames that can be occluded
        verbose: Print filtering statistics
        
    Returns:
        Tuple of:
        - filtered_tracks: (M, T, 2) filtered tracks
        - filtered_occlusion: (M, T) filtered occlusion
        - keep_indices: (M,) indices of kept tracks
    """
    N, T = occlusion.shape
    
    # Compute occlusion ratio for each trajectory
    occlusion_ratios = occlusion.sum(axis=1) / T
    
    # Keep trajectories with low occlusion
    keep_mask = occlusion_ratios <= max_occlusion_ratio
    keep_indices = np.where(keep_mask)[0]
    
    if verbose:
        print(f"[filter_occlusion] Total trajectories: {N}")
        print(f"[filter_occlusion] Trajectories with <= {max_occlusion_ratio*100:.0f}% occlusion: {len(keep_indices)}")
        if len(keep_indices) > 0:
            print(f"[filter_occlusion] Average occlusion ratio (kept): {occlusion_ratios[keep_mask].mean():.2%}")
        else:
            print(f"[filter_occlusion] WARNING: No trajectories passed filter!")
    
    filtered_tracks = tracks[keep_indices]
    filtered_occlusion = occlusion[keep_indices]
    
    return filtered_tracks, filtered_occlusion, keep_indices


# =============================================================================
# TEMPORAL SMOOTHING (SPECTRAL INTERPOLATION)
# =============================================================================

def spectral_smooth_trajectory(
    trajectory: np.ndarray,
    occlusion: np.ndarray,
    smooth_factor: float = 0.1
) -> np.ndarray:
    """Apply spectral smoothing to a single trajectory using DCT.
    
    This reduces high-frequency jitter in the tracking output.
    
    Args:
        trajectory: (T, 2) array of positions (x, y)
        occlusion: (T,) boolean array of occlusion flags
        smooth_factor: Amount of high-frequency suppression (0 = none, 1 = heavy)
        
    Returns:
        (T, 2) smoothed trajectory
    """
    T = len(trajectory)
    smoothed = trajectory.copy()
    
    if smooth_factor <= 0 or T < 4:
        return smoothed
    
    for dim in range(2):
        signal = trajectory[:, dim].copy()
        
        # For occluded frames, interpolate linearly first
        valid = ~occlusion
        if not valid.all() and valid.sum() >= 2:
            valid_indices = np.where(valid)[0]
            invalid_indices = np.where(~valid)[0]
            signal[invalid_indices] = np.interp(
                invalid_indices, valid_indices, signal[valid_indices]
            )
        
        # Apply DCT
        coeffs = dct(signal, type=2, norm='ortho')
        
        # Create frequency-dependent suppression
        freqs = np.arange(T)
        # Suppress high frequencies more strongly
        cutoff = int(T * (1 - smooth_factor))
        suppression = np.ones(T)
        if cutoff < T:
            # Gradual rolloff
            rolloff_start = max(1, cutoff // 2)
            suppression[rolloff_start:] = np.exp(-((freqs[rolloff_start:] - rolloff_start) / max(1, T - rolloff_start)) ** 2 * 3)
        
        coeffs *= suppression
        
        # Inverse DCT
        smoothed[:, dim] = idct(coeffs, type=2, norm='ortho')
    
    return smoothed


def spectral_smooth_all_trajectories(
    tracks: np.ndarray,
    occlusion: np.ndarray,
    smooth_factor: float = 0.1,
    verbose: bool = True
) -> np.ndarray:
    """Apply spectral smoothing to all trajectories.
    
    Args:
        tracks: (N, T, 2) array of tracked positions
        occlusion: (N, T) boolean array of occlusion predictions
        smooth_factor: Smoothing strength (0-1)
        verbose: Print progress
        
    Returns:
        (N, T, 2) smoothed tracks
    """
    N, T, _ = tracks.shape
    smoothed = np.zeros_like(tracks)
    
    for i in range(N):
        smoothed[i] = spectral_smooth_trajectory(
            tracks[i], occlusion[i], smooth_factor
        )
    
    if verbose:
        # Compute smoothing statistics
        orig_velocities = np.diff(tracks, axis=1)
        smooth_velocities = np.diff(smoothed, axis=1)
        orig_jitter = np.std(np.diff(orig_velocities, axis=1))
        smooth_jitter = np.std(np.diff(smooth_velocities, axis=1))
        print(f"[spectral_smooth] Velocity jitter: {orig_jitter:.4f} -> {smooth_jitter:.4f} "
              f"({(1 - smooth_jitter/max(orig_jitter, 1e-8))*100:.1f}% reduction)")
    
    return smoothed


# =============================================================================
# DENSE OPTICAL FLOW GENERATION
# =============================================================================

def compute_trajectory_velocities(
    tracks: np.ndarray
) -> np.ndarray:
    """Compute velocities from tracked positions.
    
    Args:
        tracks: (N, T, 2) array of tracked positions (x, y)
        
    Returns:
        (N, T-1, 2) array of velocities (dx, dy)
    """
    velocities = np.diff(tracks, axis=1)
    return velocities


def generate_dense_flow_rbf_gpu(
    positions: np.ndarray,
    velocities: np.ndarray,
    H: int,
    W: int,
    frame_idx: int,
    device: torch.device,
    smoothing: float = 10.0,
    kernel: str = 'gaussian',
    *,
    cancel_check: Optional[Callable[[], None]] = None,
    max_chunk_elements: int = 10_000_000,
) -> np.ndarray:
    """Generate dense optical flow field using GPU-accelerated interpolation.
    
    Supports multiple kernel types for different smoothness characteristics.
    
    Args:
        positions: (N, T, 2) array of particle positions (x, y)
        velocities: (N, T-1, 2) array of particle velocities
        H, W: Height and width of output flow field
        frame_idx: Frame index (0 to T-2)
        device: torch device for GPU acceleration
        smoothing: Smoothing/bandwidth parameter (higher = smoother)
        kernel: Interpolation kernel type ('gaussian', 'idw', 'wendland')
        
    Returns:
        (H, W, 2) dense optical flow field
    """
    # Get positions and velocities at this frame
    pts = positions[:, frame_idx, :]  # (N, 2) - x, y
    vels = velocities[:, frame_idx, :]  # (N, 2) - dx, dy
    
    # Filter out invalid points (NaN)
    valid = ~(np.isnan(pts).any(axis=1) | np.isnan(vels).any(axis=1))
    pts_valid = pts[valid]
    vels_valid = vels[valid]
    
    if len(pts_valid) < 3:
        return np.zeros((H, W, 2), dtype=np.float32)
    
    # Filter velocity outliers (remove extreme velocities)
    vel_magnitudes = np.linalg.norm(vels_valid, axis=1)
    if len(vel_magnitudes) > 10:
        vel_median = np.median(vel_magnitudes)
        vel_mad = np.median(np.abs(vel_magnitudes - vel_median))  # Median absolute deviation
        vel_threshold = vel_median + 5 * max(vel_mad, 1.0)  # 5 MAD threshold
        inlier_mask = vel_magnitudes < vel_threshold
        if inlier_mask.sum() >= 3:
            pts_valid = pts_valid[inlier_mask]
            vels_valid = vels_valid[inlier_mask]
    
    # Move to GPU
    pts_t = torch.from_numpy(pts_valid).float().to(device)  # (N, 2)
    vels_t = torch.from_numpy(vels_valid).float().to(device)  # (N, 2)

    # NOTE:
    # A naive implementation allocates dist_sq of shape (H, W, N). For typical sizes
    # (e.g., 1024x1024 with a few hundred seeds) this can be multiple GB of GPU
    # intermediates and can appear to hang. We compute in row-chunks instead.
    n_pts = int(pts_t.shape[0])
    if n_pts <= 0:
        return np.zeros((H, W, 2), dtype=np.float32)

    # Choose chunk height so (chunkH * W * N) stays bounded.
    denom = max(1, int(W) * n_pts)
    chunk_h = int(max(1, min(H, max_chunk_elements // denom)))

    # Precompute broadcastable tensors
    xx = torch.arange(W, device=device, dtype=torch.float32).view(1, W, 1)  # (1,W,1)
    pts_x = pts_t[:, 0].view(1, 1, -1)  # (1,1,N)
    pts_y = pts_t[:, 1].view(1, 1, -1)  # (1,1,N)
    vels_x = vels_t[:, 0].view(1, 1, -1)  # (1,1,N)
    vels_y = vels_t[:, 1].view(1, 1, -1)  # (1,1,N)

    flow = torch.empty((H, W, 2), device=device, dtype=torch.float32)

    for y0 in range(0, H, chunk_h):
        if cancel_check is not None:
            cancel_check()
        y1 = min(H, y0 + chunk_h)
        yy = torch.arange(y0, y1, device=device, dtype=torch.float32).view(-1, 1, 1)  # (chunkH,1,1)

        # dist_sq: (chunkH, W, N)
        dist_sq = (xx - pts_x) ** 2 + (yy - pts_y) ** 2

        # weights: (chunkH, W, N)
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

        weights_sum = weights.sum(dim=-1, keepdim=True) + 1e-8
        weights_norm = weights / weights_sum

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
    """Generate dense optical flow field using RBF interpolation (CPU fallback).
    
    Args:
        positions: (N, T, 2) array of particle positions (x, y)
        velocities: (N, T-1, 2) array of particle velocities
        H, W: Height and width of output flow field
        frame_idx: Frame index (0 to T-2)
        smoothing: RBF smoothing parameter
        kernel: RBF kernel type
        
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

    # Some kernels require epsilon in SciPy's RBFInterpolator.
    # Use smoothing as a reasonable default bandwidth.
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
        # Limit to local neighbors for speed on large grids.
        # SciPy will raise if neighbors > number of points.
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
    epsilon: float = 1e-6
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
    tracks: np.ndarray,
    H: int,
    W: int,
    smoothing: float = 10.0,
    kernel: str = 'gaussian',
    gaussian_sigma: Optional[float] = None,
    verbose: bool = True,
    device: Optional[torch.device] = None,
    use_gpu: bool = True,
    cancel_check: Optional[Callable[[], None]] = None,
) -> np.ndarray:
    """Generate dense optical flow field for all frames.
    
    Args:
        tracks: (N, T, 2) tracked positions (x, y)
        H, W: Height and width of output flow field
        smoothing: RBF/IDW smoothing parameter
        kernel: Interpolation kernel type:
            - 'gaussian': GPU-accelerated Gaussian kernel (fast, default)
            - 'gaussian_rbf': Scipy Gaussian RBF interpolation (GRBF, accurate)
            - 'idw': GPU-accelerated Inverse Distance Weighting
            - 'wendland': GPU-accelerated Wendland kernel (compact support)
            - 'thin_plate_spline': Scipy thin-plate spline RBF
            - 'multiquadric', 'inverse_multiquadric', 'linear', 'cubic', 'quintic': Other scipy RBF kernels
        gaussian_sigma: Optional Gaussian smoothing of output flow
        verbose: Print progress
        device: torch device for GPU acceleration (None for auto-detect)
        use_gpu: Whether to use GPU-accelerated interpolation
        
    Returns:
        (T-1, H, W, 2) dense optical flow field
    """
    N, T, _ = tracks.shape
    
    # Auto-detect device
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Fall back to CPU if no CUDA available
    if use_gpu and device.type != "cuda":
        if verbose:
            print("[generate_flow] CUDA not available, using CPU interpolation")
        use_gpu = False
    
    # Map gaussian_rbf to scipy's gaussian kernel name (for scipy fallback)
    scipy_kernel = 'gaussian' if kernel == 'gaussian_rbf' else kernel
    
    # Map gaussian_rbf to gaussian for GPU path (they use the same Gaussian weighting)
    gpu_kernel = 'gaussian' if kernel == 'gaussian_rbf' else kernel
    
    # GPU-supported kernels (gaussian_rbf is mapped to gaussian for GPU acceleration)
    gpu_kernels = ['gaussian', 'gaussian_rbf', 'idw', 'wendland']
    
    if verbose:
        if use_gpu and kernel in gpu_kernels:
            print(f"[generate_flow] Using GPU-accelerated interpolation on {device} (kernel={kernel})")
        else:
            print(f"[generate_flow] Using scipy RBF interpolation (kernel={scipy_kernel})")
    
    # Compute velocities
    velocities = compute_trajectory_velocities(tracks)

    # Precompute query grid once (used by scipy path). This avoids allocating
    # a (H*W, 2) array for every frame.
    yy, xx = np.mgrid[0:H, 0:W]
    query_pts = np.stack([xx.ravel(), yy.ravel()], axis=1)
    
    # Generate flow for each frame
    flows = np.zeros((T - 1, H, W, 2), dtype=np.float32)
    
    for t in range(T - 1):
        if cancel_check is not None:
            cancel_check()
        # Use GPU for supported kernels, otherwise fall back to scipy RBF
        if use_gpu and kernel in gpu_kernels:
            flows[t] = generate_dense_flow_rbf_gpu(
                tracks, velocities, H, W, t, device,
                smoothing=smoothing, kernel=gpu_kernel,
                cancel_check=cancel_check,
            )
        else:
            # Only use neighbors for large point sets (>100) where KDTree helps.
            # For small sets, the KDTree overhead actually slows things down.
            neighbors = 32 if N > 100 else None
            flows[t] = generate_dense_flow_rbf(
                tracks, velocities, H, W, t,
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

def compute_locotrack_optical_flow(
    video: np.ndarray,
    locotrack_manager: LocoTrackModelManager,
    sigma_low: float = 1.0,
    sigma_high: float = 2.0,
    dog_threshold: float = 0.01,
    min_distance: int = 5,
    max_keypoints: Optional[int] = 500,
    exclude_border: int = 0,
    max_occlusion_ratio: float = 0.5,
    flow_smoothing: float = 15.0,
    gaussian_sigma: Optional[float] = 2.0,
    temporal_smooth_factor: float = 0.1,
    kernel: str = 'gaussian',
    invert: bool = False,
    verbose: bool = True,
    seed_frames: Optional[List[int]] = None,
    median_filter: bool = False,
    subpixel: bool = False,
    dog_video_raw: Optional[np.ndarray] = None,
    cancel_check: Optional[Callable[[], None]] = None,
) -> Tuple[np.ndarray, Dict]:
    """High-level function to compute optical flow using LocoTrack.
    
    Complete pipeline:
    1. Detect seeds on specified frames using DoG (default: first frame)
    2. Track all seeds throughout video using LocoTrack
    3. Filter out trajectories with high occlusion
    4. Apply temporal smoothing (spectral) to reduce jitter
    5. Generate dense optical flow field via RBF interpolation
    
    Args:
        video: (T, H, W) video array (normalized to [0, 1] or uint8)
        locotrack_manager: Initialized LocoTrackModelManager
        sigma_low: DoG lower sigma
        sigma_high: DoG higher sigma
        dog_threshold: DoG detection threshold
        min_distance: Minimum distance between seeds
        max_keypoints: Maximum number of seeds
        exclude_border: Exclude seeds near border
        max_occlusion_ratio: Maximum occlusion ratio for filtering (default 0.5)
        flow_smoothing: RBF/kernel smoothing bandwidth (default 15.0)
        gaussian_sigma: Gaussian post-smoothing of output flow
        temporal_smooth_factor: Temporal smoothing strength (0-1, 0=none)
        kernel: Interpolation kernel type:
            - 'gaussian': GPU-accelerated Gaussian kernel (fast, default)
            - 'gaussian_rbf': Scipy Gaussian RBF interpolation (GRBF, accurate)
            - 'idw': GPU-accelerated Inverse Distance Weighting
            - 'wendland': GPU-accelerated Wendland kernel (compact support)
            - 'thin_plate_spline': Scipy thin-plate spline RBF
            - 'multiquadric', 'inverse_multiquadric', 'linear', 'cubic', 'quintic': Other scipy RBF kernels
        invert: Detect dark blobs on light background
        verbose: Print progress information
        seed_frames: List of frame indices to detect seeds (None = [0])
        median_filter: Apply 3x3 median filter before DoG detection (TrackMate option)
        subpixel: Refine positions with quadratic fitting (TrackMate option)
        
    Returns:
        Tuple of:
        - flows: (T-1, H, W, 2) dense optical flow field
        - info: Dictionary with tracking statistics
    """
    T, H, W = video.shape[:3]

    if cancel_check is not None:
        cancel_check()
    
    # Normalize video if needed
    if video.max() > 1.0:
        video = video.astype(np.float32) / 255.0
    
    if verbose:
        print(f"[locotrack_flow] Processing video: {T} frames, {H}x{W} pixels")
    
    # Default seed frames
    if seed_frames is None:
        seed_frames = [0]
    
    # Step 1: Detect seeds on specified frames using DoG (optionally on raw values)
    if verbose:
        print(f"[locotrack_flow] Step 1: Detecting seeds with DoG on frames {seed_frames}...")
        if median_filter:
            print("[locotrack_flow]   Using median filter pre-processing")
        if subpixel:
            print("[locotrack_flow]   Using sub-pixel localization")
    
    all_seeds = []
    all_seed_frames = []
    
    for frame_idx in seed_frames:
        if cancel_check is not None:
            cancel_check()
        if frame_idx >= T:
            continue
        frame = dog_video_raw[frame_idx] if dog_video_raw is not None else video[frame_idx]
        seeds = detect_dog_keypoints(
            frame,
            sigma_low=sigma_low,
            sigma_high=sigma_high,
            threshold=dog_threshold,
            min_distance=min_distance,
            max_keypoints=max_keypoints,
            exclude_border=exclude_border,
            invert=invert,
            median_filter=median_filter,
            subpixel=subpixel,
        )
        
        if len(seeds) > 0:
            all_seeds.append(seeds)
            all_seed_frames.extend([frame_idx] * len(seeds))
            if verbose:
                print(f"[locotrack_flow]   Frame {frame_idx}: detected {len(seeds)} seeds")
    
    if len(all_seeds) == 0:
        warnings.warn("No seeds detected! Check DoG parameters.")
        return np.zeros((T - 1, H, W, 2), dtype=np.float32), {"error": "no_seeds"}
    
    # Combine seeds from all frames
    all_seeds = np.vstack(all_seeds)
    all_seed_frames = np.array(all_seed_frames)
    n_seeds = len(all_seeds)
    
    if verbose:
        print(f"[locotrack_flow]   Total detected {n_seeds} seeds across {len(seed_frames)} frames")
    
    # Step 2: Prepare query points for LocoTrack
    # Query points format: (N, 3) as (frame, y, x)
    query_points = np.zeros((n_seeds, 3), dtype=np.float32)
    query_points[:, 0] = all_seed_frames
    query_points[:, 1] = all_seeds[:, 1]  # y
    query_points[:, 2] = all_seeds[:, 0]  # x
    
    # Step 3: Track seeds using LocoTrack
    if verbose:
        print("[locotrack_flow] Step 2: Tracking seeds with LocoTrack...")

    if cancel_check is not None:
        cancel_check()

    tracking_result = locotrack_manager.track_points(video, query_points, cancel_check=cancel_check)
    tracks = tracking_result['tracks']  # (N, T, 2) as (x, y)
    occlusion = tracking_result['occlusion']  # (N, T)
    
    # Free tracking_result dict (we extracted what we need)
    del tracking_result
    
    # Save average occlusion before we delete the array
    avg_occlusion_ratio = float(occlusion.mean())
    
    if verbose:
        print(f"[locotrack_flow]   Tracking complete, average occlusion: {avg_occlusion_ratio:.2%}")
    
    # Step 4: Filter trajectories by occlusion
    if verbose:
        print("[locotrack_flow] Step 3: Filtering trajectories by occlusion...")

    if cancel_check is not None:
        cancel_check()
    
    filtered_tracks, filtered_occlusion, keep_indices = filter_trajectories_by_occlusion(
        tracks, occlusion, max_occlusion_ratio=max_occlusion_ratio, verbose=verbose
    )
    
    # Free unfiltered tracks (we have filtered versions now)
    del tracks, occlusion
    
    n_filtered = len(keep_indices)
    if n_filtered == 0:
        warnings.warn("No trajectories passed occlusion filter! Try increasing max_occlusion_ratio.")
        return np.zeros((T - 1, H, W, 2), dtype=np.float32), {"error": "no_valid_trajectories"}
    
    # Step 5: Apply temporal smoothing (spectral)
    if temporal_smooth_factor > 0:
        if verbose:
            print(f"[locotrack_flow] Step 4: Applying temporal smoothing (factor={temporal_smooth_factor})...")

        if cancel_check is not None:
            cancel_check()
        smoothed_tracks = spectral_smooth_all_trajectories(
            filtered_tracks, filtered_occlusion, 
            smooth_factor=temporal_smooth_factor, verbose=verbose
        )
        # Free unsmoothed tracks
        del filtered_tracks
    else:
        smoothed_tracks = filtered_tracks
        if verbose:
            print("[locotrack_flow] Step 4: Skipping temporal smoothing (factor=0)")
    
    # Step 6: Generate dense flow
    if verbose:
        print("[locotrack_flow] Step 5: Generating dense optical flow...")

    if cancel_check is not None:
        cancel_check()
    
    flows = generate_dense_flow_all_frames(
        smoothed_tracks, H, W,
        smoothing=flow_smoothing,
        kernel=kernel,
        gaussian_sigma=gaussian_sigma,
        verbose=verbose,
        cancel_check=cancel_check
    )
    
    # Compile statistics (use saved avg_occlusion_ratio since occlusion array was freed)
    info = {
        "total_seeds": n_seeds,
        "valid_trajectories": n_filtered,
        "filtered_trajectories": n_seeds - n_filtered,
        "average_occlusion_ratio": avg_occlusion_ratio,
        "frames": T,
        "height": H,
        "width": W,
        "seed_frames": seed_frames,
        "parameters": {
            "sigma_low": sigma_low,
            "sigma_high": sigma_high,
            "dog_threshold": dog_threshold,
            "min_distance": min_distance,
            "max_keypoints": max_keypoints,
            "max_occlusion_ratio": max_occlusion_ratio,
            "flow_smoothing": flow_smoothing,
            "gaussian_sigma": gaussian_sigma,
            "temporal_smooth_factor": temporal_smooth_factor,
            "kernel": kernel,
        }
    }
    
    if verbose:
        print(f"[locotrack_flow] Complete! Flow shape: {flows.shape}")
    
    return flows, info


# =============================================================================
# PREVIEW FUNCTIONS
# =============================================================================

def generate_dog_preview(
    video: np.ndarray,
    sigma_low: float = 1.0,
    sigma_high: float = 2.0,
    dog_threshold: float = 0.01,
    min_distance: int = 5,
    max_keypoints: Optional[int] = 500,
    exclude_border: int = 0,
    invert: bool = False,
    num_preview_frames: int = 5,
    median_filter: bool = False,
    subpixel: bool = False,
) -> Tuple[np.ndarray, Dict]:
    """Generate a preview visualization of DoG detection.
    
    Shows the first frame with detected keypoints, plus intermediate frames
    with projected positions (no actual tracking).
    
    Args:
        video: (T, H, W) or (T, H, W, 3) video array
        sigma_low: DoG lower sigma
        sigma_high: DoG higher sigma
        dog_threshold: Detection threshold
        min_distance: Minimum distance between keypoints
        max_keypoints: Maximum number of keypoints
        exclude_border: Border exclusion width
        invert: Detect dark blobs on light background
        num_preview_frames: Number of frames to show in preview
        median_filter: Apply 3x3 median filter before detection (TrackMate option)
        subpixel: Refine positions with quadratic fitting (TrackMate option)
        
    Returns:
        Tuple of:
        - preview_image: (H, W*num_frames, 3) RGB preview image
        - info: Dictionary with detection statistics
    """
    import cv2
    
    # Get first frame
    if video.ndim == 4:
        T, H, W, C = video.shape
        first_frame = video[0].mean(axis=-1) if C > 1 else video[0, :, :, 0]
        video_rgb = video.copy()
    else:
        T, H, W = video.shape
        first_frame = video[0]
        # Convert to RGB
        video_rgb = np.stack([video, video, video], axis=-1)
    
    # Normalize
    if first_frame.max() > 1.0:
        first_frame = first_frame.astype(np.float32) / 255.0
    if video_rgb.max() > 1.0:
        video_rgb = (video_rgb / 255.0 * 255).astype(np.uint8)
    elif video_rgb.max() <= 1.0:
        video_rgb = (video_rgb * 255).astype(np.uint8)
    
    # Detect keypoints on first frame
    keypoints = detect_dog_keypoints(
        first_frame,
        sigma_low=sigma_low,
        sigma_high=sigma_high,
        threshold=dog_threshold,
        min_distance=min_distance,
        max_keypoints=max_keypoints,
        exclude_border=exclude_border,
        invert=invert,
        median_filter=median_filter,
        subpixel=subpixel,
    )
    
    n_keypoints = len(keypoints)
    
    # Generate colors for each keypoint
    np.random.seed(42)
    colors = []
    for i in range(n_keypoints):
        hue = (i * 37) % 180  # Spread colors
        color = cv2.cvtColor(np.array([[[hue, 255, 255]]], dtype=np.uint8), cv2.COLOR_HSV2RGB)[0, 0]
        colors.append(tuple(int(c) for c in color))
    
    # Select frames to show
    frame_indices = np.linspace(0, T - 1, num_preview_frames, dtype=int)
    
    preview_frames = []
    for idx, frame_idx in enumerate(frame_indices):
        frame_rgb = video_rgb[frame_idx].copy()
        
        # Draw keypoints (only on first frame, they are not tracked yet)
        if frame_idx == 0:
            for i, (x, y) in enumerate(keypoints):
                x_int, y_int = int(x), int(y)
                color = colors[i] if i < len(colors) else (0, 255, 0)
                cv2.circle(frame_rgb, (x_int, y_int), 5, color, 2)
                cv2.circle(frame_rgb, (x_int, y_int), 2, color, -1)
        
        # Add frame number label
        cv2.putText(frame_rgb, f"Frame {frame_idx}", (10, 25),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        if frame_idx == 0:
            cv2.putText(frame_rgb, f"Seeds: {n_keypoints}", (10, 50),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 255, 100), 1)
        
        preview_frames.append(frame_rgb)
    
    # Stack frames horizontally
    preview_image = np.hstack(preview_frames)
    
    # Compile statistics
    info = {
        "total_keypoints": n_keypoints,
        "frames": T,
        "height": H,
        "width": W,
        "parameters": {
            "sigma_low": sigma_low,
            "sigma_high": sigma_high,
            "dog_threshold": dog_threshold,
            "min_distance": min_distance,
            "max_keypoints": max_keypoints,
            "exclude_border": exclude_border,
            "invert": invert,
        }
    }
    
    return preview_image, info


# =============================================================================
# MAIN - EXAMPLE USAGE
# =============================================================================

if __name__ == "__main__":
    import tifffile
    
    print("="*60)
    print("LOCOTRACK-BASED OPTICAL FLOW GENERATION")
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
        
        # Initialize LocoTrack model
        print("Loading LocoTrack model...")
        manager = LocoTrackModelManager(model_size="base")
        
        # Compute LocoTrack-based optical flow
        flows, info = compute_locotrack_optical_flow(
            vol,
            manager,
            sigma_low=1.0,
            sigma_high=2.0,
            dog_threshold=0.01,
            min_distance=10,
            max_keypoints=300,
            max_occlusion_ratio=0.3,
            flow_smoothing=15.0,
            gaussian_sigma=2.0,
            verbose=True
        )
        
        print("\n" + "="*60)
        print("RESULTS")
        print("="*60)
        print(f"Flow shape: {flows.shape}")
        print(f"Total seeds: {info['total_seeds']}")
        print(f"Valid trajectories: {info['valid_trajectories']}")
        print(f"Filtered trajectories: {info['filtered_trajectories']}")
        
        # Save flows
        np.save("./data/locotrack_flows.npy", flows)
        print(f"Saved to: ./data/locotrack_flows.npy")
        
    except FileNotFoundError:
        print(f"Example video not found at {TIF_PATH}")
        print("This is expected - modify the path to point to your video.")
    except Exception as e:
        print(f"Error: {e}")
        raise
