#!/usr/bin/env python3
"""
Smart Frame Selection for Optimal Annotation.

This module intelligently selects which frames to annotate for maximum
fine-tuning impact. Quality over quantity - annotate 10 frames that matter,
not 100 random ones.

Key Features:
1. Diversity selection - cover different visual scenarios
2. Difficulty detection - focus on where model struggles (uncertainty)
3. Motion analysis - high-motion frames are often challenging
4. Temporal spread - ensure even coverage across video
5. One-click recommendations based on video analysis

Usage:
    from smart_sampling import SmartSampler, analyze_video_for_annotation
    
    # Get optimal frames for annotation
    sampler = SmartSampler(video)
    frames = sampler.get_optimal_frames(n_frames=10)
    
    # Or use the quick analysis function
    recommendation = analyze_video_for_annotation(video_path)
    print(f"Annotate frames: {recommendation['frames']}")
    print(f"Estimated time: {recommendation['estimated_time_minutes']} min")

Author: RIPPLE Team
Date: 2026-01-30
"""

import numpy as np
from typing import List, Dict, Tuple, Optional, Union
from pathlib import Path
import warnings

try:
    import torch
    import torch.nn.functional as F
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

try:
    from sklearn.cluster import KMeans
    from sklearn.decomposition import PCA
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False


class SmartSampler:
    """
    Intelligently select optimal frames for annotation.
    
    Combines multiple strategies:
    1. Visual diversity (cluster-based selection)
    2. Model uncertainty (Monte Carlo dropout)
    3. Motion magnitude (frame differences)
    4. Temporal coverage (even spread)
    
    Args:
        video: Video tensor (T, H, W) or (T, H, W, C)
        model: Optional LocoTrack model for uncertainty estimation
        device: Device for computation
    """
    
    def __init__(self, video: np.ndarray, model=None, device: str = None):
        self.video = video
        self.model = model
        self.device = device or ('cuda' if HAS_TORCH and torch.cuda.is_available() else 'cpu')
        
        # Video dimensions
        if video.ndim == 3:
            self.T, self.H, self.W = video.shape
            self.C = 1
        else:
            self.T, self.H, self.W, self.C = video.shape
        
        # Cache computed features
        self._frame_features = None
        self._motion_scores = None
        self._complexity_scores = None
    
    def get_diverse_frames(self, n_frames: int = 10) -> List[int]:
        """
        Select visually diverse frames using feature clustering.
        
        Uses PCA-reduced features and K-means clustering to find
        frames that represent different visual states in the video.
        
        Args:
            n_frames: Number of frames to select
            
        Returns:
            List of frame indices
        """
        if not HAS_SKLEARN:
            warnings.warn("sklearn not available, using uniform sampling")
            return self._uniform_sample(n_frames)
        
        # Extract features
        features = self._get_frame_features()
        
        # Reduce dimensionality if needed
        if features.shape[1] > 50:
            pca = PCA(n_components=min(50, features.shape[0] - 1))
            features = pca.fit_transform(features)
        
        # Cluster frames
        n_clusters = min(n_frames, self.T)
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        cluster_labels = kmeans.fit_predict(features)
        
        # Select frame closest to each cluster center
        selected = []
        for i in range(n_clusters):
            cluster_frames = np.where(cluster_labels == i)[0]
            if len(cluster_frames) == 0:
                continue
            
            distances = np.linalg.norm(
                features[cluster_frames] - kmeans.cluster_centers_[i],
                axis=1
            )
            best_idx = cluster_frames[np.argmin(distances)]
            selected.append(int(best_idx))
        
        return sorted(selected)
    
    def get_high_motion_frames(self, n_frames: int = 10) -> List[int]:
        """
        Select frames with significant motion.
        
        High-motion frames are often challenging for tracking and
        provide valuable training signal.
        
        Args:
            n_frames: Number of frames to select
            
        Returns:
            List of frame indices
        """
        motion_scores = self._get_motion_scores()
        
        # Get frames with highest motion
        sorted_indices = np.argsort(motion_scores)[::-1]
        
        # Ensure temporal spread
        selected = []
        min_gap = max(1, self.T // (n_frames * 3))
        
        for idx in sorted_indices:
            if len(selected) >= n_frames:
                break
            
            # Check distance from already selected
            if all(abs(idx - s) >= min_gap for s in selected):
                selected.append(int(idx))
        
        # Fill remaining if needed
        for idx in sorted_indices:
            if len(selected) >= n_frames:
                break
            if idx not in selected:
                selected.append(int(idx))
        
        return sorted(selected[:n_frames])
    
    def get_high_complexity_frames(self, n_frames: int = 10) -> List[int]:
        """
        Select frames with high visual complexity.
        
        Complex frames (many features, edges, textures) often contain
        more objects and are more representative for training.
        
        Args:
            n_frames: Number of frames to select
            
        Returns:
            List of frame indices
        """
        complexity_scores = self._get_complexity_scores()
        
        # Get frames with highest complexity
        sorted_indices = np.argsort(complexity_scores)[::-1]
        
        # Ensure temporal spread
        selected = []
        min_gap = max(1, self.T // (n_frames * 3))
        
        for idx in sorted_indices:
            if len(selected) >= n_frames:
                break
            if all(abs(idx - s) >= min_gap for s in selected):
                selected.append(int(idx))
        
        return sorted(selected[:n_frames])
    
    def get_uncertain_frames(self, n_frames: int = 10, 
                             sample_points: int = 20,
                             n_forward_passes: int = 5) -> List[int]:
        """
        Select frames where the model is most uncertain.
        
        Uses Monte Carlo dropout to estimate prediction uncertainty.
        These frames are the most valuable for training!
        
        Requires a model to be provided during initialization.
        
        Args:
            n_frames: Number of frames to select
            sample_points: Number of query points to test per frame
            n_forward_passes: Number of stochastic forward passes
            
        Returns:
            List of frame indices
        """
        if self.model is None or not HAS_TORCH:
            warnings.warn("Model not available, falling back to motion-based selection")
            return self.get_high_motion_frames(n_frames)
        
        uncertainties = []
        
        # Generate random query points
        query_points = np.random.rand(sample_points, 2) * np.array([self.W, self.H])
        
        for t in range(self.T):
            # Get a window of frames around this frame
            start_t = max(0, t - 4)
            end_t = min(self.T, t + 4)
            
            video_window = self.video[start_t:end_t]
            if video_window.ndim == 3:
                video_window = video_window[..., np.newaxis]
            
            # Convert to tensor
            video_tensor = torch.from_numpy(video_window).float()
            video_tensor = video_tensor.permute(0, 3, 1, 2).unsqueeze(0)
            video_tensor = video_tensor.to(self.device)
            
            # Query points for this frame (relative to window)
            frame_in_window = t - start_t
            qp = np.column_stack([
                np.full(sample_points, frame_in_window),
                query_points[:, 1],  # y
                query_points[:, 0],  # x
            ])
            qp_tensor = torch.from_numpy(qp.astype(np.float32)).unsqueeze(0).to(self.device)
            
            # Monte Carlo dropout - multiple forward passes
            predictions = []
            self.model.train()  # Enable dropout
            
            for _ in range(n_forward_passes):
                try:
                    with torch.no_grad():
                        output = self.model(video_tensor, qp_tensor)
                        pred_tracks = output['tracks'].cpu().numpy()
                        predictions.append(pred_tracks)
                except Exception:
                    break
            
            self.model.eval()
            
            if len(predictions) < 2:
                uncertainties.append(0.0)
                continue
            
            # Uncertainty = variance across predictions
            pred_stack = np.stack(predictions)
            variance = np.var(pred_stack, axis=0).mean()
            uncertainties.append(variance)
        
        # Select frames with highest uncertainty
        sorted_indices = np.argsort(uncertainties)[::-1]
        return sorted(sorted_indices[:n_frames].tolist())
    
    def get_optimal_frames(self, n_frames: int = 10, 
                           weights: Dict[str, float] = None) -> List[int]:
        """
        Combine all strategies for optimal frame selection.
        
        Scores each frame based on weighted combination of:
        - Diversity (different visual appearance)
        - Motion (challenging tracking scenarios)
        - Complexity (rich visual content)
        - Temporal spread (even coverage)
        
        Args:
            n_frames: Number of frames to select
            weights: Optional dict of strategy weights
                     {'diversity': 1.0, 'motion': 1.5, 'complexity': 1.0}
                     
        Returns:
            List of optimal frame indices
        """
        weights = weights or {
            'diversity': 1.0,
            'motion': 1.5,
            'complexity': 1.0,
        }
        
        # Get candidates from each strategy
        diverse_frames = set(self.get_diverse_frames(n_frames * 2))
        motion_frames = set(self.get_high_motion_frames(n_frames * 2))
        complex_frames = set(self.get_high_complexity_frames(n_frames * 2))
        
        # Score each frame
        frame_scores = {}
        for t in range(self.T):
            score = 0.0
            if t in diverse_frames:
                score += weights.get('diversity', 1.0)
            if t in motion_frames:
                score += weights.get('motion', 1.5)
            if t in complex_frames:
                score += weights.get('complexity', 1.0)
            frame_scores[t] = score
        
        # Sort by score
        sorted_frames = sorted(frame_scores.keys(), key=lambda x: -frame_scores[x])
        
        # Select with temporal spread constraint
        min_gap = max(1, self.T // (n_frames * 2))
        selected = []
        
        for frame in sorted_frames:
            if len(selected) >= n_frames:
                break
            if all(abs(frame - s) >= min_gap for s in selected):
                selected.append(frame)
        
        # Fill remaining if needed (relax gap constraint)
        for frame in sorted_frames:
            if len(selected) >= n_frames:
                break
            if frame not in selected:
                selected.append(frame)
        
        return sorted(selected[:n_frames])
    
    def _get_frame_features(self) -> np.ndarray:
        """Extract visual features from each frame."""
        if self._frame_features is not None:
            return self._frame_features
        
        features = []
        for t in range(self.T):
            frame = self.video[t]
            if frame.ndim == 3:
                frame = frame.mean(axis=-1)  # Convert to grayscale
            
            frame = frame.astype(np.float32)
            
            # Histogram features (global intensity distribution)
            hist, _ = np.histogram(frame.flatten(), bins=32, density=True)
            
            # Spatial features (intensity in grid cells)
            h, w = frame.shape
            grid_h, grid_w = 4, 4
            cell_h, cell_w = h // grid_h, w // grid_w
            
            spatial = []
            for i in range(grid_h):
                for j in range(grid_w):
                    cell = frame[i*cell_h:(i+1)*cell_h, j*cell_w:(j+1)*cell_w]
                    spatial.extend([cell.mean(), cell.std()])
            
            # Edge features (gradient magnitude)
            if frame.shape[0] > 2 and frame.shape[1] > 2:
                gy = np.abs(frame[1:, :] - frame[:-1, :]).mean()
                gx = np.abs(frame[:, 1:] - frame[:, :-1]).mean()
                edge_features = [gx, gy, np.sqrt(gx**2 + gy**2)]
            else:
                edge_features = [0, 0, 0]
            
            feature = np.concatenate([hist, spatial, edge_features])
            features.append(feature)
        
        self._frame_features = np.stack(features)
        return self._frame_features
    
    def _get_motion_scores(self) -> np.ndarray:
        """Compute motion magnitude for each frame."""
        if self._motion_scores is not None:
            return self._motion_scores
        
        motion_scores = [0.0]  # First frame has no motion
        
        for t in range(1, self.T):
            curr = self.video[t].astype(np.float32)
            prev = self.video[t-1].astype(np.float32)
            
            if curr.ndim == 3:
                curr = curr.mean(axis=-1)
                prev = prev.mean(axis=-1)
            
            diff = np.abs(curr - prev)
            motion_scores.append(diff.mean())
        
        self._motion_scores = np.array(motion_scores)
        return self._motion_scores
    
    def _get_complexity_scores(self) -> np.ndarray:
        """Compute visual complexity for each frame."""
        if self._complexity_scores is not None:
            return self._complexity_scores
        
        complexity_scores = []
        
        for t in range(self.T):
            frame = self.video[t].astype(np.float32)
            if frame.ndim == 3:
                frame = frame.mean(axis=-1)
            
            # Gradient magnitude (edge content)
            if frame.shape[0] > 2 and frame.shape[1] > 2:
                gy = frame[1:, :] - frame[:-1, :]
                gx = frame[:, 1:] - frame[:, :-1]
                gradient = np.sqrt(gy[:, :-1]**2 + gx[:-1, :]**2)
                edge_score = gradient.mean()
            else:
                edge_score = 0
            
            # Local variance (texture content)
            h, w = frame.shape
            grid_h, grid_w = 8, 8
            cell_h, cell_w = h // grid_h, w // grid_w
            
            variances = []
            for i in range(grid_h):
                for j in range(grid_w):
                    cell = frame[i*cell_h:(i+1)*cell_h, j*cell_w:(j+1)*cell_w]
                    variances.append(cell.std())
            
            texture_score = np.mean(variances)
            
            # Combine
            complexity = edge_score + texture_score
            complexity_scores.append(complexity)
        
        self._complexity_scores = np.array(complexity_scores)
        return self._complexity_scores
    
    def _uniform_sample(self, n_frames: int) -> List[int]:
        """Fallback: uniform temporal sampling."""
        indices = np.linspace(0, self.T - 1, n_frames, dtype=int)
        return sorted(indices.tolist())


def analyze_video_for_annotation(video_or_path: Union[np.ndarray, str, Path],
                                  target_accuracy: float = 0.95,
                                  model=None) -> Dict:
    """
    Analyze a video and provide annotation recommendations.
    
    Returns optimal frames to annotate, estimated time, and complexity assessment.
    
    Args:
        video_or_path: Video array or path to video file
        target_accuracy: Target tracking accuracy (0-1)
        model: Optional model for uncertainty estimation
        
    Returns:
        Dict with recommendations:
        - frames: List of frame indices to annotate
        - num_frames: Recommended number of frames
        - points_per_frame: Recommended annotations per frame
        - estimated_time_minutes: Time estimate for annotation
        - complexity: Video complexity score
        - message: Human-readable recommendation
    """
    # Load video if path provided
    if isinstance(video_or_path, (str, Path)):
        video = _load_video(video_or_path)
    else:
        video = video_or_path
    
    # Create sampler
    sampler = SmartSampler(video, model=model)
    
    # Analyze complexity
    motion_scores = sampler._get_motion_scores()
    complexity_scores = sampler._get_complexity_scores()
    
    avg_motion = np.mean(motion_scores)
    max_motion = np.max(motion_scores)
    avg_complexity = np.mean(complexity_scores)
    
    # Compute overall complexity score (0-1)
    motion_norm = min(1.0, avg_motion / 50)  # Normalize
    complexity_norm = min(1.0, avg_complexity / 100)
    overall_complexity = (motion_norm + complexity_norm) / 2
    
    # Determine recommendations based on complexity
    if overall_complexity < 0.2:
        n_frames = 5
        points_per_frame = 5
        message = "🟢 Simple video - minimal annotation needed!"
        category = "simple"
    elif overall_complexity < 0.4:
        n_frames = 8
        points_per_frame = 8
        message = "🟡 Moderate complexity - standard annotation recommended."
        category = "moderate"
    elif overall_complexity < 0.6:
        n_frames = 12
        points_per_frame = 10
        message = "🟠 Complex video - more thorough annotation will help."
        category = "complex"
    else:
        n_frames = 20
        points_per_frame = 15
        message = "🔴 Challenging video - comprehensive annotation recommended."
        category = "challenging"
    
    # Adjust for target accuracy
    if target_accuracy > 0.98:
        n_frames = int(n_frames * 1.5)
        points_per_frame = int(points_per_frame * 1.2)
    
    # Cap at reasonable limits
    n_frames = min(n_frames, sampler.T // 5)  # Max 20% of video
    n_frames = max(n_frames, 3)  # Minimum 3 frames
    
    # Get optimal frames
    optimal_frames = sampler.get_optimal_frames(n_frames)
    
    # Estimate annotation time (~30 seconds per point annotation)
    total_annotations = n_frames * points_per_frame
    time_minutes = total_annotations * 0.5 / 60  # 30 sec per point
    
    return {
        'frames': optimal_frames,
        'num_frames': n_frames,
        'points_per_frame': points_per_frame,
        'total_annotations': total_annotations,
        'estimated_time_minutes': round(time_minutes, 1),
        'complexity': round(overall_complexity, 3),
        'complexity_category': category,
        'message': message,
        'video_info': {
            'num_frames': sampler.T,
            'height': sampler.H,
            'width': sampler.W,
            'avg_motion': round(avg_motion, 2),
            'max_motion': round(max_motion, 2),
            'avg_complexity': round(avg_complexity, 2),
        }
    }


def _load_video(path: Union[str, Path]) -> np.ndarray:
    """Load video from file."""
    path = Path(path)
    
    if not path.exists():
        raise FileNotFoundError(f"Video not found: {path}")
    
    suffix = path.suffix.lower()
    
    if suffix in ['.tif', '.tiff']:
        try:
            import tifffile
            video = tifffile.imread(str(path))
        except ImportError:
            raise ImportError("tifffile required for TIFF videos: pip install tifffile")
    elif suffix in ['.npy', '.npz']:
        data = np.load(str(path))
        if isinstance(data, np.lib.npyio.NpzFile):
            video = data[list(data.keys())[0]]
        else:
            video = data
    else:
        try:
            import cv2
            cap = cv2.VideoCapture(str(path))
            frames = []
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                frames.append(frame)
            cap.release()
            video = np.stack(frames)
        except ImportError:
            raise ImportError(f"cv2 required for {suffix} videos: pip install opencv-python")
    
    return video


# =============================================================================
# QUICK ANALYSIS PROFILES
# =============================================================================

ADAPTATION_PROFILES = {
    "fluorescent_particles": {
        "description": "Bright spots on dark background (e.g., beads, nuclei)",
        "learning_rate": 5e-5,
        "epochs": 5,
        "lora_rank": 4,
        "recommended_samples": 10,
        "estimated_time_minutes": 3,
        "keywords": ["fluorescent", "bright", "spots", "particles", "beads"]
    },
    
    "bright_field_cells": {
        "description": "Cells in bright-field or phase contrast microscopy",
        "learning_rate": 1e-4,
        "epochs": 5,
        "lora_rank": 8,
        "recommended_samples": 15,
        "estimated_time_minutes": 5,
        "keywords": ["cells", "bright-field", "phase", "contrast"]
    },
    
    "dense_tracking": {
        "description": "Many objects close together, frequent occlusions",
        "learning_rate": 2e-4,
        "epochs": 8,
        "lora_rank": 8,
        "recommended_samples": 20,
        "estimated_time_minutes": 7,
        "keywords": ["dense", "crowded", "occlusion", "many"]
    },
    
    "low_contrast": {
        "description": "Hard to see objects, subtle features",
        "learning_rate": 1e-4,
        "epochs": 10,
        "lora_rank": 8,
        "recommended_samples": 25,
        "estimated_time_minutes": 10,
        "keywords": ["low", "contrast", "dim", "faint", "subtle"]
    },
    
    "fast_motion": {
        "description": "Rapidly moving objects, motion blur",
        "learning_rate": 3e-4,
        "epochs": 5,
        "lora_rank": 4,
        "recommended_samples": 15,
        "estimated_time_minutes": 5,
        "keywords": ["fast", "motion", "blur", "rapid", "quick"]
    }
}


def suggest_profile(video: np.ndarray) -> Tuple[str, Dict]:
    """
    Suggest an adaptation profile based on video characteristics.
    
    Args:
        video: Video array
        
    Returns:
        (profile_name, profile_dict)
    """
    # Analyze video
    sampler = SmartSampler(video)
    motion_scores = sampler._get_motion_scores()
    complexity_scores = sampler._get_complexity_scores()
    
    # Simple heuristics
    avg_motion = np.mean(motion_scores)
    avg_intensity = np.mean(video)
    intensity_std = np.std(video)
    
    # Detect characteristics
    is_dark_background = avg_intensity < np.max(video) * 0.3
    is_high_contrast = intensity_std > np.mean(video) * 0.5
    is_fast_motion = avg_motion > 20
    is_dense = complexity_scores.mean() > 50
    is_low_contrast = intensity_std < np.mean(video) * 0.2
    
    # Match to profile
    if is_dark_background and is_high_contrast:
        return "fluorescent_particles", ADAPTATION_PROFILES["fluorescent_particles"]
    elif is_fast_motion:
        return "fast_motion", ADAPTATION_PROFILES["fast_motion"]
    elif is_dense:
        return "dense_tracking", ADAPTATION_PROFILES["dense_tracking"]
    elif is_low_contrast:
        return "low_contrast", ADAPTATION_PROFILES["low_contrast"]
    else:
        return "bright_field_cells", ADAPTATION_PROFILES["bright_field_cells"]


# =============================================================================
# COMMAND LINE INTERFACE
# =============================================================================

def main():
    """CLI for video analysis."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Analyze video and recommend annotation strategy"
    )
    parser.add_argument("video_path", help="Path to video file")
    parser.add_argument("--accuracy", type=float, default=0.95,
                        help="Target accuracy (0-1)")
    parser.add_argument("--json", action="store_true",
                        help="Output as JSON")
    
    args = parser.parse_args()
    
    print(f"\n🔍 Analyzing video: {args.video_path}\n")
    
    try:
        result = analyze_video_for_annotation(
            args.video_path,
            target_accuracy=args.accuracy
        )
        
        if args.json:
            import json
            print(json.dumps(result, indent=2))
        else:
            print(f"📊 Video Analysis Results")
            print("=" * 50)
            print(f"Video: {result['video_info']['num_frames']} frames, "
                  f"{result['video_info']['width']}x{result['video_info']['height']}")
            print(f"Complexity: {result['complexity']:.2f} ({result['complexity_category']})")
            print()
            print(f"{result['message']}")
            print()
            print(f"📝 Recommendation:")
            print(f"   • Annotate {result['num_frames']} frames")
            print(f"   • {result['points_per_frame']} points per frame")
            print(f"   • Total annotations: {result['total_annotations']}")
            print(f"   • Estimated time: {result['estimated_time_minutes']} minutes")
            print()
            print(f"🎯 Optimal frames to annotate:")
            print(f"   {result['frames']}")
            
    except Exception as e:
        print(f"❌ Error: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
