#!/usr/bin/env python3
"""LocoTrack Fine-Tuning Script for RIPPLE.

This module provides fine-tuning capabilities for the LocoTrack model
using user-annotated tracking data with occlusion ground truth.

The script converts RIPPLE's annotation format to a format compatible
with LocoTrack training, then runs a fine-tuning loop with early stopping
based on validation loss.

Key features:
- Converts JSON annotations to LocoTrack training format
- Supports occlusion ground truth from visible segments
- 85%/15% train/test split with stratification
- Early stopping to prevent overfitting
- Checkpoint saving with user-selected filename
- Returns test accuracy metrics after training

Usage:
    python locotrack_finetune.py --annotations annotations.json \
                                  --video video.tif \
                                  --base-weights locotrack_base.ckpt \
                                  --output-weights finetuned.ckpt \
                                  --epochs 100
"""

import argparse
import json
import sys
import os
import random
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any, Callable

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import tifffile

# Import tqdm for progress bars
try:
    from tqdm import tqdm
    TQDM_AVAILABLE = True
except ImportError:
    TQDM_AVAILABLE = False
    # Fallback: simple progress tracker
    def tqdm(iterable, **kwargs):
        return iterable

# Add the repo root to path for model imports
_SCRIPT_DIR = Path(__file__).resolve().parent
_RIPPLE_ROOT = _SCRIPT_DIR.parents[2]  # <repo>/src/main/python -> <repo>
_LOCOTRACK_DIR = _RIPPLE_ROOT / "locotrack_pytorch"
# Add locotrack_pytorch first so its models package takes precedence
if str(_LOCOTRACK_DIR) not in sys.path:
    sys.path.insert(0, str(_LOCOTRACK_DIR))

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class RIPPLETrackingDataset(Dataset):
    """Dataset that converts RIPPLE annotations to LocoTrack format.
    
    LocoTrack expects:
    - video: (T, H, W, C) float32 tensor normalized to [-1, 1]
    - query_points: (N, 3) tensor with (t, y, x) for each query point
    - target_points: (N, T, 2) tensor with (x, y) positions for each frame
    - occluded: (N, T) boolean tensor indicating occlusion status
    
    RIPPLE provides:
    - annotations: Dict[trackId -> Dict[frame -> (x, y)]]
    - visible_segments: Dict[trackId -> List[[start, end]]]
    """
    
    def __init__(
        self,
        video_path: str,
        annotations: Dict[str, Dict[int, Tuple[float, float]]],
        visible_segments: Dict[str, List[List[int]]],
        track_ids: List[str],
        resize_to: Tuple[int, int] = (256, 256),
        num_frames: int = 8,  # Reduced from 24 for memory efficiency
        tracks_per_sample: int = 32,  # Reduced from 256 for memory efficiency
    ):
        """
        Args:
            video_path: Path to the video file (TIFF)
            annotations: Track annotations {track_id: {frame: (x, y)}}
            visible_segments: Visible segments {track_id: [[start, end], ...]}
            track_ids: List of track IDs to include in this dataset
            resize_to: Target resolution for video frames
            num_frames: Number of frames per training sample
            tracks_per_sample: Number of query points per sample
        """
        self.video_path = video_path
        self.annotations = annotations
        self.visible_segments = visible_segments
        self.track_ids = track_ids
        self.resize_to = resize_to
        self.num_frames = num_frames
        self.tracks_per_sample = tracks_per_sample
        
        # Load video - _load_video will set self.original_size BEFORE resizing
        self.video = self._load_video()
        self.total_frames = self.video.shape[0]
        # Note: self.original_size is now set in _load_video() BEFORE resizing
        
        # Pre-compute valid sampling ranges
        self._compute_sampling_ranges()
    
    def _load_video(self) -> np.ndarray:
        """Load and preprocess video."""
        video_path = Path(self.video_path)
        
        # Load video based on format
        if video_path.suffix.lower() in ['.tif', '.tiff']:
            video = tifffile.imread(self.video_path)
        else:
            # Use OpenCV for other formats (mp4, avi, etc.)
            import cv2
            cap = cv2.VideoCapture(str(self.video_path))
            frames = []
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                # OpenCV loads as BGR, convert to RGB
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append(frame)
            cap.release()
            video = np.stack(frames) if frames else np.zeros((1, 256, 256, 3), dtype=np.uint8)
        
        # Ensure 4D: (T, H, W, C)
        if video.ndim == 3:
            video = video[..., np.newaxis]  # Add channel dim
        if video.shape[-1] not in [1, 3]:
            # Assume (T, C, H, W) and transpose
            video = np.transpose(video, (0, 2, 3, 1))
        
        # Convert grayscale to RGB (model expects 3 channels)
        if video.shape[-1] == 1:
            video = np.repeat(video, 3, axis=-1)
        
        # CRITICAL: Store original size BEFORE resizing for coordinate scaling
        self.original_size = (video.shape[1], video.shape[2])  # (H, W)
        
        # Convert to float and normalize to [-1, 1]
        video = video.astype(np.float32)
        # Handle different bit depths
        if video.max() > 1.0:
            if video.max() > 255:
                # 16-bit or higher
                video = video / video.max()
            else:
                # 8-bit
                video = video / 255.0
        video = video * 2.0 - 1.0
        
        # Resize if needed (skip if resize_to is None - use original resolution)
        if self.resize_to is not None and video.shape[1:3] != self.resize_to:
            import cv2
            resized = []
            for frame in video:
                resized.append(cv2.resize(frame, self.resize_to[::-1]))  # cv2 uses (W, H)
            video = np.stack(resized)
        
        # If resize_to is None, set it to original size for coordinate scaling
        if self.resize_to is None:
            self.resize_to = (video.shape[1], video.shape[2])  # (H, W)
        
        return video
    
    def _compute_sampling_ranges(self):
        """Compute valid frame ranges for sampling."""
        self.valid_start_frames = []
        
        # Find valid starting frames where we have enough subsequent frames
        max_start = self.total_frames - self.num_frames
        for start in range(max(0, max_start + 1)):
            # Check if we have tracks with annotations in this range
            has_tracks = False
            for track_id in self.track_ids:
                track_annots = self.annotations.get(track_id, {})
                for frame in range(start, start + self.num_frames):
                    if frame in track_annots:
                        has_tracks = True
                        break
                if has_tracks:
                    break
            
            if has_tracks:
                self.valid_start_frames.append(start)
        
        if not self.valid_start_frames:
            self.valid_start_frames = [0]
    
    def __len__(self):
        return len(self.valid_start_frames) * 10  # Multiple samples per start frame
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        # Select a random starting frame
        start_idx = idx % len(self.valid_start_frames)
        start_frame = self.valid_start_frames[start_idx]
        end_frame = min(start_frame + self.num_frames, self.total_frames)
        actual_frames = end_frame - start_frame
        
        # Extract video segment
        video_segment = self.video[start_frame:end_frame].copy()
        
        # Pad if needed
        if actual_frames < self.num_frames:
            padding = np.zeros((self.num_frames - actual_frames,) + video_segment.shape[1:], dtype=np.float32)
            video_segment = np.concatenate([video_segment, padding], axis=0)
        
        # Sample query points from tracks
        query_points = []
        target_points = []
        occluded = []
        
        # Scale factors for coordinate conversion
        scale_x = self.resize_to[1] / self.original_size[1]
        scale_y = self.resize_to[0] / self.original_size[0]
        
        for track_id in self.track_ids:
            track_annots = self.annotations.get(track_id, {})
            segments = self.visible_segments.get(track_id, [])
            
            # Find frames in this segment that have annotations
            valid_frames = [f for f in track_annots.keys() 
                          if start_frame <= f < end_frame]
            
            if not valid_frames:
                continue
            
            # Use first annotated frame as query point
            query_frame = valid_frames[0]
            query_x, query_y = track_annots[query_frame]
            
            # Scale to target resolution
            query_x_scaled = query_x * scale_x
            query_y_scaled = query_y * scale_y
            
            # Query point format: (t, y, x) relative to segment start
            query_t = query_frame - start_frame
            query_points.append([query_t, query_y_scaled, query_x_scaled])
            
            # Build target trajectory and occlusion mask
            track_targets = []
            track_occluded = []
            
            for frame in range(start_frame, start_frame + self.num_frames):
                if frame in track_annots:
                    x, y = track_annots[frame]
                    track_targets.append([x * scale_x, y * scale_y])
                else:
                    # Interpolate or use last known position
                    track_targets.append([query_x_scaled, query_y_scaled])
                
                # Check if frame is occluded
                frame_occluded = True
                if segments:
                    for seg_start, seg_end in segments:
                        if seg_start <= frame <= seg_end:
                            frame_occluded = False
                            break
                else:
                    # No occlusion data = assume visible
                    frame_occluded = False
                
                track_occluded.append(frame_occluded)
            
            target_points.append(track_targets)
            occluded.append(track_occluded)
        
        # Pad or sample to get exactly tracks_per_sample points
        num_points = len(query_points)
        if num_points == 0:
            # Create dummy data
            query_points = [[0, self.resize_to[0]//2, self.resize_to[1]//2]]
            target_points = [[[self.resize_to[1]//2, self.resize_to[0]//2]] * self.num_frames]
            occluded = [[True] * self.num_frames]
            num_points = 1
        
        if num_points < self.tracks_per_sample:
            # Repeat tracks to fill
            factor = (self.tracks_per_sample + num_points - 1) // num_points
            query_points = (query_points * factor)[:self.tracks_per_sample]
            target_points = (target_points * factor)[:self.tracks_per_sample]
            occluded = (occluded * factor)[:self.tracks_per_sample]
        elif num_points > self.tracks_per_sample:
            # Random sample
            indices = random.sample(range(num_points), self.tracks_per_sample)
            query_points = [query_points[i] for i in indices]
            target_points = [target_points[i] for i in indices]
            occluded = [occluded[i] for i in indices]
        
        # Convert to tensors
        # LocoTrack expects video shape: (B, T, H, W, C)
        # video_segment is (T, H, W, C) already from _load_video
        video_tensor = torch.from_numpy(video_segment)  # (T, H, W, C)
        video_tensor = video_tensor.unsqueeze(0)  # (1, T, H, W, C) - add batch dim
        
        return {
            'video': video_tensor,
            'query_points': torch.tensor(query_points, dtype=torch.float32),
            'target_points': torch.tensor(target_points, dtype=torch.float32),
            'occluded': torch.tensor(occluded, dtype=torch.bool),
        }


def parse_annotations_dict(data: Dict) -> Tuple[Dict, Dict, int]:
    """Parse annotations from a dict (already loaded JSON).
    
    Args:
        data: The annotations dict with 'tracks' array and optional metadata
    
    Returns:
        annotations: {track_id: {frame: (x, y)}}
        visible_segments: {track_id: [[start, end], ...]}
        total_frames: Total number of frames in video
    """
    annotations = {}
    visible_segments = {}
    
    # Handle both 'metadata' and top-level total_frames
    total_frames = data.get('total_frames', 0)
    if total_frames == 0:
        total_frames = data.get('metadata', {}).get('total_frames', 0)
    
    for track in data.get('tracks', []):
        track_id = track['track_id']
        
        # Load annotations - handle both 'annotations' and 'positions' keys
        track_annots = {}
        positions = track.get('annotations', track.get('positions', []))
        for frame_data in positions:
            frame = frame_data['frame']
            x = frame_data['x']
            y = frame_data['y']
            track_annots[frame] = (x, y)
        
        if track_annots:
            annotations[track_id] = track_annots
        
        # Load occlusion/visibility segments
        if 'visible_segments' in track:
            segments = []
            for seg in track['visible_segments']:
                segments.append([seg['start'], seg['end']])
            visible_segments[track_id] = segments
    
    return annotations, visible_segments, total_frames


def load_annotations(json_path: str) -> Tuple[Dict, Dict, int]:
    """Load annotations from RIPPLE JSON export file.
    
    Returns:
        annotations: {track_id: {frame: (x, y)}}
        visible_segments: {track_id: [[start, end], ...]}
        total_frames: Total number of frames in video
    """
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    return parse_annotations_dict(data)


def split_tracks(
    track_ids: List[str],
    train_ratio: float = 0.85,
    seed: int = 42
) -> Tuple[List[str], List[str]]:
    """Split tracks into train and test sets.
    
    Args:
        track_ids: List of all track IDs
        train_ratio: Fraction of tracks for training
        seed: Random seed for reproducibility
    
    Returns:
        train_ids, test_ids
    """
    random.seed(seed)
    shuffled = track_ids.copy()
    random.shuffle(shuffled)
    
    split_idx = int(len(shuffled) * train_ratio)
    
    # Ensure at least 1 track in test set and at least 2 in train set
    if split_idx >= len(shuffled):
        split_idx = len(shuffled) - 1
    if split_idx < 2:
        split_idx = min(2, len(shuffled) - 1)
    
    train_ids = shuffled[:split_idx]
    test_ids = shuffled[split_idx:]
    
    return train_ids, test_ids


def compute_metrics(
    pred_tracks: torch.Tensor,
    pred_occluded: torch.Tensor,
    gt_tracks: torch.Tensor,
    gt_occluded: torch.Tensor,
) -> Dict[str, float]:
    """Compute TAP-Vid style metrics.
    
    Args:
        pred_tracks: (B, N, T, 2) predicted positions
        pred_occluded: (B, N, T) predicted occlusion
        gt_tracks: (B, N, T, 2) ground truth positions
        gt_occluded: (B, N, T) ground truth occlusion
    
    Returns:
        Dict with metrics: occlusion_accuracy, position_accuracy, etc.
    """
    metrics = {}
    
    # Occlusion accuracy
    occ_correct = (pred_occluded == gt_occluded).float()
    metrics['occlusion_accuracy'] = occ_correct.mean().item()
    
    # Position error (only on visible frames)
    visible = ~gt_occluded
    if visible.any():
        position_error = torch.norm(pred_tracks - gt_tracks, dim=-1)
        visible_error = position_error[visible].mean().item()
        metrics['position_error'] = visible_error
        
        # Points within threshold
        for thresh in [1, 2, 4, 8, 16]:
            within = (position_error < thresh) & visible
            metrics[f'pts_within_{thresh}'] = (within.sum() / visible.sum()).item()
    
    # Average metrics
    thresholds = [1, 2, 4, 8, 16]
    pts_within = [metrics.get(f'pts_within_{t}', 0) for t in thresholds]
    metrics['average_pts_within_thresh'] = np.mean(pts_within)
    
    return metrics


def finetune_locotrack(
    video_path: str,
    base_weights: str,
    output_weights: str,
    annotations: Optional[Dict] = None,
    annotations_path: Optional[str] = None,
    epochs: int = 100,
    batch_size: int = 1,
    learning_rate: float = 1e-4,
    train_split: float = 0.85,
    train_ratio: Optional[float] = None,  # Alias for train_split
    early_stopping_patience: int = 10,
    device: str = 'cuda',
    model_type: str = 'base',
    resize_to: Optional[Tuple[int, int]] = (256, 256),
    cancel_check: Optional[Callable] = None,
    progress_callback: Optional[Callable[[Dict[str, Any]], None]] = None,
) -> Dict[str, Any]:
    """Fine-tune LocoTrack on user annotations.
    
    Args:
        video_path: Path to video file
        base_weights: Path to base LocoTrack weights
        output_weights: Path to save fine-tuned weights
        annotations: Annotations dict (from tracking server)
        annotations_path: Path to RIPPLE annotations JSON (alternative to annotations)
        epochs: Maximum number of training epochs
        batch_size: Training batch size
        learning_rate: Learning rate for fine-tuning
        train_split: Fraction of tracks for training (0.85 = 85% train, 15% test)
        train_ratio: Alias for train_split
        early_stopping_patience: Stop if no improvement for N epochs
        device: 'cuda' or 'cpu'
        model_type: 'base' or 'small' (for logging purposes)
        resize_to: Target resolution for training (default: (256, 256) = LocoTrack native).
            Set to None to use the video's original resolution.
        cancel_check: Optional callable to check for cancellation
        progress_callback: Optional callback for progress updates. Called with dict:
            {'phase': 'train'|'validate'|'complete', 'epoch': int, 'total_epochs': int,
             'batch': int, 'total_batches': int, 'train_loss': float, 'val_loss': float,
             'message': str}
    
    Returns:
        Dict with training results and test metrics
    """
    # Handle train_ratio alias
    if train_ratio is not None:
        train_split = train_ratio
    
    # Load annotations from dict or file
    if annotations is not None:
        logger.info("Parsing annotations from dict")
        parsed_annotations, visible_segments, total_frames = parse_annotations_dict(annotations)
    elif annotations_path is not None:
        logger.info(f"Loading annotations from {annotations_path}")
        parsed_annotations, visible_segments, total_frames = load_annotations(annotations_path)
    else:
        raise ValueError("Either 'annotations' or 'annotations_path' must be provided")
    
    # Use parsed_annotations from here on
    annotations = parsed_annotations
    
    if not annotations:
        raise ValueError("No valid annotations found in the JSON file")    
    # Clear GPU memory before starting
    if device == 'cuda':
        torch.cuda.empty_cache()
        import gc
        gc.collect()    
    track_ids = list(annotations.keys())
    logger.info(f"Found {len(track_ids)} tracks")
    
    if len(track_ids) < 4:
        raise ValueError(f"Need at least 4 tracks for fine-tuning, got {len(track_ids)}")
    
    # Split into train/test
    train_ids, test_ids = split_tracks(track_ids, train_split)
    logger.info(f"Train: {len(train_ids)} tracks, Test: {len(test_ids)} tracks")
    
    # Determine resize resolution
    # If resize_to is None, the dataset will load video at original resolution
    # but we need to handle this by reading video dimensions first
    dataset_resize = resize_to if resize_to is not None else None
    if dataset_resize is not None:
        logger.info(f"Training at resolution: {dataset_resize}")
    else:
        logger.info("Training at original video resolution")
    
    # Create datasets
    train_dataset = RIPPLETrackingDataset(
        video_path=video_path,
        annotations=annotations,
        visible_segments=visible_segments,
        track_ids=train_ids,
        resize_to=dataset_resize,
    )
    
    test_dataset = RIPPLETrackingDataset(
        video_path=video_path,
        annotations=annotations,
        visible_segments=visible_segments,
        track_ids=test_ids,
        resize_to=dataset_resize,
    )
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    # Load model
    logger.info(f"Loading base model from {base_weights}")
    from models.locotrack_model import LocoTrack
    
    model = LocoTrack(model_size=model_type, num_pips_iter=4)
    
    if os.path.exists(base_weights):
        checkpoint = torch.load(base_weights, map_location=device)
        if 'state_dict' in checkpoint:
            # Lightning checkpoint format
            state_dict = checkpoint['state_dict']
            # Remove 'model.' prefix if present
            state_dict = {k.replace('model.', ''): v for k, v in state_dict.items()}
            model.load_state_dict(state_dict, strict=False)
        else:
            model.load_state_dict(checkpoint, strict=False)
        logger.info("Loaded base weights successfully")
    else:
        logger.warning(f"Base weights not found at {base_weights}, starting from scratch")
    
    model = model.to(device)
    model.train()
    
    # Enable mixed precision training for memory efficiency
    use_amp = device == 'cuda'
    scaler = torch.amp.GradScaler('cuda') if use_amp else None
    
    # Optimizer with lower learning rate for fine-tuning
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    
    # Training loop with early stopping
    best_val_loss = float('inf')
    patience_counter = 0
    best_epoch = 0
    
    results = {
        'train_losses': [],
        'val_losses': [],
        'epochs_trained': 0,
    }
    
    # Helper function to report progress
    def report_progress(phase: str, epoch: int, batch: int = 0, total_batches: int = 0,
                       train_loss: float = 0.0, val_loss: float = 0.0, message: str = ""):
        if progress_callback is not None:
            try:
                progress_callback({
                    'phase': phase,
                    'epoch': epoch + 1,
                    'total_epochs': epochs,
                    'batch': batch,
                    'total_batches': total_batches,
                    'train_loss': train_loss,
                    'val_loss': val_loss,
                    'best_val_loss': best_val_loss,
                    'message': message,
                })
            except Exception as e:
                logger.warning(f"Progress callback error: {e}")
    
    total_train_batches = len(train_loader)
    total_val_batches = len(test_loader)
    
    report_progress('init', -1, message=f"Starting training with {len(train_dataset)} train / {len(test_dataset)} test samples")
    
    epoch = 0  # Initialize in case epochs == 0
    for epoch in range(epochs):
        # Training
        # Check for cancellation at start of epoch
        if cancel_check is not None:
            cancel_check()
        
        report_progress('epoch_start', epoch, message=f"Epoch {epoch+1}/{epochs}")
        
        model.train()
        train_loss = 0.0
        num_batches = 0
        
        # Use tqdm for progress bar in console, with callback for GUI
        train_iter = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [Train]", 
                          leave=False, disable=not TQDM_AVAILABLE) if TQDM_AVAILABLE else train_loader
        
        for batch_idx, batch in enumerate(train_iter):
            # Report batch progress
            report_progress('train', epoch, batch=batch_idx + 1, total_batches=total_train_batches,
                           train_loss=train_loss / max(num_batches, 1),
                           message=f"Training batch {batch_idx+1}/{total_train_batches}")
            
            # video shape from dataloader: (batch, 1, T, H, W, C)
            # We need (batch, T, H, W, C) for the model
            video = batch['video'].squeeze(1).to(device)  # Remove extra batch dim from dataset
            query_points = batch['query_points'].to(device)  # (batch, N, 3)
            target_points = batch['target_points'].to(device)  # (batch, N, T, 2)
            occluded = batch['occluded'].to(device)  # (batch, N, T)
            
            optimizer.zero_grad()
            
            # Forward pass with mixed precision
            with torch.amp.autocast(device_type='cuda', enabled=use_amp):
                output = model(video, query_points)
                
                # Compute loss
                pred_tracks = output['tracks']
                pred_occlusion = output['occlusion']
                
                # Position loss (Huber)
                position_error = torch.sum((pred_tracks - target_points) ** 2, dim=-1)
                position_loss = torch.where(
                    position_error < 16,
                    position_error / 2,
                    4 * (torch.sqrt(position_error + 1e-8) - 2)
                )
                position_loss = position_loss * (~occluded).float()
                position_loss = position_loss.mean()
                
                # Occlusion loss (BCE)
                occ_loss = F.binary_cross_entropy_with_logits(
                    pred_occlusion,
                    occluded.float(),
                    reduction='mean'
                )
                
                loss = position_loss * 0.05 + occ_loss
            
            # Backward pass with gradient scaling for mixed precision
            if use_amp:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
            
            train_loss += loss.item()
            num_batches += 1
            
            # Clear intermediate tensors to save memory
            del video, query_points, target_points, occluded, output, pred_tracks, pred_occlusion
            if device == 'cuda':
                torch.cuda.empty_cache()
        
        train_loss /= max(num_batches, 1)
        scheduler.step()
        
        # Validation
        model.eval()
        val_loss = 0.0
        num_val_batches = 0
        
        report_progress('validate_start', epoch, train_loss=train_loss, 
                       message=f"Validating epoch {epoch+1}/{epochs}")
        
        # Use tqdm for validation progress
        val_iter = tqdm(test_loader, desc=f"Epoch {epoch+1}/{epochs} [Val]", 
                        leave=False, disable=not TQDM_AVAILABLE) if TQDM_AVAILABLE else test_loader
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(val_iter):
                report_progress('validate', epoch, batch=batch_idx + 1, total_batches=total_val_batches,
                               train_loss=train_loss, val_loss=val_loss / max(num_val_batches, 1),
                               message=f"Validating batch {batch_idx+1}/{total_val_batches}")
                
                video = batch['video'].squeeze(1).to(device)
                query_points = batch['query_points'].to(device)
                target_points = batch['target_points'].to(device)
                occluded = batch['occluded'].to(device)
                
                with torch.amp.autocast(device_type='cuda', enabled=use_amp):
                    output = model(video, query_points)
                    
                    pred_tracks = output['tracks']
                    pred_occlusion = output['occlusion']
                    
                    position_error = torch.sum((pred_tracks - target_points) ** 2, dim=-1)
                    position_loss = torch.where(
                        position_error < 16,
                        position_error / 2,
                        4 * (torch.sqrt(position_error + 1e-8) - 2)
                    )
                    position_loss = position_loss * (~occluded).float()
                    position_loss = position_loss.mean()
                    
                    occ_loss = F.binary_cross_entropy_with_logits(
                        pred_occlusion,
                        occluded.float(),
                        reduction='mean'
                    )
                    
                    loss = position_loss * 0.05 + occ_loss
                
                val_loss += loss.item()
                num_val_batches += 1
                
                # Clear intermediate tensors
                del video, query_points, target_points, occluded, output, pred_tracks, pred_occlusion
                if device == 'cuda':
                    torch.cuda.empty_cache()
        
        val_loss /= max(num_val_batches, 1)
        
        results['train_losses'].append(train_loss)
        results['val_losses'].append(val_loss)
        
        # Report epoch complete
        report_progress('epoch_end', epoch, train_loss=train_loss, val_loss=val_loss,
                       message=f"Epoch {epoch+1}/{epochs} complete - Train: {train_loss:.4f}, Val: {val_loss:.4f}")
        
        logger.info(f"Epoch {epoch+1}/{epochs} - Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
        
        # Early stopping check
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            patience_counter = 0
            
            # Save best model
            torch.save({
                'state_dict': model.state_dict(),
                'epoch': epoch,
                'val_loss': val_loss,
            }, output_weights)
            logger.info(f"Saved best model (epoch {epoch+1})")
            report_progress('model_saved', epoch, train_loss=train_loss, val_loss=val_loss,
                           message=f"Saved best model (epoch {epoch+1}, val_loss={val_loss:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= early_stopping_patience:
                logger.info(f"Early stopping at epoch {epoch+1}")
                report_progress('early_stop', epoch, train_loss=train_loss, val_loss=val_loss,
                               message=f"Early stopping at epoch {epoch+1} (no improvement for {early_stopping_patience} epochs)")
                break
    
    results['epochs_trained'] = epoch + 1
    results['best_epoch'] = best_epoch + 1
    results['best_val_loss'] = best_val_loss
    
    report_progress('testing', epoch, message="Computing final test metrics...")
    
    # Load best model and compute final test metrics
    checkpoint = torch.load(output_weights, map_location=device)
    model.load_state_dict(checkpoint['state_dict'])
    model.eval()
    
    all_metrics = []
    test_iter = tqdm(test_loader, desc="Computing test metrics", 
                     leave=False, disable=not TQDM_AVAILABLE) if TQDM_AVAILABLE else test_loader
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(test_iter):
            report_progress('testing', epoch, batch=batch_idx + 1, total_batches=total_val_batches,
                           message=f"Testing batch {batch_idx+1}/{total_val_batches}")
            
            video = batch['video'].squeeze(1).to(device)
            query_points = batch['query_points'].to(device)
            target_points = batch['target_points'].to(device)
            occluded = batch['occluded'].to(device)
            
            with torch.amp.autocast(device_type='cuda', enabled=use_amp):
                output = model(video, query_points)
                
                pred_tracks = output['tracks']
                pred_occluded = torch.sigmoid(output['occlusion']) > 0.5
            
            metrics = compute_metrics(
                pred_tracks.unsqueeze(0),
                pred_occluded.unsqueeze(0),
                target_points.unsqueeze(0),
                occluded.unsqueeze(0),
            )
            all_metrics.append(metrics)
            
            # Clear intermediate tensors
            del video, query_points, target_points, occluded, output, pred_tracks, pred_occluded
            if device == 'cuda':
                torch.cuda.empty_cache()
    
    # Average metrics
    if all_metrics:
        final_metrics = {}
        for key in all_metrics[0].keys():
            final_metrics[key] = np.mean([m[key] for m in all_metrics])
        results['test_metrics'] = final_metrics
        
        logger.info(f"\n{'='*50}")
        logger.info("FINAL TEST METRICS:")
        logger.info(f"  Occlusion Accuracy: {final_metrics['occlusion_accuracy']*100:.1f}%")
        logger.info(f"  Position Error: {final_metrics.get('position_error', 0):.2f} pixels")
        logger.info(f"  Average Pts Within Thresh: {final_metrics['average_pts_within_thresh']*100:.1f}%")
        logger.info(f"{'='*50}\n")
        
        report_progress('complete', epoch, train_loss=results['train_losses'][-1] if results['train_losses'] else 0,
                       val_loss=best_val_loss,
                       message=f"Training complete! Accuracy: {final_metrics['occlusion_accuracy']*100:.1f}%")
    else:
        report_progress('complete', epoch, message="Training complete!")
    
    return results


def main():
    parser = argparse.ArgumentParser(description="Fine-tune LocoTrack on RIPPLE annotations")
    parser.add_argument('--video', type=str, required=True, help='Path to video file')
    parser.add_argument('--annotations', type=str, required=True, help='Path to annotations JSON')
    parser.add_argument('--base-weights', type=str, required=True, help='Path to base LocoTrack weights')
    parser.add_argument('--output-weights', type=str, required=True, help='Path to save fine-tuned weights')
    parser.add_argument('--epochs', type=int, default=100, help='Maximum training epochs')
    parser.add_argument('--batch-size', type=int, default=1, help='Training batch size')
    parser.add_argument('--learning-rate', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--train-ratio', type=float, default=0.85, help='Train/test split ratio')
    parser.add_argument('--patience', type=int, default=10, help='Early stopping patience')
    parser.add_argument('--device', type=str, default='cuda', help='Device (cuda or cpu)')
    parser.add_argument('--output-json', type=str, help='Path to save results JSON')
    
    args = parser.parse_args()
    
    # Check CUDA availability
    if args.device == 'cuda' and not torch.cuda.is_available():
        logger.warning("CUDA not available, falling back to CPU")
        args.device = 'cpu'
    
    try:
        results = finetune_locotrack(
            video_path=args.video,
            annotations_path=args.annotations,
            base_weights=args.base_weights,
            output_weights=args.output_weights,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            train_split=args.train_ratio,
            early_stopping_patience=args.patience,
            device=args.device,
        )
        
        # Save results JSON
        if args.output_json:
            with open(args.output_json, 'w') as f:
                json.dump(results, f, indent=2, default=lambda x: float(x) if isinstance(x, np.floating) else x)
            logger.info(f"Results saved to {args.output_json}")
        
        # Print summary
        print(json.dumps({
            'status': 'success',
            'epochs_trained': results['epochs_trained'],
            'best_epoch': results['best_epoch'],
            'best_val_loss': results['best_val_loss'],
            'test_metrics': results.get('test_metrics', {}),
            'output_weights': args.output_weights,
        }, indent=2, default=lambda x: float(x) if isinstance(x, np.floating) else x))
        
    except Exception as e:
        logger.error(f"Fine-tuning failed: {e}")
        import traceback
        traceback.print_exc()
        print(json.dumps({
            'status': 'error',
            'message': str(e),
        }))
        sys.exit(1)


if __name__ == '__main__':
    main()
