#!/usr/bin/env python3
"""
Real-Time Training Dashboard for LocoTrack Fine-tuning.

Provides beautiful, informative progress updates during training so users
know exactly what's happening and when it will finish.

Features:
- Real-time progress bars with ETA
- Loss curves (text-based visualization)
- GPU memory monitoring
- Early stopping indicators
- Automatic checkpointing status

Can be used standalone or integrated with the Java UI via IPC.

Author: RIPPLE Team
Date: 2026-01-30
"""

import time
import sys
import os
from dataclasses import dataclass, field
from typing import Callable, Optional, Dict, List, Any
from collections import deque
import json


@dataclass
class TrainingMetrics:
    """Container for training metrics at a point in time."""
    epoch: int = 0
    total_epochs: int = 0
    batch: int = 0
    total_batches: int = 0
    loss: float = 0.0
    best_loss: float = float('inf')
    position_error: float = 0.0
    learning_rate: float = 0.0
    elapsed_seconds: float = 0.0
    estimated_remaining_seconds: float = 0.0
    samples_per_second: float = 0.0
    gpu_memory_used_mb: float = 0.0
    gpu_memory_total_mb: float = 0.0
    phase: str = "training"  # training, validation, complete
    message: str = ""
    val_loss: float = 0.0
    improvement: float = 0.0  # % improvement from baseline


@dataclass
class TrainingHistory:
    """Stores training history for visualization."""
    train_losses: List[float] = field(default_factory=list)
    val_losses: List[float] = field(default_factory=list)
    learning_rates: List[float] = field(default_factory=list)
    position_errors: List[float] = field(default_factory=list)
    epochs: List[int] = field(default_factory=list)


class TrainingDashboard:
    """
    Real-time training dashboard with CLI visualization.
    
    Displays:
    - Progress bar with percentage and ETA
    - Current loss and best loss
    - Learning rate
    - GPU memory usage
    - Mini loss plot (ASCII art)
    
    Args:
        total_epochs: Total number of epochs
        batches_per_epoch: Number of batches per epoch
        callback: Optional callback for UI integration (receives TrainingMetrics)
        quiet: If True, suppress CLI output (callback only)
    """
    
    def __init__(self, total_epochs: int, batches_per_epoch: int,
                 callback: Callable[[TrainingMetrics], None] = None,
                 quiet: bool = False):
        self.total_epochs = total_epochs
        self.batches_per_epoch = batches_per_epoch
        self.callback = callback
        self.quiet = quiet
        
        self.start_time = time.time()
        self.best_loss = float('inf')
        self.best_epoch = 0
        self.history = TrainingHistory()
        
        # For smooth ETA estimation
        self.recent_batch_times = deque(maxlen=20)
        self.last_batch_time = self.start_time
        
        # Baseline for improvement calculation
        self.baseline_error = None
        
        # Check for GPU
        self.has_gpu = self._check_gpu()
    
    def _check_gpu(self) -> bool:
        """Check if GPU monitoring is available."""
        try:
            import torch
            return torch.cuda.is_available()
        except ImportError:
            return False
    
    def _get_gpu_memory(self) -> tuple:
        """Get GPU memory usage."""
        if not self.has_gpu:
            return 0, 0
        try:
            import torch
            used = torch.cuda.memory_allocated() / 1024**2
            total = torch.cuda.get_device_properties(0).total_memory / 1024**2
            return used, total
        except Exception:
            return 0, 0
    
    def _format_time(self, seconds: float) -> str:
        """Format seconds as human-readable time."""
        if seconds < 60:
            return f"{int(seconds)}s"
        elif seconds < 3600:
            mins = int(seconds // 60)
            secs = int(seconds % 60)
            return f"{mins}m {secs}s"
        else:
            hours = int(seconds // 3600)
            mins = int((seconds % 3600) // 60)
            return f"{hours}h {mins}m"
    
    def _make_progress_bar(self, progress: float, width: int = 30) -> str:
        """Create ASCII progress bar."""
        filled = int(width * progress)
        bar = "█" * filled + "░" * (width - filled)
        return f"[{bar}]"
    
    def _make_mini_plot(self, values: List[float], width: int = 20, height: int = 5) -> List[str]:
        """Create ASCII mini plot of loss curve."""
        if len(values) < 2:
            return ["  (collecting data...)"]
        
        # Take last N values
        values = values[-width:]
        
        min_val = min(values)
        max_val = max(values)
        val_range = max_val - min_val
        
        if val_range < 1e-8:
            val_range = 1.0
        
        lines = []
        for row in range(height):
            threshold = max_val - (row + 0.5) * val_range / height
            line = "  "
            for val in values:
                if val >= threshold:
                    line += "█"
                else:
                    line += " "
            lines.append(line)
        
        # Add axis labels
        lines.insert(0, f"  {max_val:.4f}")
        lines.append(f"  {min_val:.4f}")
        
        return lines
    
    def start(self):
        """Initialize dashboard for new training run."""
        self.start_time = time.time()
        self.best_loss = float('inf')
        self.history = TrainingHistory()
        
        if not self.quiet:
            print("\n" + "=" * 60)
            print("🚀 LocoTrack Fine-tuning Started")
            print("=" * 60)
            print(f"   Epochs: {self.total_epochs}")
            print(f"   Batches/epoch: {self.batches_per_epoch}")
            if self.has_gpu:
                _, total = self._get_gpu_memory()
                print(f"   GPU Memory: {total:.0f} MB available")
            print()
    
    def update(self, epoch: int, batch: int, loss: float, 
               lr: float = 0.0, position_error: float = 0.0,
               val_loss: float = None, phase: str = "training"):
        """
        Update dashboard with current training state.
        
        Args:
            epoch: Current epoch (0-indexed)
            batch: Current batch (0-indexed)
            loss: Current training loss
            lr: Current learning rate
            position_error: Current position error in pixels
            val_loss: Validation loss (if validation phase)
            phase: 'training', 'validation', or 'complete'
        """
        now = time.time()
        elapsed = now - self.start_time
        
        # Track batch time for ETA
        batch_time = now - self.last_batch_time
        self.recent_batch_times.append(batch_time)
        self.last_batch_time = now
        
        # Calculate progress
        total_batches_done = epoch * self.batches_per_epoch + batch + 1
        total_batches = self.total_epochs * self.batches_per_epoch
        progress = total_batches_done / total_batches
        
        # Estimate remaining time
        if len(self.recent_batch_times) > 0:
            avg_batch_time = sum(self.recent_batch_times) / len(self.recent_batch_times)
            remaining_batches = total_batches - total_batches_done
            remaining_seconds = remaining_batches * avg_batch_time
        else:
            remaining_seconds = 0
        
        # Track best loss
        if val_loss is not None and val_loss < self.best_loss:
            self.best_loss = val_loss
            self.best_epoch = epoch
        elif val_loss is None and loss < self.best_loss:
            self.best_loss = loss
            self.best_epoch = epoch
        
        # Store history
        if phase == "training" and batch == self.batches_per_epoch - 1:
            self.history.train_losses.append(loss)
            self.history.learning_rates.append(lr)
            self.history.epochs.append(epoch)
        if val_loss is not None:
            self.history.val_losses.append(val_loss)
        if position_error > 0:
            self.history.position_errors.append(position_error)
            if self.baseline_error is None:
                self.baseline_error = position_error
        
        # Calculate improvement
        improvement = 0.0
        if self.baseline_error and position_error > 0:
            improvement = (self.baseline_error - position_error) / self.baseline_error * 100
        
        # Get GPU memory
        gpu_used, gpu_total = self._get_gpu_memory()
        
        # Create metrics object
        metrics = TrainingMetrics(
            epoch=epoch + 1,
            total_epochs=self.total_epochs,
            batch=batch + 1,
            total_batches=self.batches_per_epoch,
            loss=loss,
            best_loss=self.best_loss,
            position_error=position_error,
            learning_rate=lr,
            elapsed_seconds=elapsed,
            estimated_remaining_seconds=remaining_seconds,
            samples_per_second=total_batches_done / elapsed if elapsed > 0 else 0,
            gpu_memory_used_mb=gpu_used,
            gpu_memory_total_mb=gpu_total,
            phase=phase,
            val_loss=val_loss if val_loss is not None else 0.0,
            improvement=improvement,
        )
        
        # Send to callback
        if self.callback:
            self.callback(metrics)
        
        # CLI display
        if not self.quiet:
            self._display_progress(metrics)
    
    def _display_progress(self, m: TrainingMetrics):
        """Display progress in terminal."""
        progress = (m.epoch - 1 + m.batch / m.total_batches) / m.total_epochs
        
        # Build status line
        bar = self._make_progress_bar(progress)
        elapsed_str = self._format_time(m.elapsed_seconds)
        remaining_str = self._format_time(m.estimated_remaining_seconds)
        
        # Color-coded loss indicator
        if m.loss <= m.best_loss * 1.01:
            loss_indicator = "🟢"
        elif m.loss <= m.best_loss * 1.1:
            loss_indicator = "🟡"
        else:
            loss_indicator = "🔴"
        
        # Main progress line
        status = (
            f"\r{bar} {progress*100:5.1f}% | "
            f"Epoch {m.epoch}/{m.total_epochs} | "
            f"{loss_indicator} Loss: {m.loss:.4f} (best: {m.best_loss:.4f}) | "
            f"⏱️ {elapsed_str} / ~{remaining_str}"
        )
        
        # GPU memory
        if m.gpu_memory_total_mb > 0:
            gpu_pct = m.gpu_memory_used_mb / m.gpu_memory_total_mb * 100
            status += f" | 🎮 {gpu_pct:.0f}%"
        
        print(status, end="", flush=True)
        
        # End of epoch summary
        if m.batch == m.total_batches and m.phase == "training":
            print()  # New line
    
    def epoch_end(self, epoch: int, train_loss: float, val_loss: float = None,
                  position_error: float = 0.0, saved_checkpoint: bool = False):
        """
        Called at end of epoch for summary.
        
        Args:
            epoch: Epoch number (0-indexed)
            train_loss: Training loss
            val_loss: Validation loss
            position_error: Position error in pixels
            saved_checkpoint: Whether a checkpoint was saved
        """
        if self.quiet:
            return
        
        print(f"\n📊 Epoch {epoch + 1}/{self.total_epochs} Summary:")
        print(f"   Train Loss: {train_loss:.4f}")
        
        if val_loss is not None:
            print(f"   Val Loss:   {val_loss:.4f}")
        
        if position_error > 0:
            print(f"   Position Error: {position_error:.2f} px")
            if self.baseline_error:
                improvement = (self.baseline_error - position_error) / self.baseline_error * 100
                if improvement > 0:
                    print(f"   Improvement: ↑ {improvement:.1f}%")
                else:
                    print(f"   Improvement: ↓ {abs(improvement):.1f}%")
        
        if saved_checkpoint:
            print("   💾 Checkpoint saved (new best!)")
        
        print()
    
    def show_loss_plot(self):
        """Show ASCII loss curve."""
        if self.quiet or len(self.history.train_losses) < 2:
            return
        
        print("\n📈 Training Progress:")
        lines = self._make_mini_plot(self.history.train_losses)
        for line in lines:
            print(line)
        print(f"     {'─' * len(self.history.train_losses)}")
        print(f"     Epoch 1{' ' * (len(self.history.train_losses) - 4)}{self.total_epochs}")
    
    def finish(self, final_metrics: Dict = None):
        """Training complete - show final summary."""
        elapsed = time.time() - self.start_time
        
        if self.quiet:
            return TrainingMetrics(
                phase="complete",
                elapsed_seconds=elapsed,
                best_loss=self.best_loss,
            )
        
        print("\n" + "=" * 60)
        print("✅ Training Complete!")
        print("=" * 60)
        print(f"   Total time: {self._format_time(elapsed)}")
        print(f"   Best loss: {self.best_loss:.4f} (epoch {self.best_epoch + 1})")
        
        if len(self.history.position_errors) > 0:
            final_error = self.history.position_errors[-1]
            print(f"   Final position error: {final_error:.2f} px")
            if self.baseline_error:
                improvement = (self.baseline_error - final_error) / self.baseline_error * 100
                print(f"   Total improvement: {improvement:.1f}%")
        
        if final_metrics:
            print(f"\n   Final Test Metrics:")
            for key, value in final_metrics.items():
                if isinstance(value, float):
                    print(f"      {key}: {value:.4f}")
                else:
                    print(f"      {key}: {value}")
        
        self.show_loss_plot()
        print()
        
        return TrainingMetrics(
            phase="complete",
            elapsed_seconds=elapsed,
            best_loss=self.best_loss,
        )


class IPCProgressReporter:
    """
    Progress reporter for Java UI integration.
    
    Sends JSON progress updates that can be parsed by the Java frontend.
    """
    
    def __init__(self, output_stream=None):
        self.output = output_stream or sys.stderr
    
    def __call__(self, metrics: TrainingMetrics):
        """Send metrics as JSON to output stream."""
        data = {
            "type": "training_progress",
            "epoch": metrics.epoch,
            "total_epochs": metrics.total_epochs,
            "batch": metrics.batch,
            "total_batches": metrics.total_batches,
            "loss": metrics.loss,
            "best_loss": metrics.best_loss,
            "position_error": metrics.position_error,
            "learning_rate": metrics.learning_rate,
            "elapsed_seconds": metrics.elapsed_seconds,
            "estimated_remaining_seconds": metrics.estimated_remaining_seconds,
            "gpu_memory_used_mb": metrics.gpu_memory_used_mb,
            "gpu_memory_total_mb": metrics.gpu_memory_total_mb,
            "phase": metrics.phase,
            "improvement": metrics.improvement,
        }
        
        # Send as single-line JSON
        json_str = json.dumps(data)
        print(f"RIPPLE_PROGRESS:{json_str}", file=self.output, flush=True)


def create_progress_callback(mode: str = "cli") -> Callable:
    """
    Create a progress callback based on mode.
    
    Args:
        mode: "cli" for terminal display, "ipc" for Java integration, "quiet" for no output
        
    Returns:
        Callback function
    """
    if mode == "ipc":
        return IPCProgressReporter()
    elif mode == "quiet":
        return lambda x: None
    else:
        # CLI mode - the dashboard handles its own display
        return None


# =============================================================================
# STANDALONE DEMO
# =============================================================================

def demo():
    """Run a demo of the training dashboard."""
    import random
    
    print("\n🎬 Training Dashboard Demo\n")
    
    epochs = 5
    batches = 20
    
    dashboard = TrainingDashboard(epochs, batches)
    dashboard.baseline_error = 10.0  # Starting at 10px error
    dashboard.start()
    
    loss = 1.0
    
    for epoch in range(epochs):
        for batch in range(batches):
            # Simulate training
            time.sleep(0.05)
            loss = loss * 0.98 + random.uniform(-0.02, 0.01)
            lr = 1e-4 * (0.95 ** epoch)
            
            dashboard.update(
                epoch=epoch,
                batch=batch,
                loss=loss,
                lr=lr,
                position_error=10 - epoch * 1.5 + random.uniform(-0.5, 0.5),
            )
        
        # End of epoch
        val_loss = loss + random.uniform(-0.05, 0.05)
        dashboard.epoch_end(
            epoch=epoch,
            train_loss=loss,
            val_loss=val_loss,
            position_error=10 - epoch * 1.5,
            saved_checkpoint=(val_loss < dashboard.best_loss)
        )
    
    dashboard.finish({
        "test_accuracy": 0.95,
        "avg_position_error": 2.5,
    })


if __name__ == "__main__":
    demo()
