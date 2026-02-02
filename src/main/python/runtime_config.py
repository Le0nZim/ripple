#!/usr/bin/env python3
"""RIPPLE Runtime Configuration and Hardware Detection.

This module provides automatic detection of hardware capabilities and
configures the application to run in the appropriate mode (CPU or GPU).

Usage:
    from runtime_config import RuntimeConfig
    
    config = RuntimeConfig()
    print(f"Running in {config.mode} mode")
    
    if config.can_use_raft:
        # Use RAFT optical flow
        pass
    else:
        # Fall back to DIS optical flow
        pass

The module is designed to be imported early in the application startup
and provides a single source of truth for hardware capabilities.
"""

import os
import sys
import platform
import subprocess
import json
import logging
from dataclasses import dataclass, field
from typing import Optional, Dict, List, Any
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("RIPPLE.RuntimeConfig")


@dataclass
class GPUInfo:
    """Information about a detected NVIDIA CUDA GPU."""
    index: int
    name: str
    memory_total_mb: int
    memory_free_mb: int
    cuda_capability: tuple
    driver_version: str
    
    def __str__(self) -> str:
        return (f"GPU {self.index}: {self.name} "
                f"({self.memory_total_mb}MB, CUDA {self.cuda_capability[0]}.{self.cuda_capability[1]})")


@dataclass
class SystemInfo:
    """System information for diagnostics."""
    os_name: str
    os_version: str
    python_version: str
    python_executable: str
    cpu_count: int
    memory_total_gb: float
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "os": f"{self.os_name} {self.os_version}",
            "python": self.python_version,
            "python_path": self.python_executable,
            "cpu_count": self.cpu_count,
            "memory_gb": round(self.memory_total_gb, 1)
        }


class RuntimeConfig:
    """
    RIPPLE Runtime Configuration Manager.
    
    Automatically detects hardware capabilities and determines the
    appropriate execution mode for the current system.
    
    Supports:
    - NVIDIA CUDA GPUs (Linux, Windows)
    - CPU fallback (all platforms including macOS)
    
    Attributes:
        mode: "cpu" or "gpu" (CUDA)
        gpu_available: True if GPU acceleration is available
        gpus: List of detected GPUInfo objects
        system_info: SystemInfo about the current system
    """
    
    # Minimum CUDA compute capability required
    MIN_CUDA_CAPABILITY = (3, 5)
    
    # Minimum GPU memory required (MB)
    MIN_GPU_MEMORY_MB = 2048
    
    def __init__(self, force_mode: Optional[str] = None):
        """
        Initialize runtime configuration.
        
        Args:
            force_mode: Force a specific mode ("cpu" or "gpu"). 
                        If None, auto-detect based on hardware.
        """
        self._mode: str = "cpu"
        self._gpu_available: bool = False
        self._gpus: List[GPUInfo] = []
        self._torch_cuda_available: bool = False
        self._cuda_version: Optional[str] = None
        self._cudnn_version: Optional[str] = None
        self._system_info: Optional[SystemInfo] = None
        
        # Perform detection
        self._detect_system_info()
        self._detect_gpu()
        
        # Set mode
        if force_mode:
            valid_modes = ("cpu", "gpu")
            if force_mode not in valid_modes:
                raise ValueError(f"Invalid mode: {force_mode}. Must be one of {valid_modes}")
            if force_mode == "gpu" and not self._gpu_available:
                logger.warning("GPU mode requested but no CUDA GPU available. Falling back to CPU.")
                self._mode = "cpu"
            else:
                self._mode = force_mode
        else:
            # Auto-detect: prefer CUDA GPU, else CPU
            if self._gpu_available:
                self._mode = "gpu"
            else:
                self._mode = "cpu"
        
        logger.info(f"RIPPLE initialized in {self._mode.upper()} mode")
    
    @property
    def mode(self) -> str:
        """Current execution mode: 'cpu' or 'gpu'."""
        return self._mode
    
    @property
    def is_gpu_mode(self) -> bool:
        """True if running in GPU mode (CUDA)."""
        return self._mode == "gpu"
    
    @property
    def is_cpu_mode(self) -> bool:
        """True if running in CPU mode."""
        return self._mode == "cpu"
    
    @property
    def gpu_available(self) -> bool:
        """True if CUDA GPU acceleration is available."""
        return self._gpu_available
    
    @property
    def cuda_available(self) -> bool:
        """True if NVIDIA CUDA is available."""
        return self._gpu_available
    
    @property
    def gpus(self) -> List[GPUInfo]:
        """List of detected GPUs."""
        return self._gpus.copy()
    
    @property
    def system_info(self) -> SystemInfo:
        """System information."""
        return self._system_info
    
    @property
    def cuda_version(self) -> Optional[str]:
        """CUDA version if available."""
        return self._cuda_version
    
    @property
    def cudnn_version(self) -> Optional[str]:
        """cuDNN version if available."""
        return self._cudnn_version
    
    # =========================================================================
    # FEATURE AVAILABILITY
    # =========================================================================
    
    @property
    def can_use_raft(self) -> bool:
        """True if RAFT optical flow is available (requires CUDA GPU)."""
        return self._mode == "gpu"
    
    @property
    def can_use_locotrack(self) -> bool:
        """True if LocoTrack is available (requires CUDA GPU)."""
        return self._mode == "gpu"
    
    @property
    def can_use_trackmate_dog(self) -> bool:
        """True if TrackMate-style DoG detection is available (CPU)."""
        return True  # Always available
    
    @property
    def can_use_trackpy(self) -> bool:
        """True if TrackPy particle tracking is available (CPU)."""
        try:
            import trackpy
            return True
        except ImportError:
            return False
    
    @property
    def can_use_dis_flow(self) -> bool:
        """True if DIS optical flow is available (CPU, OpenCV)."""
        try:
            import cv2
            # Check if DIS is available
            dis = cv2.DISOpticalFlow_create()
            return True
        except (ImportError, AttributeError):
            return False
    
    def get_available_features(self) -> Dict[str, bool]:
        """Get a dictionary of all available features."""
        return {
            "raft_optical_flow": self.can_use_raft,
            "locotrack": self.can_use_locotrack,
            "trackmate_dog": self.can_use_trackmate_dog,
            "trackpy": self.can_use_trackpy,
            "dis_optical_flow": self.can_use_dis_flow,
        }
    
    # =========================================================================
    # TORCH DEVICE HELPERS
    # =========================================================================
    
    def get_torch_device(self) -> "torch.device":
        """Get the appropriate torch device for the current mode."""
        import torch
        if self._mode == "gpu" and torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")
    
    def get_device_string(self) -> str:
        """Get device string for torch ('cuda' or 'cpu')."""
        return "cuda" if self._mode == "gpu" else "cpu"
    
    # =========================================================================
    # DETECTION METHODS
    # =========================================================================
    
    def _detect_system_info(self) -> None:
        """Detect basic system information."""
        import multiprocessing
        
        # Get memory info
        memory_gb = 8.0  # Default
        try:
            import psutil
            memory_gb = psutil.virtual_memory().total / (1024 ** 3)
        except ImportError:
            # Try reading from /proc/meminfo on Linux
            try:
                with open('/proc/meminfo', 'r') as f:
                    for line in f:
                        if line.startswith('MemTotal:'):
                            memory_kb = int(line.split()[1])
                            memory_gb = memory_kb / (1024 ** 2)
                            break
            except:
                pass
        
        self._system_info = SystemInfo(
            os_name=platform.system(),
            os_version=platform.release(),
            python_version=platform.python_version(),
            python_executable=sys.executable,
            cpu_count=multiprocessing.cpu_count(),
            memory_total_gb=memory_gb
        )
    
    def _detect_gpu(self) -> None:
        """Detect GPU availability and capabilities."""
        # Try using PyTorch for CUDA detection
        try:
            import torch
            self._torch_cuda_available = torch.cuda.is_available()
            
            if self._torch_cuda_available:
                self._cuda_version = torch.version.cuda
                if torch.backends.cudnn.is_available():
                    self._cudnn_version = str(torch.backends.cudnn.version())
                
                # Get GPU info
                for i in range(torch.cuda.device_count()):
                    props = torch.cuda.get_device_properties(i)
                    
                    # Get memory info
                    try:
                        memory_total = props.total_memory // (1024 * 1024)
                        memory_free = memory_total  # Approximate
                    except:
                        memory_total = 0
                        memory_free = 0
                    
                    gpu_info = GPUInfo(
                        index=i,
                        name=props.name,
                        memory_total_mb=memory_total,
                        memory_free_mb=memory_free,
                        cuda_capability=(props.major, props.minor),
                        driver_version=self._get_nvidia_driver_version()
                    )
                    
                    # Check if GPU meets minimum requirements
                    if (gpu_info.cuda_capability >= self.MIN_CUDA_CAPABILITY and
                        gpu_info.memory_total_mb >= self.MIN_GPU_MEMORY_MB):
                        self._gpus.append(gpu_info)
                        logger.info(f"Detected: {gpu_info}")
                
                if self._gpus:
                    self._gpu_available = True
                else:
                    logger.warning("GPUs detected but none meet minimum requirements")
            else:
                logger.info("No CUDA-capable GPU detected by PyTorch")
                
        except ImportError:
            logger.warning("PyTorch not installed, cannot detect GPU")
        except Exception as e:
            logger.warning(f"Error detecting GPU: {e}")
    
    def _get_nvidia_driver_version(self) -> str:
        """Get NVIDIA driver version."""
        try:
            result = subprocess.run(
                ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
                capture_output=True, text=True, timeout=5
            )
            if result.returncode == 0:
                return result.stdout.strip().split('\n')[0]
        except:
            pass
        return "unknown"
    
    # =========================================================================
    # DIAGNOSTICS
    # =========================================================================
    
    def get_diagnostics(self) -> Dict[str, Any]:
        """Get full diagnostic information for troubleshooting."""
        return {
            "mode": self._mode,
            "system": self._system_info.to_dict() if self._system_info else {},
            "gpu": {
                "available": self._gpu_available,
                "torch_cuda": self._torch_cuda_available,
                "cuda_version": self._cuda_version,
                "cudnn_version": self._cudnn_version,
                "devices": [
                    {
                        "index": g.index,
                        "name": g.name,
                        "memory_mb": g.memory_total_mb,
                        "cuda_capability": f"{g.cuda_capability[0]}.{g.cuda_capability[1]}",
                        "driver": g.driver_version
                    }
                    for g in self._gpus
                ]
            },
            "features": self.get_available_features()
        }
    
    def print_diagnostics(self) -> None:
        """Print diagnostic information to console."""
        diag = self.get_diagnostics()
        
        print("\n" + "=" * 60)
        print("RIPPLE System Diagnostics")
        print("=" * 60)
        
        print(f"\nExecution Mode: {diag['mode'].upper()}")
        
        print("\nSystem:")
        sys_info = diag['system']
        print(f"  OS: {sys_info.get('os', 'Unknown')}")
        print(f"  Python: {sys_info.get('python', 'Unknown')}")
        print(f"  CPUs: {sys_info.get('cpu_count', 'Unknown')}")
        print(f"  Memory: {sys_info.get('memory_gb', 'Unknown')} GB")
        
        print("\nGPU:")
        gpu_info = diag['gpu']
        print(f"  Available: {'Yes' if gpu_info['available'] else 'No'}")
        if gpu_info['available']:
            print(f"  CUDA Version: {gpu_info['cuda_version']}")
            print(f"  cuDNN Version: {gpu_info['cudnn_version']}")
            for dev in gpu_info['devices']:
                print(f"  GPU {dev['index']}: {dev['name']}")
                print(f"    Memory: {dev['memory_mb']} MB")
                print(f"    CUDA Capability: {dev['cuda_capability']}")
        
        print("\nAvailable Features:")
        for feature, available in diag['features'].items():
            status = "✓" if available else "✗"
            print(f"  [{status}] {feature}")
        
        print("\n" + "=" * 60)
    
    def save_diagnostics(self, path: str) -> None:
        """Save diagnostic information to a JSON file."""
        with open(path, 'w') as f:
            json.dump(self.get_diagnostics(), f, indent=2)
        logger.info(f"Diagnostics saved to {path}")


# =============================================================================
# MODULE-LEVEL SINGLETON
# =============================================================================

# Global configuration instance (lazy initialization)
_config: Optional[RuntimeConfig] = None


def get_config(force_mode: Optional[str] = None) -> RuntimeConfig:
    """
    Get the global RuntimeConfig instance.
    
    Args:
        force_mode: Force a specific mode on first initialization.
                   Ignored on subsequent calls.
    
    Returns:
        RuntimeConfig instance
    """
    global _config
    if _config is None:
        _config = RuntimeConfig(force_mode=force_mode)
    return _config


def get_mode() -> str:
    """Get the current execution mode ('cpu' or 'gpu')."""
    return get_config().mode


def is_gpu_mode() -> bool:
    """Check if running in GPU mode."""
    return get_config().is_gpu_mode


def get_device() -> str:
    """Get the torch device string ('cuda' or 'cpu')."""
    return get_config().get_device_string()


# =============================================================================
# CLI INTERFACE
# =============================================================================

def main():
    """Command-line interface for system diagnostics."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="RIPPLE Runtime Configuration and System Diagnostics"
    )
    parser.add_argument(
        "--mode", choices=["cpu", "gpu"],
        help="Force a specific execution mode"
    )
    parser.add_argument(
        "--json", action="store_true",
        help="Output diagnostics as JSON"
    )
    parser.add_argument(
        "--save", metavar="FILE",
        help="Save diagnostics to a file"
    )
    parser.add_argument(
        "--features", action="store_true",
        help="List available features only"
    )
    
    args = parser.parse_args()
    
    config = RuntimeConfig(force_mode=args.mode)
    
    if args.features:
        features = config.get_available_features()
        for name, available in features.items():
            status = "available" if available else "not available"
            print(f"{name}: {status}")
    elif args.json:
        print(json.dumps(config.get_diagnostics(), indent=2))
    else:
        config.print_diagnostics()
    
    if args.save:
        config.save_diagnostics(args.save)


if __name__ == "__main__":
    main()
