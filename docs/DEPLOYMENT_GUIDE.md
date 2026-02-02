# RIPPLE Deployment Guide

## Overview

This guide provides a comprehensive solution for deploying RIPPLE as a user-friendly, cross-platform application that biology labs can easily install and use without technical expertise.

## Architecture

RIPPLE is a hybrid application consisting of:
- **Java GUI** (VideoAnnotationTool) - Built with ImageJ/Swing
- **Python Backend** (tracking_server.py, locotrack_flow.py) - PyTorch-based tracking
- **Shell Interface** - Orchestrates the Python server

## Deployment Strategy

### Option 1: jpackage + Embedded Python (Recommended)

This approach bundles everything into a single installable package:

```
RIPPLE/
├── RIPPLE.exe (Windows) or RIPPLE (Linux)
├── runtime/                    # Bundled JRE
├── python/                     # Embedded Python environment
│   ├── python.exe / python3
│   └── Lib/site-packages/
├── locotrack_pytorch/          # LocoTrack model code and weights
│   ├── models/                 # Model architecture
│   └── weights/                # Pre-trained weights
│       ├── locotrack_base.ckpt
│       └── locotrack_small.ckpt
└── resources/
    ├── tracking_server.py
    └── locotrack_flow.py
```

### Option 2: Docker Container

For users comfortable with Docker:
- Single `docker run` command
- Automatically handles GPU passthrough on Linux
- Works on any system with Docker installed

### Option 3: Conda Installer

For users with Anaconda/Miniconda:
- `conda env create -f environment.yml`
- Platform-specific environment files

## CPU vs GPU Mode

The application automatically detects hardware capabilities:

| Mode | Features | Hardware Required |
|------|----------|-------------------|
| CPU | TrackMate DoG, TrackPy, Basic Tracking | Any modern CPU |
| GPU | All CPU features + RAFT, LocoTrack | NVIDIA GPU with CUDA |

### Automatic Detection

```python
# runtime_config.py handles this automatically
import torch
GPU_AVAILABLE = torch.cuda.is_available()
COMPUTE_MODE = "gpu" if GPU_AVAILABLE else "cpu"
```

## Build Instructions

### Prerequisites

- JDK 17+ (with jpackage)
- Python 3.10+
- Node.js 18+ (for Electron wrapper, optional)

### Step 1: Build the Java Application

```bash
# Create fat JAR with all dependencies
mvn clean package -Pproduction

# Create native installer
./build/create_installer.sh
```

### Step 2: Bundle Python Environment

```bash
# Create standalone Python environment
./build/bundle_python.sh
```

### Step 3: Download Models

```bash
# Pre-download all required models
python build/download_models.py
```

### Step 4: Create Final Package

```bash
# Linux
./build/package_linux.sh

# Windows
.\build\package_windows.ps1
```

## File Structure After Restructuring

```
RIPPLE/
├── pom.xml                          # Maven build configuration
├── build/                           # Build and packaging scripts
│   ├── create_installer.sh
│   ├── bundle_python.sh
│   ├── download_models.py
│   ├── package_linux.sh
│   └── package_windows.ps1
├── src/
│   └── main/
│       ├── java/
│       │   └── com/example/imagej/
│       │       ├── VideoAnnotationTool.java
│       │       ├── ConfigurationManager.java
│       │       ├── Constants.java
│       │       ├── Anchor.java
│       │       └── RuntimeConfig.java    # NEW: CPU/GPU detection
│       ├── resources/
│       │   └── runtime/
│       │       ├── tracking_server.py
│       │       ├── locotrack_flow.py
│       │       ├── trackmate_dog.py
│       │       ├── trackpy_flow.py
│       │       └── runtime_config.py     # NEW: Python runtime config
│       └── python/
│           ├── requirements-base.txt     # CPU-only dependencies
│           ├── requirements-gpu.txt      # GPU dependencies (CUDA)
│           └── environment.yml           # Conda environment file
├── installer/                       # Installer configurations
│   ├── windows/
│   │   └── ripple.iss              # Inno Setup script
│   └── linux/
│       └── ripple.desktop
└── docker/
    ├── Dockerfile
    └── docker-compose.yml
```

## Cross-Platform Considerations

### Windows

- Uses embedded Python from python.org
- Bundled with NVIDIA CUDA Runtime (optional, for GPU mode)
- Created with Inno Setup or jpackage

### Linux

- AppImage or .deb/.rpm packages
- GPU support via system NVIDIA drivers
- Optional: Snap or Flatpak packaging

## Testing Installation

After installation, users can verify the setup:

```bash
# Windows
RIPPLE.exe --check-system

# Linux
./RIPPLE --check-system
```

This will display:
- Python version and location
- Available compute mode (CPU/GPU)
- GPU information (if available)
- Model file status

## Troubleshooting

### Common Issues

1. **"CUDA not available"**: Install NVIDIA drivers and CUDA toolkit
2. **"Model files missing"**: Run `RIPPLE --download-models`
3. **"Python not found"**: Reinstall RIPPLE with bundled Python option

## Support

For biology labs needing assistance:
1. Use the CPU mode if GPU setup is problematic
2. Contact support with the output of `RIPPLE --check-system`
