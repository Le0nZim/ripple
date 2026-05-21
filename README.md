# RIPPLE - Video Annotation Tool for Biology

A hybrid Java/Python application for video annotation and particle tracking in biological microscopy.

## 🎬 Demos

A small selection of demo videos is embedded below. The **complete set** (app demos, RIPPLE vs. full manual annotation comparisons, deep-learning tracker failures, and SLEAP results) is available in the companion Google Drive folder:

📁 **[Full demo collection on Google Drive](https://drive.google.com/drive/folders/186gzAZ_zwIbH4ug-lEKC2fDDzT9ss4Yr?usp=sharing)**

### App demo — neural-activity dataset

A walkthrough of the RIPPLE annotation workflow on a neural-activity microscopy video.

<video src="https://github.com/Le0nZim/ripple/raw/main/demo_videos/app_demo_neural.mp4" controls width="720"></video>

### RIPPLE vs. full manual annotation — freely-moving dataset

Overlay of RIPPLE predictions (squares) vs. ground-truth manual annotations (circles); line length encodes per-frame error. One color per matched track pair.

<video src="https://github.com/Le0nZim/ripple/raw/main/demo_videos/ripple_vs_manual_freely.mp4" controls width="720"></video>

### Deep-learning tracker failure — freely-moving dataset

LocoTrack predictions on the same freely-moving dataset, illustrating typical failure modes of a state-of-the-art deep-learning point tracker on biological microscopy data.

<video src="https://github.com/Le0nZim/ripple/raw/main/demo_videos/dl_tracker_failure_freely.mp4" controls width="720"></video>

> The Google Drive folder also contains companion `tracks.xlsx` files with the raw matched coordinates and per-dataset `README.txt` files describing the visual encoding and frame counts.

## 🚀 Quick Start

### Prerequisites
- **Java 11+** (OpenJDK 17+ recommended)
- **Conda** (Miniconda or Anaconda)
- **Maven 3.8+** (needed to build the Java app)

### One-Command Installation

```bash
bash quickstart.sh
```

That's it! The script will:
1. Create a new conda environment `ripple-env` (or use existing one)
2. Install all Python dependencies (GPU or CPU automatically detected)
3. Build the Java application
4. Launch RIPPLE

### Windows
```cmd
quickstart.bat
```

## 🔨 Build & Run (Maven)

If you prefer to build/run manually (instead of using `quickstart.sh` / `quickstart.bat`), Maven produces a single shaded JAR at `target/ripple.jar`.

### Build

```bash
mvn clean package -DskipTests
```

Output:

- `target/ripple.jar`

### Run

```bash
java -jar target/ripple.jar
```

### Developer run (no packaging)

```bash
mvn exec:java
```

> Note: RIPPLE is a hybrid Java/Python app — you’ll typically still want your Conda environment set up so the Python tracking backend can run.

## 🧠 Persistent Tracking Server (optional)

For iterative workflows, RIPPLE includes an optional long-running tracking server that keeps models warm in memory. The helper scripts manage the server lifecycle and forward tracking commands.

### Windows

```cmd
scripts\run_persistent_tracking.bat start
scripts\run_persistent_tracking.bat status
scripts\run_persistent_tracking.bat ping
```

### Linux/macOS

By default (Windows parity), Linux/macOS also uses TCP on `127.0.0.1:9876`.

```bash
./scripts/run_persistent_tracking.sh start
./scripts/run_persistent_tracking.sh status
./scripts/run_persistent_tracking.sh ping
```

If you prefer a Unix domain socket transport on Linux/macOS:

```bash
RIPPLE_TRANSPORT=unix SOCKET_PATH=/tmp/ripple-env.sock ./scripts/run_persistent_tracking.sh start
RIPPLE_TRANSPORT=unix SOCKET_PATH=/tmp/ripple-env.sock ./scripts/run_persistent_tracking.sh status
```

### Configuration (both)

You can override defaults via environment variables:

- `CONDA_ENV` (default: `ripple-env`)
- `MODEL_SIZE` (default: `large`)
- `SOCKET_HOST` / `SOCKET_PORT` (TCP mode)
- `SOCKET_PATH` (Unix socket mode on Linux/macOS)

## 📁 Project Structure

```
RIPPLE/
├── quickstart.sh / .bat         # One-click installer and launcher
├── pom.xml                      # Maven build configuration
│
├── src/main/
│   ├── java/com/ripple/         # Java GUI application
│   └── python/                  # Python tracking backend
│
├── requirements/                # Python dependencies
│   ├── requirements-cpu.txt     # CPU-only packages
│   └── requirements-gpu.txt     # GPU (CUDA) packages
│
├── conda/                       # Conda environment files
│   ├── environment.yml          # GPU environment (CUDA)
│   └── environment-cpu.yml      # CPU-only environment
│
└── scripts/                     # Build and utility scripts
```

## 📋 Requirements

### System Requirements
- **Java**: 11+ (17+ recommended)
- **Conda**: Miniconda or Anaconda
- **Maven**: 3.8+ (for building from source)

### GPU Support

| Platform | GPU | Mode | Available Features |
|----------|-----|------|-------------------|
| Linux/Windows | NVIDIA CUDA | GPU | RAFT, LocoTrack, TrackPy, DIS |
| macOS | None | CPU | TrackPy, DIS |
| Any | None | CPU | TrackPy, DIS |

> **Note:** RAFT and LocoTrack require an NVIDIA GPU with CUDA support.
> On systems without CUDA, RIPPLE runs in CPU mode with TrackPy and DIS optical flow.

## 📖 Features

- **Video Annotation**: Load and annotate TIFF video stacks
- **Particle Tracking**:
  - RAFT optical flow (GPU only)
  - LocoTrack point tracking (GPU only)
  - TrackPy (CPU/GPU)
  - DIS optical flow (CPU)
- **Track Correction Methods**:
  - Full-Blend: Pure optical flow with bidirectional blending
  - Corridor-DP: Dynamic programming with adaptive corridor search
  - Blob-Assisted: Flow + blob detection for particle-like objects
- **Trajectory Editing**: Multi-anchor trajectory optimization
- **Export**: JSON and CSV export formats

## 🔧 Troubleshooting

### Java not found
Install OpenJDK 25:
```bash
# Ubuntu/Debian
sudo apt install openjdk-25-jdk

# macOS
brew install openjdk@25

# Windows: Download from https://www.oracle.com/java/technologies/downloads/#jdk25-windows
```

### Conda not found
Install Miniconda:
```bash
# Linux
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh

# macOS
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-MacOSX-x86_64.sh
bash Miniconda3-latest-MacOSX-x86_64.sh
```

### GPU not detected
Ensure NVIDIA drivers and CUDA toolkit are installed:
```bash
nvidia-smi  # Should show GPU info
```

## 📄 License

MIT License. See [LICENSE](https://github.com/Le0nZim/ripple/blob/main/LICENCE).

## 🙏 Acknowledgments

- [ImageJ](https://fiji.sc/) - Image analysis platform
- [RAFT](https://github.com/princeton-vl/RAFT) - Optical flow
- [LocoTrack](https://github.com/cvlab-kaist/locotrack) - Point tracking
- [TrackPy](https://soft-matter.github.io/trackpy/v0.7/) - Particle tracking
- [TAP-Vid](https://github.com/google-deepmind/tapnet/blob/main/tapnet/tapvid/README.md) - Track Assist
