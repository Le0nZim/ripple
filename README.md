# RIPPLE - Video Annotation Tool for Biology

A hybrid Java/Python application for video annotation and particle tracking in biological microscopy.

## 🎬 Demos

A small selection of demo videos is embedded below. The **complete set** (app demos, RIPPLE vs. full manual annotation comparisons, deep-learning tracker failures, and SLEAP results) is available in the companion Google Drive folder:

📁 **[Full demo collection on Google Drive](https://drive.google.com/drive/folders/186gzAZ_zwIbH4ug-lEKC2fDDzT9ss4Yr?usp=sharing)**

### App demo — neural-activity dataset

A walkthrough of the RIPPLE annotation workflow on a neural-activity microscopy video. *(Preview shows the first ~12s; click for the full video.)*

[![App demo — neural-activity dataset](https://github.com/Le0nZim/ripple/releases/download/demos-v1/app_demo_neural.gif)](https://github.com/Le0nZim/ripple/releases/download/demos-v1/app_demo_neural.mp4)

### RIPPLE vs. full manual annotation — freely-moving dataset

Overlay of RIPPLE predictions (squares) vs. ground-truth manual annotations (circles); line length encodes per-frame error. One color per matched track pair. *(Preview shows the first ~12s; click for the full video.)*

[![RIPPLE vs. full manual annotation — freely-moving dataset](https://github.com/Le0nZim/ripple/releases/download/demos-v1/ripple_vs_manual_freely.gif)](https://github.com/Le0nZim/ripple/releases/download/demos-v1/ripple_vs_manual_freely.mp4)

### Deep-learning tracker failure — freely-moving dataset

LocoTrack predictions on the same freely-moving dataset, illustrating typical failure modes of a state-of-the-art deep-learning point tracker on biological microscopy data. *(Preview shows the first ~12s; click for the full video.)*

[![Deep-learning tracker failure — freely-moving dataset](https://github.com/Le0nZim/ripple/releases/download/demos-v1/dl_tracker_failure_freely.gif)](https://github.com/Le0nZim/ripple/releases/download/demos-v1/dl_tracker_failure_freely.mp4)

> The Google Drive folder also contains companion `tracks.xlsx` files with the raw matched coordinates and per-dataset `README.txt` files describing the visual encoding and frame counts.

## 📥 Fast download (avoid copying local bloat)

The **git repository is only ~2–3 MB**. Most slow downloads come from copying the whole working folder, which accumulates local-only data:

| Path | Typical size | Needed for build? |
|------|--------------|-------------------|
| `.venv/` | ~460 MB | **No** — use `bash quickstart.sh` (creates conda `ripple-env`) |
| `tools/` | ~200 MB | **No** — portable JDK/Maven downloaded by quickstart if needed |
| `locotrack_pytorch/weights/*.ckpt` | ~76 MB | **No** for CPU; optional for GPU ([download links](locotrack_pytorch/weights/README.md)) |
| `target/` | ~5 MB | **No** — rebuilt by Maven |

**Recommended:**

```bash
git clone https://github.com/Le0nZim/ripple.git
cd ripple
bash quickstart.sh
```

**Minimal zip (tracked source only, no venv/weights/build):**

```bash
bash scripts/make_source_archive.sh
unzip ripple-source.zip -d ripple && cd ripple && bash quickstart.sh
```

Do **not** zip or sync the entire working tree if it contains `.venv/` — that alone adds hundreds of MB and is recreated by the installer.

## 🚀 Quick Start

You do **not** need to install Java, Maven, or Conda first. On the first run, quickstart looks for them on your PATH, then downloads a portable JDK and Maven into `tools/` and Miniconda into your home folder if they are missing (one `Y/n` prompt, no admin/sudo).

### macOS

```bash
git clone https://github.com/Le0nZim/ripple.git
cd ripple
bash quickstart.sh
```

If macOS Gatekeeper quarantines the folder, the script offers to clear it (`xattr -cr`). After setup, double-click `RIPPLE.command` or run `./RIPPLE.sh`.

### Linux (including WSL)

```bash
git clone https://github.com/Le0nZim/ripple.git
cd ripple
bash quickstart.sh
```

GPU mode is offered only when `nvidia-smi` works. After setup, use `./RIPPLE.sh` or the desktop / applications-menu shortcut.

### Windows

1. Download or clone the repository.
2. Double-click `quickstart.bat` (or run it from Command Prompt).
3. If SmartScreen appears, choose **More info → Run anyway**.

After setup, double-click `RIPPLE.bat` or the Desktop shortcut.

### Optional flags

```bash
bash quickstart.sh --help
bash quickstart.sh --yes --cpu --no-launch   # non-interactive setup
bash quickstart.sh --check                   # doctor: report tools only
```

Windows accepts the same flags: `quickstart.bat --yes`, `--cpu`, `--gpu`, `--check`, `--no-launch`.

That's it! The script will:
1. Detect or download JDK 17+, Maven 3.8+, and Miniconda
2. Create a conda environment `ripple-env` (or reuse it)
3. Install Python dependencies (GPU or CPU)
4. Build the Java application
5. Create launchers / shortcuts and start RIPPLE

## 🔄 Updating RIPPLE

If you installed from a `git clone`, use the **Update** button (bottom-right, next to Help) to check [github.com/Le0nZim/ripple](https://github.com/Le0nZim/ripple) for new commits. If an update is available, RIPPLE closes, fast-forwards the repo, rebuilds, and reopens.

Local source edits cancel the update so they are not overwritten. Zip downloads and packaged AppImage/`.deb` installs cannot update in place — clone the repository and run `quickstart` instead.

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
└── scripts/                     # Build, doctor tests, and portable-tool bootstrap
```

## 📋 Requirements

### System Requirements
- **Java**: JDK 17+ (full JDK with `javac`, not a JRE). Quickstart can download a portable Temurin JDK into `tools/jdk`.
- **Conda**: Miniconda or Anaconda. Quickstart can install Miniconda to `~/miniconda3` (or `%USERPROFILE%\miniconda3`).
- **Maven**: 3.8+ (for building from source). Quickstart can download a portable copy into `tools/maven`.

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

### Java, Maven, or Conda not found
Re-run the installer and accept the portable download:

```bash
# macOS / Linux
bash quickstart.sh --yes

# Windows
quickstart.bat --yes
```

If the download fails, check your network, firewall, or proxy, then try again. You can also install the tools yourself (JDK 17+ from [Adoptium](https://adoptium.net/), Maven 3.8+ from [maven.apache.org](https://maven.apache.org/download.cgi), Miniconda from the [Miniconda docs](https://docs.conda.io/en/latest/miniconda.html)) and re-run quickstart.

Doctor mode (no install, no launch):

```bash
bash quickstart.sh --check
quickstart.bat --check
```

### GPU not detected
Ensure NVIDIA drivers are installed and `nvidia-smi` works:

```bash
nvidia-smi  # Should show GPU info
```

On macOS, RIPPLE always uses CPU mode (TrackPy and DIS). RAFT and LocoTrack need an NVIDIA GPU.

## 📄 License

MIT License. See [LICENSE](https://github.com/Le0nZim/ripple/blob/main/LICENCE).

## 🙏 Acknowledgments

- [ImageJ](https://fiji.sc/) - Image analysis platform
- [RAFT](https://github.com/princeton-vl/RAFT) - Optical flow
- [LocoTrack](https://github.com/cvlab-kaist/locotrack) - Point tracking
- [TrackPy](https://soft-matter.github.io/trackpy/v0.7/) - Particle tracking
- [TAP-Vid](https://github.com/google-deepmind/tapnet/blob/main/tapnet/tapvid/README.md) - Track Assist
