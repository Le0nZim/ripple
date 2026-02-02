#!/usr/bin/env python3
"""
RIPPLE Cross-Platform Launcher

This script provides a universal entry point for RIPPLE that works on both
Windows and Linux. It handles:
1. Python environment detection/setup
2. GPU/CPU mode detection
3. Java application launch
4. Python backend server management

Usage:
    python ripple_launcher.py              # Auto-detect mode
    python ripple_launcher.py --cpu        # Force CPU mode
    python ripple_launcher.py --gpu        # Force GPU mode
    python ripple_launcher.py --check      # System diagnostics only
    python ripple_launcher.py --setup      # First-time setup
"""

import os
import sys
import platform
import subprocess
import shutil
import json
import argparse
import signal
import time
import tempfile
from pathlib import Path
from typing import Optional, List, Dict, Any


# =============================================================================
# CONFIGURATION
# =============================================================================

class Config:
    """Application configuration."""
    
    # Application info
    APP_NAME = "RIPPLE"
    APP_VERSION = "1.0.0"
    
    # Directory structure
    # This script is in src/main/python/, project root is 3 levels up
    SCRIPT_DIR = Path(__file__).parent.resolve()  # src/main/python
    PROJECT_ROOT = SCRIPT_DIR.parent.parent.parent  # project root
    
    # For development
    JAVA_SRC_DIR = PROJECT_ROOT / "src" / "main" / "java"
    PYTHON_RUNTIME_DIR = SCRIPT_DIR  # src/main/python
    REQUIREMENTS_DIR = PROJECT_ROOT / "requirements"
    
    # For packaged app
    BUNDLED_JRE_DIR = PROJECT_ROOT / "runtime"
    BUNDLED_PYTHON_DIR = PROJECT_ROOT / "python"
    # LocoTrack weights directory (canonical location)
    LOCOTRACK_WEIGHTS_DIR = PROJECT_ROOT / "locotrack_pytorch" / "weights"
    TARGET_DIR = PROJECT_ROOT / "target"
    
    # Java settings
    MAIN_CLASS = "com.ripple.VideoAnnotationTool"
    JAR_NAME = "ripple.jar"
    
    # Python settings
    CONDA_ENV_NAME = "ripple-env"  # The conda environment for RIPPLE
    
    # Model files
    REQUIRED_MODELS = {
        "locotrack-base": {
            "filename": "locotrack_base.ckpt",
            "url": "https://huggingface.co/spaces/locotrack/locotrack_demo/resolve/main/checkpoints/locotrack_base.ckpt",
            "size_mb": 85,
            "subdir": "locotrack_pytorch/weights"
        },
        "locotrack-small": {
            "filename": "locotrack_small.ckpt",
            "url": "https://huggingface.co/datasets/hamacojr/LocoTrack-pytorch-weights/resolve/main/locotrack_small.ckpt",
            "size_mb": 40,
            "subdir": "locotrack_pytorch/weights"
        }
    }


# =============================================================================
# PLATFORM UTILITIES
# =============================================================================

def is_windows() -> bool:
    return platform.system() == "Windows"


def is_linux() -> bool:
    return platform.system() == "Linux"


def is_macos() -> bool:
    return platform.system() == "Darwin"


def get_platform_name() -> str:
    if is_windows():
        return "windows"
    elif is_linux():
        return "linux"
    elif is_macos():
        return "macos"
    return "unknown"


def find_executable(name: str, additional_paths: List[Path] = None) -> Optional[Path]:
    """Find an executable in PATH or additional locations."""
    # Add .exe suffix on Windows
    if is_windows() and not name.endswith('.exe'):
        names = [name + '.exe', name + '.bat', name + '.cmd', name]
    else:
        names = [name]
    
    # Check additional paths first
    if additional_paths:
        for path in additional_paths:
            for n in names:
                exe_path = path / n
                if exe_path.exists() and os.access(exe_path, os.X_OK):
                    return exe_path
    
    # Check PATH
    for n in names:
        result = shutil.which(n)
        if result:
            return Path(result)
    
    return None


# =============================================================================
# PYTHON ENVIRONMENT
# =============================================================================

class PythonEnvironment:
    """Manages Python environment detection and setup."""
    
    def __init__(self):
        self.python_path: Optional[Path] = None
        self.pip_path: Optional[Path] = None
        self.env_type: str = "unknown"  # bundled, conda, system
        self.env_path: Optional[Path] = None
    
    def detect(self) -> bool:
        """Detect the Python environment to use.
        
        Priority order:
        1. Bundled Python (for packaged app)
        2. Conda environment (ripple-env) - preferred for development
        3. System Python (fallback)
        """
        
        # 1. Check for bundled Python (packaged app)
        if self._check_bundled():
            return True
        
        # 2. Check for conda environment (ripple-env)
        if self._check_conda():
            return True
        
        # 3. Fall back to system Python
        if self._check_system():
            return True
        
        return False
    
    def _check_bundled(self) -> bool:
        """Check for bundled Python in packaged app."""
        bundled_dir = Config.BUNDLED_PYTHON_DIR
        
        if is_windows():
            python_exe = bundled_dir / "python.exe"
        else:
            python_exe = bundled_dir / "bin" / "python3"
        
        if python_exe.exists():
            self.python_path = python_exe
            self.pip_path = python_exe.parent / ("pip.exe" if is_windows() else "pip")
            self.env_type = "bundled"
            self.env_path = bundled_dir
            return True
        
        return False
    
    def _check_conda(self) -> bool:
        """Check for conda environment - accepts any activated conda env or known RIPPLE envs."""
        # First check if we're already in a conda environment (any name)
        conda_prefix = os.environ.get("CONDA_PREFIX")
        if conda_prefix:
            conda_path = Path(conda_prefix)
            env_name = conda_path.name
            # Skip base environment, but accept any other activated conda env
            if env_name != "base":
                if is_windows():
                    python_exe = conda_path / "python.exe"
                else:
                    python_exe = conda_path / "bin" / "python"
                
                if python_exe.exists():
                    self.python_path = python_exe
                    self.pip_path = python_exe.parent / ("pip.exe" if is_windows() else "pip")
                    self.env_type = "conda"
                    self.env_path = conda_path
                    return True
        
        # Check for known RIPPLE environments in common Anaconda locations
        known_envs = [Config.CONDA_ENV_NAME, "ripple-cpu"]
        anaconda_bases = [
            Path.home() / "anaconda3" / "envs",
            Path.home() / "miniconda3" / "envs",
            Path("/opt/anaconda3/envs"),
            Path("/opt/miniconda3/envs"),
        ]
        
        for base in anaconda_bases:
            for env_name in known_envs:
                conda_path = base / env_name
                if is_windows():
                    python_exe = conda_path / "python.exe"
                else:
                    python_exe = conda_path / "bin" / "python"
                
                if python_exe.exists():
                    self.python_path = python_exe
                    self.pip_path = python_exe.parent / ("pip.exe" if is_windows() else "pip")
                    self.env_type = "conda"
                    self.env_path = conda_path
                    return True
        
        return False
    
    def _check_system(self) -> bool:
        """Fall back to system Python."""
        python_exe = find_executable("python3") or find_executable("python")
        
        if python_exe:
            self.python_path = python_exe
            self.pip_path = find_executable("pip3") or find_executable("pip")
            self.env_type = "system"
            return True
        
        return False
    
    def get_info(self) -> Dict[str, Any]:
        """Get environment information."""
        return {
            "python_path": str(self.python_path) if self.python_path else None,
            "pip_path": str(self.pip_path) if self.pip_path else None,
            "env_type": self.env_type,
            "env_path": str(self.env_path) if self.env_path else None
        }
    
    def run_python(self, args: List[str], **kwargs) -> subprocess.CompletedProcess:
        """Run a Python command."""
        if not self.python_path:
            raise RuntimeError("Python environment not detected")
        
        cmd = [str(self.python_path)] + args
        return subprocess.run(cmd, **kwargs)
    
    def check_package(self, package: str) -> bool:
        """Check if a package is installed."""
        result = self.run_python(
            ["-c", f"import {package}"],
            capture_output=True
        )
        return result.returncode == 0


# =============================================================================
# JAVA ENVIRONMENT
# =============================================================================

class JavaEnvironment:
    """Manages Java environment detection."""
    
    def __init__(self):
        self.java_path: Optional[Path] = None
        self.java_version: Optional[str] = None
        self.env_type: str = "unknown"  # bundled, system
    
    def detect(self) -> bool:
        """Detect Java installation."""
        
        # 1. Check for bundled JRE
        if self._check_bundled():
            return True
        
        # 2. Check JAVA_HOME
        if self._check_java_home():
            return True
        
        # 3. Check system PATH
        if self._check_system():
            return True
        
        return False
    
    def _check_bundled(self) -> bool:
        """Check for bundled JRE."""
        bundled_dir = Config.BUNDLED_JRE_DIR
        
        if is_windows():
            java_exe = bundled_dir / "bin" / "java.exe"
        else:
            java_exe = bundled_dir / "bin" / "java"
        
        if java_exe.exists():
            self.java_path = java_exe
            self.env_type = "bundled"
            self._get_version()
            return True
        
        return False
    
    def _check_java_home(self) -> bool:
        """Check JAVA_HOME environment variable."""
        java_home = os.environ.get("JAVA_HOME")
        if java_home:
            java_home_path = Path(java_home)
            if is_windows():
                java_exe = java_home_path / "bin" / "java.exe"
            else:
                java_exe = java_home_path / "bin" / "java"
            
            if java_exe.exists():
                self.java_path = java_exe
                self.env_type = "java_home"
                self._get_version()
                return True
        
        return False
    
    def _check_system(self) -> bool:
        """Check system PATH for Java."""
        java_exe = find_executable("java")
        if java_exe:
            self.java_path = java_exe
            self.env_type = "system"
            self._get_version()
            return True
        
        return False
    
    def _get_version(self) -> None:
        """Get Java version."""
        try:
            result = subprocess.run(
                [str(self.java_path), "-version"],
                capture_output=True,
                text=True
            )
            # Java version is printed to stderr
            output = result.stderr
            for line in output.split('\n'):
                if 'version' in line.lower():
                    # Extract version string
                    parts = line.split('"')
                    if len(parts) >= 2:
                        self.java_version = parts[1]
                        break
        except:
            pass
    
    def get_info(self) -> Dict[str, Any]:
        """Get environment information."""
        return {
            "java_path": str(self.java_path) if self.java_path else None,
            "java_version": self.java_version,
            "env_type": self.env_type
        }


# =============================================================================
# GPU DETECTION
# =============================================================================

def check_gpu_available(python_env: PythonEnvironment) -> Dict[str, Any]:
    """Check GPU availability using PyTorch."""
    result = python_env.run_python([
        "-c",
        """
import json
import sys

info = {
    "cuda_available": False,
    "gpu_count": 0,
    "gpu_name": None,
    "cuda_version": None,
    "error": None
}

try:
    import torch
    info["cuda_available"] = torch.cuda.is_available()
    if info["cuda_available"]:
        info["gpu_count"] = torch.cuda.device_count()
        info["gpu_name"] = torch.cuda.get_device_name(0)
        info["cuda_version"] = torch.version.cuda
except Exception as e:
    info["error"] = str(e)

print(json.dumps(info))
"""
    ], capture_output=True, text=True)
    
    if result.returncode == 0:
        try:
            return json.loads(result.stdout.strip())
        except:
            pass
    
    return {
        "cuda_available": False,
        "gpu_count": 0,
        "gpu_name": None,
        "cuda_version": None,
        "error": result.stderr if result.stderr else "Failed to detect GPU"
    }


# =============================================================================
# APPLICATION LAUNCHER
# =============================================================================

class RippleLauncher:
    """Main application launcher."""
    
    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.python_env = PythonEnvironment()
        self.java_env = JavaEnvironment()
        self.mode: str = "cpu"
        self.gpu_info: Dict[str, Any] = {}
        self.server_process: Optional[subprocess.Popen] = None
    
    def run(self) -> int:
        """Main entry point."""
        
        # Handle special commands
        if self.args.check:
            return self.run_diagnostics()
        
        if self.args.setup:
            return self.run_setup()
        
        if self.args.download_models:
            return self.download_models()
        
        # Normal launch
        return self.launch_application()
    
    def run_diagnostics(self) -> int:
        """Run system diagnostics."""
        print(f"\n{'=' * 60}")
        print(f"{Config.APP_NAME} System Diagnostics")
        print(f"{'=' * 60}")
        
        print(f"\nPlatform: {platform.system()} {platform.release()}")
        print(f"Architecture: {platform.machine()}")
        
        # Python
        print("\n--- Python Environment ---")
        if self.python_env.detect():
            info = self.python_env.get_info()
            print(f"Type: {info['env_type']}")
            print(f"Path: {info['python_path']}")
            
            # Check GPU
            print("\n--- GPU Detection ---")
            self.gpu_info = check_gpu_available(self.python_env)
            if self.gpu_info["cuda_available"]:
                print(f"CUDA Available: Yes")
                print(f"GPU: {self.gpu_info['gpu_name']}")
                print(f"CUDA Version: {self.gpu_info['cuda_version']}")
            else:
                print(f"CUDA Available: No")
                if self.gpu_info["error"]:
                    print(f"Reason: {self.gpu_info['error']}")
        else:
            print("Python: NOT FOUND")
        
        # Java
        print("\n--- Java Environment ---")
        if self.java_env.detect():
            info = self.java_env.get_info()
            print(f"Type: {info['env_type']}")
            print(f"Path: {info['java_path']}")
            print(f"Version: {info['java_version']}")
        else:
            print("Java: NOT FOUND")
        
        # Models
        print("\n--- Model Files ---")
        for name, info in Config.REQUIRED_MODELS.items():
            model_dir = Config.PROJECT_ROOT / info.get("subdir", "locotrack_pytorch/weights")
            model_path = model_dir / info["filename"]
            status = "✓ Found" if model_path.exists() else "✗ Missing"
            print(f"{name}: {status}")
        
        print(f"\n{'=' * 60}\n")
        return 0
    
    def run_setup(self) -> int:
        """Run first-time setup."""
        print(f"\n{Config.APP_NAME} Setup")
        print("=" * 40)
        
        # Check Python
        if not self.python_env.detect():
            print("\nERROR: Python not found!")
            print("Please install Python 3.10+ and try again.")
            return 1
        
        print(f"\nUsing Python: {self.python_env.python_path}")
        
        # Determine mode
        self.gpu_info = check_gpu_available(self.python_env)
        if self.gpu_info["cuda_available"]:
            print(f"\nGPU detected: {self.gpu_info['gpu_name']}")
            self.mode = "gpu"
        else:
            print("\nNo GPU detected, will install CPU-only packages")
            self.mode = "cpu"
        
        # Install requirements
        print("\nInstalling Python dependencies...")
        req_file = "requirements-gpu.txt" if self.mode == "gpu" else "requirements-cpu.txt"
        req_path = Config.REQUIREMENTS_DIR / req_file
        
        if req_path.exists():
            result = subprocess.run([
                str(self.python_env.pip_path),
                "install", "-r", str(req_path)
            ])
            if result.returncode != 0:
                print("ERROR: Failed to install dependencies")
                return 1
        else:
            print(f"WARNING: {req_file} not found")
        
        print("\nSetup complete!")
        return 0
    
    def download_models(self) -> int:
        """Download required model files."""
        print(f"\n{Config.APP_NAME} Model Download")
        print("=" * 40)
        
        for name, info in Config.REQUIRED_MODELS.items():
            model_dir = Config.PROJECT_ROOT / info.get("subdir", "locotrack_pytorch/weights")
            model_dir.mkdir(parents=True, exist_ok=True)
            model_path = model_dir / info["filename"]
            
            if model_path.exists():
                print(f"\n{name}: Already downloaded")
                continue
            
            print(f"\nDownloading {name} ({info['size_mb']} MB)...")
            
            try:
                import urllib.request
                urllib.request.urlretrieve(info["url"], model_path)
                print(f"  Saved to: {model_path}")
            except Exception as e:
                print(f"  ERROR: {e}")
        
        print("\nModel download complete!")
        return 0
    
    def launch_application(self) -> int:
        """Launch the main application."""
        
        # Detect environments
        if not self.python_env.detect():
            print("ERROR: Python not found. Run with --setup first.")
            return 1
        
        if not self.java_env.detect():
            print("ERROR: Java not found. Please install Java 11+ or use the bundled version.")
            return 1
        
        # Determine mode
        if self.args.cpu:
            self.mode = "cpu"
        elif self.args.gpu:
            self.gpu_info = check_gpu_available(self.python_env)
            if not self.gpu_info["cuda_available"]:
                print("WARNING: GPU mode requested but no GPU available. Using CPU mode.")
                self.mode = "cpu"
            else:
                self.mode = "gpu"
        else:
            # Auto-detect
            self.gpu_info = check_gpu_available(self.python_env)
            self.mode = "gpu" if self.gpu_info["cuda_available"] else "cpu"
        
        print(f"{Config.APP_NAME} starting in {self.mode.upper()} mode...")
        
        # Set environment variables
        env = os.environ.copy()
        env["RIPPLE_MODE"] = self.mode
        env["RIPPLE_PYTHON"] = str(self.python_env.python_path)
        env["RIPPLE_RUNTIME_DIR"] = str(Config.PYTHON_RUNTIME_DIR)
        
        # Start Python tracking server
        self._start_tracking_server(env)
        
        # Start Java application
        try:
            jar_path = Config.TARGET_DIR / Config.JAR_NAME
            if not jar_path.exists():
                # Try alternative locations
                jar_path = Config.PROJECT_ROOT / Config.JAR_NAME
            
            if jar_path.exists():
                # Run from JAR
                java_cmd = [
                    str(self.java_env.java_path),
                    "-jar", str(jar_path)
                ]
            else:
                # Development mode - compile and run
                print("JAR not found, running in development mode...")
                # This would need classpath setup
                return self._run_development_mode(env)
            
            result = subprocess.run(java_cmd, env=env)
            return result.returncode
            
        finally:
            self._stop_tracking_server()
    
    def _start_tracking_server(self, env: Dict[str, str]) -> None:
        """Start the Python tracking server in background."""
        server_script = Config.PYTHON_RUNTIME_DIR / "tracking_server.py"
        
        if not server_script.exists():
            print(f"WARNING: Tracking server not found at {server_script}")
            return
        
        print("Starting tracking server...")
        
        # Prepare server command
        cmd = [
            str(self.python_env.python_path),
            str(server_script),
            "--mode", self.mode
        ]
        
        # Start in background
        if is_windows():
            # Windows: Use CREATE_NEW_PROCESS_GROUP
            self.server_process = subprocess.Popen(
                cmd,
                env=env,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                creationflags=subprocess.CREATE_NEW_PROCESS_GROUP
            )
        else:
            # Linux: Use process group
            self.server_process = subprocess.Popen(
                cmd,
                env=env,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                start_new_session=True
            )
        
        # Wait a bit for server to start
        time.sleep(2)
        
        if self.server_process.poll() is not None:
            print("WARNING: Tracking server failed to start")
            self.server_process = None
    
    def _stop_tracking_server(self) -> None:
        """Stop the Python tracking server."""
        if self.server_process:
            print("Stopping tracking server...")
            try:
                if is_windows():
                    self.server_process.send_signal(signal.CTRL_BREAK_EVENT)
                else:
                    os.killpg(os.getpgid(self.server_process.pid), signal.SIGTERM)
                
                self.server_process.wait(timeout=5)
            except:
                self.server_process.kill()
    
    def _run_development_mode(self, env: Dict[str, str]) -> int:
        """Run in development mode (compile and run Java)."""
        print("Development mode not fully implemented.")
        print("Please run 'mvn package' first to create the JAR file.")
        return 1


# =============================================================================
# MAIN ENTRY POINT
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description=f"{Config.APP_NAME} - Cross-Platform Launcher",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python ripple_launcher.py              # Auto-detect GPU/CPU and launch
  python ripple_launcher.py --cpu        # Force CPU mode
  python ripple_launcher.py --gpu        # Force GPU mode (fails if no GPU)
  python ripple_launcher.py --check      # System diagnostics
  python ripple_launcher.py --setup      # First-time setup
        """
    )
    
    mode_group = parser.add_mutually_exclusive_group()
    mode_group.add_argument(
        "--cpu", action="store_true",
        help="Force CPU mode (no GPU acceleration)"
    )
    mode_group.add_argument(
        "--gpu", action="store_true",
        help="Force GPU mode (requires NVIDIA GPU)"
    )
    
    parser.add_argument(
        "--check", action="store_true",
        help="Run system diagnostics and exit"
    )
    parser.add_argument(
        "--setup", action="store_true",
        help="Run first-time setup (install dependencies)"
    )
    parser.add_argument(
        "--download-models", action="store_true",
        help="Download required model files"
    )
    
    args = parser.parse_args()
    
    launcher = RippleLauncher(args)
    sys.exit(launcher.run())


if __name__ == "__main__":
    main()
