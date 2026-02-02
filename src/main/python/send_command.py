#!/usr/bin/env python3
"""Send commands to the Ripple tracking server via TCP or Unix sockets.

This script is a simple wrapper that takes command-line arguments,
constructs a JSON request, and sends it to the server.

Usage:
    python send_command.py <command> [--option value ...]

Examples:
    python send_command.py ping
    python send_command.py compute_flow --tiff video.tif --output-dir ./output
    python send_command.py track_seed --tiff video.tif --seed-x 100 --seed-y 200 --seed-frame 0
"""

import argparse
import json
import os
import socket
import sys

# Default connection settings
DEFAULT_TCP_HOST = "127.0.0.1"
DEFAULT_TCP_PORT = 9876
DEFAULT_SOCKET_PATH = "/tmp/ripple-env.sock"


def send_tcp_command(host: str, port: int, command: dict) -> dict:
    """Send command to server via TCP socket."""
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.settimeout(600)  # 10 minute timeout for long operations
    
    try:
        sock.connect((host, port))
        # Send JSON with trailing newline (protocol delimiter)
        sock.sendall(json.dumps(command).encode('utf-8') + b"\n")
        sock.shutdown(socket.SHUT_WR)
        
        # Read response
        response_data = b""
        while True:
            chunk = sock.recv(4096)
            if not chunk:
                break
            response_data += chunk
        
        if response_data:
            return json.loads(response_data.decode('utf-8'))
        return {"status": "error", "message": "No response from server"}
    finally:
        sock.close()


def send_unix_command(socket_path: str, command: dict) -> dict:
    """Send command to server via Unix socket."""
    sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    sock.settimeout(600)  # 10 minute timeout
    
    try:
        sock.connect(socket_path)
        # Send JSON with trailing newline (protocol delimiter)
        sock.sendall(json.dumps(command).encode('utf-8') + b"\n")
        sock.shutdown(socket.SHUT_WR)
        
        # Read response
        response_data = b""
        while True:
            chunk = sock.recv(4096)
            if not chunk:
                break
            response_data += chunk
        
        if response_data:
            return json.loads(response_data.decode('utf-8'))
        return {"status": "error", "message": "No response from server"}
    finally:
        sock.close()


def _env_truthy(value: str) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "y", "on"}


def send_command(
    command: dict,
    use_tcp: bool = None,
    tcp_host: str = None,
    tcp_port: int = None,
    socket_path: str = None,
) -> dict:
    """Send command to server, choosing TCP or Unix socket.

    Auto-selection order (most explicit wins):
    1) Function arg `use_tcp`
    2) Env `RIPPLE_TRANSPORT` in {tcp, unix}
    3) Env `RIPPLE_USE_TCP` truthy -> TCP
    4) Presence of `SOCKET_HOST`/`SOCKET_PORT` -> TCP
    5) Presence of `SOCKET_PATH` -> Unix
    6) Platform default (Windows -> TCP; others -> Unix)
    """

    if use_tcp is None:
        transport = os.environ.get("RIPPLE_TRANSPORT", "").strip().lower()
        if transport in {"tcp", "inet"}:
            use_tcp = True
        elif transport in {"unix", "uds", "socket"}:
            use_tcp = False
        elif _env_truthy(os.environ.get("RIPPLE_USE_TCP", "")):
            use_tcp = True
        elif os.environ.get("SOCKET_HOST") is not None or os.environ.get("SOCKET_PORT") is not None:
            use_tcp = True
        elif os.environ.get("SOCKET_PATH") is not None:
            use_tcp = False
        else:
            use_tcp = sys.platform == "win32"

    if use_tcp:
        host = tcp_host or os.environ.get("SOCKET_HOST", DEFAULT_TCP_HOST)
        port = tcp_port or int(os.environ.get("SOCKET_PORT", DEFAULT_TCP_PORT))
        return send_tcp_command(host, port, command)

    path = socket_path or os.environ.get("SOCKET_PATH", DEFAULT_SOCKET_PATH)
    return send_unix_command(path, command)


def parse_args():
    """Parse command-line arguments into a command dictionary.

    Supports optional connection flags (consumed locally, not sent to server):
      --tcp / --unix
      --tcp-host HOST
      --tcp-port PORT
      --socket-path PATH
    """
    if len(sys.argv) < 2:
        print("Usage: send_command.py <command> [--option value ...]", file=sys.stderr)
        print("Commands: ping, stop, compute_flow, track_seed, track_anchors, etc.", file=sys.stderr)
        sys.exit(1)

    command_name = sys.argv[1]
    
    # Command name translation: Java uses different command names than server
    command_translation = {
        "track_seed": "propagate_track",
        "track_anchors": "optimize_track",  # singular - uses anchors parameter
    }
    command_name = command_translation.get(command_name, command_name)
    
    # Build command dict from remaining arguments
    command = {"command": command_name}

    # Connection overrides (not part of the command payload)
    conn = {
        "use_tcp": None,
        "tcp_host": None,
        "tcp_port": None,
        "socket_path": None,
    }
    
    # Parameter name translation: CLI argument name -> server expected name
    # The Java code uses --tiff but the Python server expects video_path
    param_translation = {
        "tiff": "video_path",
        "output_dir": "output_path",
        "flow_method": "method",
        "force_recompute": "force_recompute",
        "working_width": "target_width",
        "working_height": "target_height",
        "dis_downsample_factor": "downsample_factor",
        "flow_incremental_allocation": "incremental_allocation",  # General flow setting
    }
    
    # Parse --key value pairs
    args = sys.argv[2:]
    i = 0
    while i < len(args):
        arg = args[i]
        if arg.startswith("--"):
            key = arg[2:].replace("-", "_")  # Convert --output-dir to output_dir

            # Connection flags (consume and do not add to payload)
            if key in {"tcp", "use_tcp"}:
                conn["use_tcp"] = True
                i += 1
                continue
            if key in {"unix", "use_unix"}:
                conn["use_tcp"] = False
                i += 1
                continue
            if key == "tcp_host":
                if i + 1 >= len(args):
                    print("[ERROR] --tcp-host requires a value", file=sys.stderr)
                    sys.exit(1)
                conn["tcp_host"] = args[i + 1]
                i += 2
                continue
            if key == "tcp_port":
                if i + 1 >= len(args):
                    print("[ERROR] --tcp-port requires a value", file=sys.stderr)
                    sys.exit(1)
                try:
                    conn["tcp_port"] = int(args[i + 1])
                except ValueError:
                    print("[ERROR] --tcp-port must be an integer", file=sys.stderr)
                    sys.exit(1)
                i += 2
                continue
            if key in {"socket_path", "socket"}:
                if i + 1 >= len(args):
                    print("[ERROR] --socket-path requires a value", file=sys.stderr)
                    sys.exit(1)
                conn["socket_path"] = args[i + 1]
                i += 2
                continue

            # Apply parameter translation
            key = param_translation.get(key, key)
            if i + 1 < len(args) and not args[i + 1].startswith("--"):
                value = args[i + 1]
                # Try to convert to appropriate type
                if value.lower() == "true":
                    value = True
                elif value.lower() == "false":
                    value = False
                else:
                    try:
                        # Try int first
                        value = int(value)
                    except ValueError:
                        try:
                            # Then float
                            value = float(value)
                        except ValueError:
                            pass  # Keep as string
                command[key] = value
                i += 2
            else:
                # Flag without value, treat as boolean True
                command[key] = True
                i += 1
        else:
            i += 1
    
    # Post-processing: route compute_flow to the correct handler based on method
    # The server has separate handlers: compute_flow (RAFT), compute_dis_flow, 
    # compute_locotrack_flow, compute_trackpy_flow
    if command.get("command") == "compute_flow":
        method = command.get("method", "raft")
        if method == "dis":
            command["command"] = "compute_dis_flow"
        elif method == "locotrack":
            command["command"] = "compute_locotrack_flow"
        elif method == "trackpy":
            command["command"] = "compute_trackpy_flow"
        # else: keep as compute_flow for RAFT

    # Post-processing: allow load_flow to work with either:
    #  1) explicit --flow-path
    #  2) an output directory + video name + method (choose newest matching cache file)
    if command.get("command") == "load_flow" and "flow_path" not in command:
        output_path = command.get("output_path")
        video_path = command.get("video_path")
        video_name = command.get("video_name")
        method = str(command.get("method", "raft"))
        target_w = command.get("target_width")
        target_h = command.get("target_height")

        # Derive a base name if not provided
        if not video_name and isinstance(video_path, str) and video_path:
            video_name = os.path.splitext(os.path.basename(video_path))[0]

        if not output_path or not video_name:
            raise ValueError("load_flow requires --flow-path OR (--output-dir and --video-name/--tiff)")

        if not os.path.isdir(output_path):
            raise ValueError(f"--output-dir must be a directory for load_flow inference: {output_path}")

        prefix = f"{video_name}_{method}"
        suffix = "_optical_flow.npz"

        candidates = []
        try:
            for name in os.listdir(output_path):
                if not (name.startswith(prefix) and name.endswith(suffix)):
                    continue
                full = os.path.join(output_path, name)
                if not os.path.isfile(full):
                    continue
                candidates.append(full)
        except OSError as e:
            raise ValueError(f"Failed to scan output dir for cached flows: {e}")

        if target_w and target_h:
            # Prefer exact resolution-tagged caches first: _{W}x{H}_
            tag = f"_{int(target_w)}x{int(target_h)}_"
            tagged = [p for p in candidates if tag in os.path.basename(p)]
            if tagged:
                candidates = tagged

        if not candidates:
            raise ValueError(
                f"No cached flow files found for method={method} in {output_path}. "
                "Compute flow first or pass --flow-path explicitly."
            )

        # Newest file wins
        candidates.sort(key=lambda p: os.path.getmtime(p), reverse=True)
        command["flow_path"] = candidates[0]
    
    # Post-processing: if 'anchors' is a file path, read and parse the JSON file
    # The server expects anchors to be actual JSON data, not a file path
    if "anchors" in command:
        anchors_value = command["anchors"]
        if isinstance(anchors_value, str) and os.path.isfile(anchors_value):
            try:
                with open(anchors_value, 'r', encoding='utf-8') as f:
                    anchors_data = json.load(f)
                command["anchors"] = anchors_data
                print(f"[DEBUG] Loaded anchors from file: {anchors_value}", file=sys.stderr)
                print(f"[DEBUG] Anchors count: {len(anchors_data) if isinstance(anchors_data, list) else 'N/A'}", file=sys.stderr)
            except Exception as e:
                print(f"[ERROR] Failed to read anchors file: {e}", file=sys.stderr)

    # Post-processing: optimize_tracks can accept a bundle file created by the Java UI.
    # On Linux, run_persistent_tracking.sh expands this bundle into a JSON request with a
    # top-level 'tracks' field. On Windows, run_persistent_tracking.bat forwards args
    # directly to this script, so we must expand the bundle here.
    if "anchors_bundle" in command:
        bundle_path = command.get("anchors_bundle")
        if isinstance(bundle_path, str) and os.path.isfile(bundle_path):
            try:
                with open(bundle_path, 'r', encoding='utf-8') as f:
                    bundle_data = json.load(f)

                if isinstance(bundle_data, dict) and "tracks" in bundle_data:
                    command["tracks"] = bundle_data.get("tracks") or []
                elif isinstance(bundle_data, list):
                    # Allow bundle file to be a raw list of tracks specs
                    command["tracks"] = bundle_data
                else:
                    raise ValueError("anchors bundle JSON must be an object with a 'tracks' array or a raw tracks array")

                # Do not send the file path to the server; send the expanded tracks list.
                del command["anchors_bundle"]
                print(f"[DEBUG] Loaded anchors bundle from file: {bundle_path}", file=sys.stderr)
                print(f"[DEBUG] Bundle tracks: {len(command.get('tracks', []))}", file=sys.stderr)
            except Exception as e:
                print(f"[ERROR] Failed to read anchors bundle file: {e}", file=sys.stderr)
    
    return command, conn


def main():
    command, conn = parse_args()
    
    try:
        response = send_command(
            command,
            use_tcp=conn.get("use_tcp"),
            tcp_host=conn.get("tcp_host"),
            tcp_port=conn.get("tcp_port"),
            socket_path=conn.get("socket_path"),
        )
        
        # Print response as JSON
        print(json.dumps(response, indent=2))
        
        # Exit with appropriate code
        if response.get("status") == "ok":
            sys.exit(0)
        else:
            sys.exit(1)
            
    except ConnectionRefusedError:
        print(json.dumps({
            "status": "error", 
            "message": "Connection refused. Is the server running?"
        }), file=sys.stderr)
        sys.exit(1)
    except socket.timeout:
        print(json.dumps({
            "status": "error",
            "message": "Connection timed out"
        }), file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(json.dumps({
            "status": "error",
            "message": str(e)
        }), file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
