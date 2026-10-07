"""Regression tests against the real CPU tracking backend (no model downloads)."""

import contextlib
import io
import json
from pathlib import Path
import socket
import subprocess
import sys
import tempfile
import threading
import types
import unittest
from unittest.mock import Mock, patch

import numpy as np
import tifffile

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "main" / "python"))
import tracking_server as backend


class TrackingServerRegressionTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.output = Path(self.temp.name)
        self.video = str(self.output / "neurons.tif")
        with contextlib.redirect_stdout(io.StringIO()):
            self.server = backend.TrackingServer(None, backend.CPUModelManager())
        self.addCleanup(self.server._clear_video_accessor_cache)

    def cache_flow(self, start=10, end=13, dx=1):
        flows = np.zeros((end - start, 32, 32, 2), dtype=np.float32)
        flows[..., 0] = dx
        entry = {"flows_np": flows, "metadata": {
            "method": "dis", "frame_start": start, "frame_end": end,
            "resized_shape": (end - start + 1, 32, 32),
            "original_shape": (20, 32, 32), "flow_to_input_scale": 1,
        }}
        self.server._remember_flow(self.video, entry, frame_start=start, frame_end=end)
        return entry

    def request(self, **kwargs):
        return {"video_path": self.video, "frame_start": 10, "frame_end": 13,
                "output_path": str(self.output / "result.json"), **kwargs}

    def exchange(self, request):
        client, server = socket.socketpair()
        with client:
            client.settimeout(5)
            client.sendall(json.dumps(request).encode("utf-8") + b"\n")
            client.shutdown(socket.SHUT_WR)
            self.server._handle_client(server)
            with client.makefile("r", encoding="utf-8") as response:
                return json.load(response)

    def test_cache_never_substitutes_another_clip_or_full_video(self):
        entry = self.cache_flow()
        self.assertIs(entry, self.server._get_cached_flow(self.video, self.request()))
        self.assertIsNone(self.server._get_cached_flow(
            self.video, {"frame_start": 13, "frame_end": 16}))
        self.assertIsNone(self.server._get_cached_flow(self.video, {}))

    def test_memory_status_counts_shared_flow_only_once(self):
        entry = self.cache_flow()
        self.assertEqual(entry["flows_np"].nbytes,
                         self.server._memory_status({})["flow_cache_bytes"])

    def test_full_video_disk_lookup_rejects_clip_only_cache(self):
        (self.output / "neurons_raft_32x32_f10-13_optical_flow.npz").touch()
        found = self.server._find_existing_flow_file(
            str(self.output / "neurons_raft_optical_flow.npz"), "raft")
        self.assertIsNone(found)

    def test_batch_anchors_keep_global_frame_numbers(self):
        self.cache_flow()
        result = self.server._optimize_tracks(self.request(tracks=[{
            "track_id": "neuron α", "anchors": [
                {"frame": 10, "x": 8, "y": 8},
                {"frame": 13, "x": 11, "y": 8},
            ],
        }]))
        frames = result["tracks"][0]["frames"]
        self.assertEqual(list(range(10, 14)), [p["frame"] for p in frames])
        self.assertEqual(list(range(8, 12)), [p["x"] for p in frames])
        with open(result["output_path"]) as handle:
            self.assertEqual(result["tracks"], json.load(handle)["tracks"])

    def test_batch_seed_uses_clip_local_index(self):
        self.cache_flow()
        result = self.server._optimize_tracks(self.request(tracks=[{
            "track_id": "neuron", "seed": {"frame": 11, "x": 9, "y": 8},
        }]))
        frames = result["tracks"][0]["frames"]
        self.assertEqual(list(range(8, 12)), [p["x"] for p in frames])

    def test_invalid_anchor_reports_track_and_range(self):
        self.cache_flow()
        with self.assertRaisesRegex(ValueError, "neuron.*frame.*9.*10.*13"):
            self.server._optimize_tracks(self.request(tracks=[{
                "track_id": "neuron", "anchors": [{"frame": 9, "x": 8, "y": 8}],
            }]))

    def test_empty_anchors_report_validation_error(self):
        self.cache_flow()
        with self.assertRaisesRegex(ValueError, "anchor"):
            self.server._optimize_track(self.request(anchors=[]))

    def test_blob_accessor_reads_only_requested_clip(self):
        volume = np.stack([np.full((32, 32), i, dtype=np.uint16) for i in range(20)])
        for compression in (None, "deflate"):
            with self.subTest(compression=compression):
                tifffile.imwrite(self.video, volume, photometric="minisblack", compression=compression)
                accessor = self.server._get_or_create_video_accessor(
                    self.video, 16, 16, use_cache=False, frame_start=10, frame_end=13)
                try:
                    self.assertEqual((4, 16, 16), accessor.shape)
                    np.testing.assert_array_equal(accessor.get_patch(0, 0, 2, 0, 2), 10)
                    np.testing.assert_array_equal(accessor.get_patch(3, 0, 2, 0, 2), 13)
                finally:
                    accessor.close()

    def test_full_video_accessor_still_resizes_patches(self):
        tifffile.imwrite(self.video, np.ones((5, 32, 32), np.uint8), photometric="minisblack")
        accessor = self.server._get_or_create_video_accessor(self.video, 16, 16, use_cache=False)
        try:
            self.assertEqual((5, 16, 16), accessor.shape)
            self.assertEqual((16, 16), accessor.get_patch(0, 0, 32, 0, 32).shape)
        finally:
            accessor.close()

    def test_blob_accessor_cache_does_not_reuse_previous_clip(self):
        volume = np.stack([np.full((32, 32), i, dtype=np.uint16) for i in range(20)])
        tifffile.imwrite(self.video, volume, photometric="minisblack")
        first = self.server._get_or_create_video_accessor(
            self.video, 32, 32, frame_start=0, frame_end=10)
        second = self.server._get_or_create_video_accessor(
            self.video, 32, 32, frame_start=10, frame_end=19)
        self.assertIsNot(first, second)
        self.assertIsNone(first._memmap)
        np.testing.assert_array_equal(second.get_patch(0, 0, 2, 0, 2), 10)

    def test_load_legacy_dis_flow_infers_scale_from_height_not_frames(self):
        flow_file = self.output / "legacy.npz"
        np.savez(flow_file, flows=np.zeros((3, 16, 16, 2), np.float32),
                 resized_shape=(4, 32, 32))
        result = self.server._load_flow(self.request(flow_path=str(flow_file), method="dis"))
        self.assertEqual("ok", result["status"])
        self.assertEqual(2, result["flow_to_input_scale"])

    def test_cached_raft_clip_does_not_recompute_or_lose_frame_range(self):
        flow_file = self.output / "neurons_raft_32x32_f10-13_optical_flow.npz"
        np.savez(flow_file, flows=np.zeros((3, 32, 32, 2), np.float32),
                 original_shape=(20, 32, 32), resized_shape=(4, 32, 32), is_avi=False, method="raft")
        with patch.object(self.server, "_compute_flow_from_video", side_effect=AssertionError("Unneeded recompute")):
            result = self.server._compute_flow(self.request(output_path=str(self.output)))
        self.assertEqual("ok", result["status"])
        self.assertEqual((3, 32, 32, 2), self.server._get_cached_flow(self.video, self.request())["flows_np"].shape)

    def test_segment_cache_invalidated_when_flow_changes(self):
        self.cache_flow()
        self.server._get_cached_flow(self.video, self.request())
        key = self.server.segment_cache.make_key(0, 8, 8, 3, 11, 8, None)
        self.server.segment_cache.put(key, np.array([[8, 8], [11, 8]]))
        self.cache_flow(dx=2)
        self.server._get_cached_flow(self.video, self.request())
        self.assertIsNone(self.server.segment_cache.get(key))

    def test_same_filename_in_other_directory_clears_previous_video(self):
        self.cache_flow()
        self.server._current_video_path = self.video
        self.server._clear_flow_cache_for_new_video(str(self.output / "other" / "neurons.tif"))
        self.assertEqual({}, self.server.flow_cache)
        self.assertEqual({}, self.server.video_metadata)

    def test_particle_flow_adapters_keep_clip_metadata_and_restore_selected_cache(self):
        # Exercise adapter/cache plumbing without invoking optional particle models.
        # The actual CPU flow algorithms are exercised separately below.
        volume = np.zeros((20, 32, 32), np.uint8)
        tifffile.imwrite(self.video, volume, photometric="minisblack")
        for method in ("trackpy", "locotrack"):
            with self.subTest(method=method):
                module = types.ModuleType(method + "_flow")
                compute = Mock(return_value=(np.zeros((3, 32, 32, 2), np.float32), {}))
                setattr(module, "compute_" + method + "_optical_flow", compute)
                handler = getattr(self.server, "_compute_" + method + "_flow")
                request = self.request(output_path=None, save_to_disk=False)
                with patch.dict(sys.modules, {method + "_flow": module}), \
                        patch.object(self.server, "_get_locotrack_manager", return_value=Mock()):
                    response = handler(request)
                    self.assertEqual("ok", response["status"], response)
                    entry = self.server._get_cached_flow(self.video, request)
                    self.assertIsNotNone(entry)
                    self.assertEqual((10, 13), (entry["metadata"]["frame_start"], entry["metadata"]["frame_end"]))
                    self.cache_flow(dx=2)  # Simulate choosing another method, then reselecting this one.
                    handler(request)
                    selected = self.server._get_cached_flow(self.video, request)
                    self.assertEqual(method, selected["metadata"]["method"])
                    compute.assert_called_once()

    def test_batch_cli_transport_handles_unicode_and_explicit_temporary_output(self):
        self.cache_flow()
        bundle = self.output / "anchors α.json"
        bundle.write_text(json.dumps({"tracks": [{"track_id": "neuron α", "anchors": [
            {"frame": 10, "x": 8, "y": 8}, {"frame": 13, "x": 11, "y": 8}]}]}, ensure_ascii=False),
            encoding="utf-8")
        output = self.output / "temporary batch result.json"
        saved = self.output / "neurons_annotations.json"
        saved.write_text("saved annotations", encoding="utf-8")
        with socket.socket() as listener:
            listener.bind(("127.0.0.1", 0))
            listener.listen(1)
            listener.settimeout(5)
            thread = threading.Thread(target=lambda: self.server._handle_client(listener.accept()[0]), daemon=True)
            thread.start()
            command = [sys.executable, str(Path(backend.__file__).with_name("send_command.py")),
                "optimize_tracks", "--tcp", "--tcp-host", "127.0.0.1", "--tcp-port", str(listener.getsockname()[1]),
                "--tiff", self.video, "--video-name", "neurons", "--anchors-bundle", str(bundle),
                "--output-dir", str(output), "--frame-start", "10", "--frame-end", "13"]
            proc = subprocess.run(command, capture_output=True, text=True, timeout=10)
            thread.join(timeout=5)
        self.assertEqual(0, proc.returncode, proc.stderr + proc.stdout)
        self.assertEqual("ok", json.loads(proc.stdout)["status"])
        payload = json.loads(output.read_text())
        self.assertEqual("neuron α", payload["tracks"][0]["track_id"])
        self.assertEqual(10, payload["tracks"][0]["frames"][0]["frame"])
        self.assertEqual("saved annotations", saved.read_text())

    def test_blob_accessors_closed_on_failed_single_track_operations(self):
        self.cache_flow()
        for operation in ("propagate", "optimize"):
            with self.subTest(operation=operation):
                accessor = Mock(ndim=3)
                with patch.object(self.server, "_get_or_create_video_accessor", return_value=accessor) as factory:
                    if operation == "propagate":
                        with patch.object(self.server, "_propagate_with_flows_blob_memmap", side_effect=RuntimeError("failed")):
                            with self.assertRaisesRegex(RuntimeError, "failed"):
                                self.server._propagate_track(self.request(
                                    seed_x=8, seed_y=8, seed_frame=11, use_blob_detection=True))
                    else:
                        with patch.object(backend.FlowBlendBlobTrackBuilder, "build_track", side_effect=RuntimeError("failed")):
                            with self.assertRaisesRegex(RuntimeError, "failed"):
                                self.server._optimize_track(self.request(correction_method="blob_assisted",
                                    anchors=[{"frame": 10, "x": 8, "y": 8}]))
                self.assertEqual(10, factory.call_args.kwargs["frame_start"])
                self.assertEqual(13, factory.call_args.kwargs["frame_end"])
                accessor.close.assert_called_once()

    def test_cache_mutations_cannot_run_during_tracking(self):
        entry = self.cache_flow()
        self.server._operation_lock.acquire()
        self.addCleanup(self.server._operation_lock.release)
        self.server._operation_in_progress.set()
        for command in ("clear_cache", "clear_memory", "load_flow", "physics_optimize_global"):
            with self.subTest(command=command):
                result = self.exchange({"command": command})
                self.assertTrue(result.get("busy"), result)
                self.assertIs(entry, self.server.flow_cache[self.video])

    def test_invalid_batch_returns_useful_error_and_releases_busy_flag(self):
        self.cache_flow()
        result = self.exchange(self.request(command="optimize_tracks", tracks=[{
            "track_id": "neuron", "anchors": [{"frame": 50, "x": 8, "y": 8}],
        }]))
        self.assertEqual("error", result["status"])
        self.assertIn("neuron", result["message"])
        self.assertIn("[10, 13]", result["message"])
        self.assertFalse(self.server._operation_in_progress.is_set())
        self.assertFalse(self.server._operation_lock.locked())

    def test_cpu_split_video_flow_and_batch_tracking_over_socket(self):
        yy, xx = np.mgrid[:64, :64]
        volume = np.stack([255 * np.exp(-((xx - 20 - t) ** 2 + (yy - 30) ** 2) / 18)
                           for t in range(9)]).astype(np.uint8)
        tifffile.imwrite(self.video, volume, photometric="minisblack")
        # A saved user export must survive intermediate tracking operations.
        saved = self.output / "neurons_annotations.json"
        saved.write_text("saved user annotations", encoding="utf-8")
        for flow_method in ("compute_dis_flow", "compute_dis_fast_flow"):
            for start, end in ((0, 4), (4, 8)):
                with self.subTest(method=flow_method, start=start):
                    flow = self.exchange({"command": flow_method, "video_path": self.video,
                        "frame_start": start, "frame_end": end, "save_to_disk": False,
                        "downsample_factor": 1, "incremental_allocation": False})
                    self.assertEqual("ok", flow["status"], flow)
                    self.assertEqual([4, 64, 64, 2], flow["shape"])
                    for correction in ("full_blend", "blob_assisted", "corridor_dp"):
                        result = self.exchange({"command": "optimize_tracks", "video_path": self.video,
                            "frame_start": start, "frame_end": end, "correction_method": correction,
                            "tracks": [{"track_id": "neuron α", "anchors": [
                                {"frame": start, "x": 20 + start, "y": 30},
                                {"frame": end, "x": 20 + end, "y": 30}]}]})
                        self.assertEqual("ok", result["status"], result)
                        frames = result["tracks"][0]["frames"]
                        self.assertEqual(list(range(start, end + 1)), [p["frame"] for p in frames])
                        self.assertEqual((20 + start, 30), (frames[0]["x"], frames[0]["y"]))
                        self.assertEqual((20 + end, 30), (frames[-1]["x"], frames[-1]["y"]))
        self.assertEqual("saved user annotations", saved.read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
