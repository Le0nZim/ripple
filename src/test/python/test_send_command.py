import contextlib
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "main" / "python"))
import send_command


class SendCommandTest(unittest.TestCase):
    def test_expands_unicode_batch_and_preserves_clip_range_and_output_path(self):
        with tempfile.TemporaryDirectory() as directory:
            bundle = Path(directory) / "anchors α.json"
            tracks = [{"track_id": "neuron α", "anchors": [{"frame": 10, "x": 8, "y": 9}]}]
            bundle.write_text(json.dumps({"tracks": tracks}, ensure_ascii=False), encoding="utf-8")
            output = str(Path(directory) / "batch result.json")
            argv = ["send_command.py", "optimize_tracks", "--anchors-bundle", str(bundle),
                    "--tiff", "neurons_compressed.tif", "--video-name", "neurons",
                    "--output-dir", output, "--frame-start", "10", "--frame-end", "13"]
            with patch.object(sys, "argv", argv):
                request, _ = send_command.parse_args()
            self.assertEqual(tracks, request["tracks"])
            self.assertEqual(output, request["output_path"])
            self.assertEqual((10, 13), (request["frame_start"], request["frame_end"]))
            self.assertNotIn("anchors_bundle", request)

    def test_invalid_bundle_reports_local_error_without_contacting_server(self):
        with tempfile.TemporaryDirectory() as directory:
            bundle = Path(directory) / "bundle.json"
            for contents in (None, "{bad json", '{"tracks": "wrong type"}', '{"tracks": []}'):
                with self.subTest(contents=contents):
                    if contents is not None:
                        bundle.write_text(contents)
                    stderr = io.StringIO()
                    with patch.object(sys, "argv", ["send_command.py", "optimize_tracks", "--anchors-bundle", str(bundle)]), \
                            patch.object(send_command, "send_command") as send, contextlib.redirect_stderr(stderr):
                        with self.assertRaises(SystemExit) as raised:
                            send_command.main()
                    self.assertEqual(1, raised.exception.code)
                    send.assert_not_called()
                    self.assertEqual("error", json.loads(stderr.getvalue())["status"])


if __name__ == "__main__":
    unittest.main()
