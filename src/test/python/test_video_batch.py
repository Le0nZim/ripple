import os
import sys
import unittest

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "main", "python"))
sys.path.insert(0, ROOT)

import video_batch


class VideoBatchHelpersTest(unittest.TestCase):
    def test_resolve_and_key(self):
        self.assertEqual(video_batch.resolve_frame_range({}), (None, None))
        self.assertEqual(video_batch.resolve_frame_range({"frame_start": 10, "frame_end": 20}), (10, 20))
        self.assertEqual(
            video_batch.flow_cache_key("/tmp/v.tif", 10, 20, "dis_ds2"),
            "/tmp/v.tif_dis_ds2#f10-20",
        )

    def test_filename_range_filter(self):
        self.assertTrue(video_batch.filename_matches_range("vid_raft_512x384_f0-839_optical_flow.npz", 0, 839))
        self.assertFalse(video_batch.filename_matches_range("vid_raft_512x384_optical_flow.npz", 0, 839))
        self.assertTrue(video_batch.filename_matches_range("vid_raft_512x384_optical_flow.npz", None, None))
        self.assertFalse(video_batch.filename_matches_range("vid_raft_512x384_f0-839_optical_flow.npz", None, None))

    def test_slice_and_remap(self):
        import numpy as np
        flows = np.arange(20).reshape(10, 1, 1, 2).astype(np.float32)
        sliced = video_batch.slice_flow_to_range(flows, 2, 5)
        self.assertEqual(sliced.shape[0], 3)
        self.assertEqual(video_batch.to_local_frame(12, 10), 2)
        self.assertEqual(video_batch.to_global_frame(2, 10), 12)
        track = {"frames": [{"frame": 0, "x": 1, "y": 2}]}
        video_batch.shift_track_frames(track, 10)
        self.assertEqual(track["frames"][0]["frame"], 10)


if __name__ == "__main__":
    unittest.main()
