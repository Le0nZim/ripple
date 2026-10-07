"""Unit tests for tracking parameter validation helpers."""

from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "main" / "python"))
from tracking_server import _normalize_blob_search_radius as normalize_blob_search_radius


class TrackingParameterValidationTest(unittest.TestCase):
    def test_blob_search_radius_below_five(self):
        self.assertEqual(2, normalize_blob_search_radius(2))
        self.assertEqual(1, normalize_blob_search_radius(1))
        self.assertEqual(4, normalize_blob_search_radius(4.4))

    def test_blob_search_radius_rejects_invalid(self):
        self.assertEqual(15, normalize_blob_search_radius(0))
        self.assertEqual(15, normalize_blob_search_radius(-1))
        self.assertEqual(15, normalize_blob_search_radius("bad"))


if __name__ == "__main__":
    unittest.main()
