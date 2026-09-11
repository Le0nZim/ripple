"""Unit tests for tracking parameter validation helpers."""

import math
import unittest


def normalize_blob_search_radius(value, default=15):
    try:
        radius = float(value)
    except (TypeError, ValueError):
        radius = float(default)
    if not math.isfinite(radius) or radius <= 0:
        radius = float(default)
    return max(1, int(round(radius)))


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
