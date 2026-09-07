import glob
import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src", "hotwheels_parking"))

from laplacian_detector import predict_file

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "hotwheels_samples")


class LaplacianDetectorTests(unittest.TestCase):
    def test_recognizes_occupied_samples(self):
        for path in glob.glob(os.path.join(DATA_DIR, "occupied", "*.jpg")):
            self.assertTrue(predict_file(path), "expected %s to be classified as occupied" % path)

    def test_recognizes_empty_samples(self):
        for path in glob.glob(os.path.join(DATA_DIR, "empty", "*.jpg")):
            self.assertFalse(predict_file(path), "expected %s to be classified as empty" % path)


if __name__ == "__main__":
    unittest.main()
