import glob
import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src", "hotwheels_parking"))

from detect import HotWheelsDetector
from features import extract_features, read_image
import cv2

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
MODEL_PATH = os.path.join(DATA_DIR, "hotwheels_model.joblib")


class FeaturesTests(unittest.TestCase):
    def test_extract_features_returns_fixed_length_vector(self):
        image = read_image(glob.glob(os.path.join(DATA_DIR, "hotwheels_samples", "occupied", "*.jpg"))[0],
                           cv2.IMREAD_GRAYSCALE)
        features = extract_features(image)
        self.assertEqual(features.shape, (16,))


class HotWheelsDetectorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.detector = HotWheelsDetector(MODEL_PATH)

    def test_recognizes_occupied_samples(self):
        occupied_files = glob.glob(os.path.join(DATA_DIR, "hotwheels_samples", "occupied", "*.jpg"))
        self.assertTrue(occupied_files)
        results = [self.detector.predict_file(f) for f in occupied_files]
        self.assertTrue(any(results), "expected at least one occupied sample to be classified as occupied")

    def test_predict_returns_a_bool(self):
        sample = glob.glob(os.path.join(DATA_DIR, "hotwheels_samples", "empty", "*.jpg"))[0]
        result = self.detector.predict_file(sample)
        self.assertIsInstance(result, bool)


if __name__ == "__main__":
    unittest.main()
