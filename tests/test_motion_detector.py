import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src", "real_parking_lot"))

from motion_detector import MotionDetector


class MotionDetectorTests(unittest.TestCase):
    def test_coordinates_extracts_numpy_array(self):
        spot = {"id": 0, "coordinates": [[0, 0], [10, 0], [10, 10], [0, 10]]}
        coordinates = MotionDetector._coordinates(spot)
        self.assertTrue(np.array_equal(coordinates, np.array(spot["coordinates"])))

    def test_same_status(self):
        statuses = [True, False]
        self.assertTrue(MotionDetector.same_status(statuses, 0, True))
        self.assertFalse(MotionDetector.same_status(statuses, 0, False))

    def test_status_changed(self):
        statuses = [True, False]
        self.assertTrue(MotionDetector.status_changed(statuses, 1, True))
        self.assertFalse(MotionDetector.status_changed(statuses, 1, False))

    def test_mask_covers_only_the_spot_polygon(self):
        detector = MotionDetector(video=None, coordinates=[], start_frame=0)
        spot = {"id": 0, "coordinates": [[0, 0], [9, 0], [9, 9], [0, 9]]}
        coordinates = detector._coordinates(spot)
        rect = (0, 0, 20, 20)

        import cv2
        mask = cv2.drawContours(
            np.zeros((rect[3], rect[2]), dtype=np.uint8),
            [coordinates],
            contourIdx=-1,
            color=255,
            thickness=-1,
            lineType=cv2.LINE_8)

        self.assertGreater((mask == 255).sum(), 0)
        self.assertLess((mask == 255).sum(), mask.size)


if __name__ == "__main__":
    unittest.main()
