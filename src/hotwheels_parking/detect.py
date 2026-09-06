import argparse
import os

import cv2
import joblib

from features import extract_features, read_image

DEFAULT_MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "..", "data", "hotwheels_model.joblib")


class HotWheelsDetector:
    """Classifies a single cropped parking-spot photo as occupied or empty."""

    def __init__(self, model_path=DEFAULT_MODEL_PATH):
        self.classifier = joblib.load(model_path)

    def predict(self, image):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
        features = extract_features(gray).reshape(1, -1)
        return bool(self.classifier.predict(features)[0])

    def predict_file(self, image_path):
        return self.predict(read_image(image_path))


def parse_args():
    parser = argparse.ArgumentParser(description="Classify a Hot Wheels parking spot as occupied or empty")
    parser.add_argument("image", help="Path to the spot image to classify")
    parser.add_argument("--model", dest="model_path", default=DEFAULT_MODEL_PATH,
                        help="Path to a trained model file")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    detector = HotWheelsDetector(args.model_path)
    is_occupied = detector.predict_file(args.image)
    print("occupied" if is_occupied else "empty")
