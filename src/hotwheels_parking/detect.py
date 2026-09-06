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


def show_prediction(image_path, is_occupied):
    image = read_image(image_path)
    label = "OCCUPIED" if is_occupied else "EMPTY"
    color = (0, 0, 255) if is_occupied else (0, 200, 0)

    display = cv2.resize(image, None, fx=2, fy=2) if max(image.shape[:2]) < 200 else image.copy()
    cv2.putText(display, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2, cv2.LINE_AA)

    cv2.imshow(os.path.basename(image_path), display)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def parse_args():
    parser = argparse.ArgumentParser(description="Classify a Hot Wheels parking spot as occupied or empty")
    parser.add_argument("image", help="Path to the spot image to classify")
    parser.add_argument("--model", dest="model_path", default=DEFAULT_MODEL_PATH,
                        help="Path to a trained model file")
    parser.add_argument("--show", action="store_true",
                        help="Display the image with the predicted label in a window")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    detector = HotWheelsDetector(args.model_path)
    is_occupied = detector.predict_file(args.image)
    print("occupied" if is_occupied else "empty")

    if args.show:
        show_prediction(args.image, is_occupied)
