import argparse
import os

import cv2
import numpy as np

from features import read_image

# Calibrated by scoring every photo in data/hotwheels_raw and picking the
# threshold that best separates the two classes (see calibrate()) - occupied
# photos average ~4.5, empty ones ~1.6, so 3.06 sits cleanly between them.
DEFAULT_THRESHOLD = 3.06


def laplacian_score(image):
    """Same edge-density measure as the real parking lot detector: a flat,
    empty surface has far less high-frequency detail than one with a car
    (Hot Wheels-sized or real) on it."""
    blurred = cv2.GaussianBlur(image, (5, 5), 3)
    gray = cv2.cvtColor(blurred, cv2.COLOR_BGR2GRAY) if blurred.ndim == 3 else blurred
    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    return float(np.mean(np.abs(laplacian)))


def predict(image, threshold=DEFAULT_THRESHOLD):
    return laplacian_score(image) > threshold


def predict_file(image_path, threshold=DEFAULT_THRESHOLD):
    return predict(read_image(image_path), threshold)


def calibrate(data_dir):
    """Scores every occupied/empty photo in data_dir and returns the
    threshold that classifies the most photos correctly."""
    scores, labels = [], []
    for class_name, label in (("occupied", 1), ("empty", 0)):
        class_dir = os.path.join(data_dir, class_name)
        for filename in os.listdir(class_dir):
            scores.append(laplacian_score(read_image(os.path.join(class_dir, filename))))
            labels.append(label)

    scores = np.array(scores)
    labels = np.array(labels)

    best_accuracy, best_threshold = 0.0, DEFAULT_THRESHOLD
    for candidate in np.linspace(scores.min(), scores.max(), 200):
        accuracy = ((scores > candidate).astype(int) == labels).mean()
        if accuracy > best_accuracy:
            best_accuracy, best_threshold = accuracy, candidate

    return best_threshold, best_accuracy


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
    parser = argparse.ArgumentParser(
        description="Classify a Hot Wheels parking spot using edge-density thresholding "
                    "(the same technique the real parking lot detector uses)")
    parser.add_argument("image", nargs="?", help="Path to the spot image to classify")
    parser.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD,
                        help="Edge-density threshold above which a spot counts as occupied")
    parser.add_argument("--show", action="store_true",
                        help="Display the image with the predicted label in a window")
    parser.add_argument("--calibrate", metavar="DATA_DIR",
                        help="Find the best threshold from a labeled occupied/empty dataset instead of predicting")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    if args.calibrate:
        threshold, accuracy = calibrate(args.calibrate)
        print("Best threshold: %.2f (accuracy on this data: %.1f%%)" % (threshold, accuracy * 100))
    else:
        is_occupied = predict_file(args.image, args.threshold)
        print("occupied" if is_occupied else "empty")
        if args.show:
            show_prediction(args.image, is_occupied)
