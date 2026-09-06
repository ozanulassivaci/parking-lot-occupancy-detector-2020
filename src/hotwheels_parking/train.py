import argparse
import os

import joblib
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC

from features import load_image_as_features

LABEL_OCCUPIED = 1
LABEL_EMPTY = 0

DEFAULT_DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "data")
DEFAULT_MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "..", "data", "hotwheels_model.joblib")


def load_dataset(data_dir):
    features, labels = [], []
    for class_name, label in (("occupied", LABEL_OCCUPIED), ("empty", LABEL_EMPTY)):
        class_dir = os.path.join(data_dir, class_name)
        for filename in os.listdir(class_dir):
            image_path = os.path.join(class_dir, filename)
            try:
                features.append(load_image_as_features(image_path))
                labels.append(label)
            except FileNotFoundError:
                continue
    return np.array(features, dtype=np.float32), np.array(labels, dtype=np.int32)


def train(data_dir, model_path):
    X, y = load_dataset(data_dir)
    if len(set(y.tolist())) < 2:
        raise ValueError("Need images from both the 'occupied' and 'empty' folders to train")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y)

    classifier = LinearSVC(max_iter=5000)
    classifier.fit(X_train, y_train)

    accuracy = classifier.score(X_test, y_test)
    print("Trained on %d images, held-out accuracy: %.1f%%" % (len(y_train), accuracy * 100))

    joblib.dump(classifier, model_path)
    print("Model saved to %s" % model_path)


def parse_args():
    parser = argparse.ArgumentParser(description="Train the Hot Wheels parking-spot occupancy classifier")
    parser.add_argument("--data", dest="data_dir", default=os.path.join(DEFAULT_DATA_DIR, "hotwheels_samples"),
                        help="Directory containing 'occupied' and 'empty' subfolders")
    parser.add_argument("--model", dest="model_path", default=DEFAULT_MODEL_PATH,
                        help="Where to save the trained model")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train(args.data_dir, args.model_path)
