import cv2
import numpy as np

IMAGE_SIZE = (64, 64)
ORIENTATION_BINS = 16


def extract_features(gray_image):
    """Builds a gradient-orientation histogram for one grayscale image.

    OpenCV's own HOGDescriptor/ml modules are not available in every OpenCV
    build (they are split into optional modules), so the histogram is
    computed by hand from Sobel gradients instead of relying on them.
    """
    resized = cv2.resize(gray_image, IMAGE_SIZE)

    grad_x = cv2.Sobel(resized, cv2.CV_32F, 1, 0)
    grad_y = cv2.Sobel(resized, cv2.CV_32F, 0, 1)
    magnitude, angle = cv2.cartToPolar(grad_x, grad_y)

    bin_index = np.int32(angle / (2 * np.pi) * ORIENTATION_BINS) % ORIENTATION_BINS
    histogram = np.bincount(bin_index.ravel(), weights=magnitude.ravel(), minlength=ORIENTATION_BINS)

    norm = np.linalg.norm(histogram)
    if norm > 0:
        histogram = histogram / norm

    return histogram.astype(np.float32)


def read_image(image_path, flags=cv2.IMREAD_COLOR):
    # cv2.imread() cannot open files whose absolute path contains non-ASCII
    # characters on Windows, so images are read as bytes and decoded instead.
    data = np.fromfile(image_path, dtype=np.uint8)
    image = cv2.imdecode(data, flags)
    if image is None:
        raise FileNotFoundError("Could not read image: %s" % image_path)
    return image


def load_image_as_features(image_path):
    image = read_image(image_path, cv2.IMREAD_GRAYSCALE)
    return extract_features(image)
