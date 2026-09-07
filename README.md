# Parking Lot Occupancy Detector

Detects whether parking spots are occupied or empty, two ways: motion-based
analysis of a real parking lot camera feed, and an image classifier trained
on my own photos of a Hot Wheels toy parking lot.

## Features

- **Real parking lot detector** (`src/real_parking_lot/`): given a still
  image of a parking lot, an interactive tool lets you click out the corners
  of each parking spot and saves them to a YAML file. A separate detector
  then reads that file and, frame by frame on a video feed, decides whether
  each spot is occupied by comparing the amount of edge detail (via a
  Laplacian filter) inside each spot's region against a threshold — an empty,
  flat asphalt patch has far less high-frequency detail than one with a car
  parked on it.
- **Hot Wheels parking lot classifier** (`src/hotwheels_parking/`): two ways
  to classify photos I took of Hot Wheels cars in a toy parking lot:
  - `laplacian_detector.py` applies the exact same idea as the real parking
    lot detector — an empty, flat spot has far less high-frequency detail
    than one with a car on it — directly to the whole photo. On this dataset
    it gets every sample right (100%), better than the trained classifier,
    since these are already tightly cropped single-spot photos rather than
    a small region carved out of a wider camera frame.
  - `train.py`/`detect.py` reduce each cropped spot photo to a 16-bin
    histogram of gradient orientations (a hand-rolled, lightweight stand-in
    for HOG, since the OpenCV build originally used here didn't ship the
    `ml` module) and classify it with a linear SVM.

## Tech stack

- Python 3
- OpenCV (`opencv-python`) — image/video I/O, gradients, contours, GUI display
- NumPy
- PyYAML — parking spot coordinate storage
- scikit-learn + joblib — the Hot Wheels SVM classifier and model persistence

## Installation

```bash
pip install -r requirements.txt
```

## Usage

### Real parking lot detector

Mark out the parking spots on a reference image (click 4 corners per spot,
`r` to redo the current image, `q` when done), then run detection on a video:

```bash
cd src/real_parking_lot
python main.py --image ../../media/images/parking_lot_1.png \
                --data ../../data/coordinates/parking_lot_1.yml \
                --video ../../media/videos/parking_lot_2.mp4
```

If you already have a coordinates file, drop `--image` and it will detect
directly:

```bash
python main.py --data ../../data/coordinates/parking_lot_1.yml \
                --video ../../media/videos/parking_lot_2.mp4
```

### Hot Wheels parking lot classifier

Edge-density thresholding (no training needed, add `--show` to see it in a window):

```bash
cd src/hotwheels_parking
python laplacian_detector.py ../../data/hotwheels_samples/occupied/145353551.jpg
```

Or the trained classifier — a pre-trained model (`data/hotwheels_model.joblib`)
is included, trained on the full photo set:

```bash
python detect.py ../../data/hotwheels_samples/occupied/145353551.jpg
```

To retrain the classifier (e.g. on your own photos, dropped into `occupied/`
and `empty/` subfolders):

```bash
python train.py --data path/to/your/dataset --model path/to/save/model.joblib
```

To re-calibrate the edge-density threshold on your own labeled dataset:

```bash
python laplacian_detector.py --calibrate path/to/your/dataset
```

## Project structure

```
src/
  real_parking_lot/     real-camera motion/occupancy detector
  hotwheels_parking/     Hot Wheels photo classifier
data/
  coordinates/           saved parking-spot YAML files
  hotwheels_samples/     small sample set of labeled Hot Wheels photos
  hotwheels_raw/         full photo set (not committed, see below)
  hotwheels_model.joblib pre-trained classifier
media/
  images/                reference parking lot photos
  videos/                sample detection video (not committed, see below)
docs/experiments/        early OpenCV edge/line-detection experiments
tests/                   unit tests
```

## Limitations

- The Hot Wheels dataset is small and imbalanced (25 occupied vs. 138 empty
  photos), so the trained SVM classifier (`train.py`/`detect.py`) is biased
  toward predicting "empty" and only reaches about 90% held-out accuracy.
  The edge-density threshold in `laplacian_detector.py` isn't affected by
  this, since it doesn't train on the class balance.
- The real parking lot detector needs a GUI-capable OpenCV build
  (`opencv-python`, not `opencv-python-headless`) since it displays results
  in a window.
- The full-resolution Hot Wheels photo set (`data/hotwheels_raw/`, ~450 MB)
  and the sample video (`media/videos/`) are not committed to this repository
  due to their size; only a small labeled sample and the already-trained
  model are included.

> TODO: add the course/resource that inspired this project

This project was a simple self-study exercise I built in high school (2020)
to develop my computer/programming skills through courses I was taking at
the time. It was reorganized and cleaned up in 2026 for public release.

## License

MIT — see [LICENSE](LICENSE).
