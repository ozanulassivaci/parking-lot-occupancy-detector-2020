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
- **Hot Wheels parking lot classifier** (`src/hotwheels_parking/`): a small
  image classifier trained on photos I took of Hot Wheels cars in a toy
  parking lot. Each cropped spot photo is reduced to a 16-bin histogram of
  gradient orientations (a hand-rolled, lightweight stand-in for HOG, since
  the OpenCV build used here doesn't ship the `ml` module), and a linear SVM
  classifies the spot as `occupied` or `empty`.

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

A pre-trained model (`data/hotwheels_model.joblib`) is included, trained on
the full photo set. Classify a cropped spot photo:

```bash
cd src/hotwheels_parking
python detect.py ../../data/hotwheels_samples/occupied/789784634586_1121.jpg
```

To retrain (e.g. on your own photos, dropped into `occupied/` and `empty/`
subfolders):

```bash
python train.py --data path/to/your/dataset --model path/to/save/model.joblib
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

- The Hot Wheels dataset is small and imbalanced (138 occupied vs. 25 empty
  photos), so the classifier is biased toward predicting "occupied" and only
  reaches about 88% held-out accuracy.
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
