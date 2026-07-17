# Drowsy-Face

A Python desktop application for webcam-based drowsiness detection using OpenCV, Tkinter, and a trained TensorFlow/Keras eye-state model.

## Repository files

- `change.py` — Main application file (Tkinter UI + webcam/detection logic).
- `Drowsy model code.ipynb` — Jupyter notebook used for model experimentation/training workflow.
- `Drowsy model code.html` — HTML export of the notebook.
- `eye.png` — UI background image shown in the app window.
- `beep-07.wav` — Alert sound played when drowsiness is detected.
- `haar_face.xml` — Haar face cascade used in drowsiness mode.
- `haar_eye.xml` — Haar eye cascade asset.
- `haarcascade_eye.xml` — Eye cascade used for eye detection in drowsiness mode.
- `haarcascade_eye_tree_eyeglasses.xml` — Eye-with-glasses cascade used in detection mode.
- `lbpcascade_frontalface.xml` — LBP frontal-face cascade used in detection mode.

## Prerequisites

- Python 3.8+
- Webcam access
- A trained model file named `my_model.h5` in the repository root (same folder as `change.py`)

## Install dependencies

From the repository root (`/home/runner/work/Drowsy-Face/Drowsy-Face`):

```bash
python -m pip install --upgrade pip
python -m pip install numpy opencv-python pygame tensorflow ttkthemes
```

> `tkinter` is part of standard Python on most systems. If missing on Linux, install OS package `python3-tk`.

## How to run the program

1. Keep all repository files in the same root folder.
2. Place `my_model.h5` in the same folder.
3. Run:

```bash
python change.py
```

## App controls

- **Open Cam**: Opens webcam preview.
- **Open Cam & Detect**: Detects face/eyes using cascade classifiers.
- **Detect Drowsiness With Sound**: Runs eye-state model and plays alert sound when drowsiness is detected.
- Press **`q`** in OpenCV windows to close webcam feed.
