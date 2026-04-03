# Emotion Recognition System using Multimodal Deep Learning

This project aims to detect human emotions using:

- facial expressions from video frames
- voice patterns from audio

The long-term goal is a multimodal emotion recognition system that combines both face and audio predictions.

## Current Progress

The project has completed the face-preprocessing and face-model baseline stages.

Completed so far:

- extracted face images from the RAVDESS video dataset
- organized extracted faces into emotion-wise folders
- built a face-only training pipeline
- added a face-model testing script for image and webcam input

Current emotion classes:

- angry
- calm
- disgust
- fear
- happy
- neutral
- sad
- surprise

## Project Structure

```text
Emotion Recognition
├── data_processing/
│   └── extract_faces.py
├── training/
│   └── train_face_model.py
├── testing/
│   └── test_face_model.py
├── models/
├── processed_faces/
├── RAVDESS dataset/
└── main.py
```

## Environment

Recommended setup:

- PyCharm
- Anaconda / Conda environment
- Python 3.10

Suggested environment name:

- `emotion_ai`

Main libraries used:

- TensorFlow / Keras
- OpenCV
- NumPy
- Matplotlib
- scikit-learn
- pandas

## Dataset

This project uses the **RAVDESS** dataset for facial emotion extraction and training.

The raw dataset and generated face images are intentionally excluded from GitHub because they are large generated assets and should be stored locally.

## Notes

- This repository currently focuses on the face-only pipeline.
- Audio modeling and multimodal fusion are planned next.
- Trained model files are not committed by default.
