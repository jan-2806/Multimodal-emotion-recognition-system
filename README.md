# Emotion AI: Multimodal Emotion Recognition System

## Overview

Emotion AI is a **multimodal deep learning system** designed to detect human emotions by analyzing both **facial expressions (video frames)** and **speech signals (audio)**. By combining these two modalities using a **late fusion approach**, the system improves prediction accuracy, robustness, and real-world applicability.

This project follows a modular pipeline including preprocessing, feature extraction, independent predictions, and fusion, making it scalable and easy to extend.

---

## Features

* **Multimodal Learning (Face + Audio):** Combines visual and audio cues to improve emotion recognition accuracy and reliability compared to single-modality systems.
* **Face Emotion Recognition:** Utilizes MobileNetV2 with transfer learning to extract deep spatial features from facial expressions, enabling accurate classification even with limited data.
* **Audio Emotion Recognition:** Leverages YAMNet pretrained embeddings to capture meaningful acoustic patterns such as tone, pitch, and intensity from speech signals.
* **Late Fusion Mechanism:** Integrates probability outputs from both face and audio models using weighted fusion, enhancing overall prediction robustness.
* **Efficient Preprocessing Pipeline:** Includes face detection, cropping, resizing, and audio normalization to ensure consistent and high-quality inputs.
* **Real-Time Inference Support:** Supports testing using webcam (face) and microphone (audio) inputs for practical, real-world usage.
* **Modular Architecture:** Each component (preprocessing, feature extraction, prediction, fusion) is independently designed, allowing easy updates and scalability.

---

## Installation

### 1. Clone the Repository

```bash
git clone https://github.com/jan-2806/Multimodal-emotion-recognition-system.git
cd Multimodal-emotion-recognition-system
```

### 2. Create Virtual Environment

```bash
conda create -n emotion_ai python=3.10
conda activate emotion_ai
```

### 3. Install Dependencies

```bash
pip install tensorflow opencv-python numpy matplotlib scikit-learn pandas
```

---

## Usage

### 1. Preprocess Data

Extract faces from video dataset:

```bash
python data_processing/extract_faces.py
```
### 2. Preprocess audio

```bash
python data_processing/extract_audio_features.py
```

### 3. Train Models

Train face and audio models:

```bash
python training/train_face_model.py
python training/train_audio_model.py
```

### 4. Test Models

Run predictions using trained models:

```bash
python testing/test_face_model.py
python testing/test_audio_model.py
```

### 5. Multimodal Fusion

Combine predictions from both models:

```bash
python testing/evaluate_fusion_model.py
python testing/test_fusion_model.py
```

---

## Methodology

The system follows a structured pipeline:

1. **Input**: Video containing face and audio
2. **Preprocessing**:

   * Face: detection, cropping, resizing (48×48 grayscale)
   * Audio: extraction, normalization, fixed-length processing
3. **Feature Extraction**:

   * Face: MobileNetV2
   * Audio: YAMNet embeddings
4. **Prediction**:

   * Independent emotion classification
5. **Fusion**:

   * Late fusion of probabilities for final output

---

## Emotion Classes

* 01: Neutral
* 02: Calm
* 03: Happy
* 04: Sad
* 05: Angry
* 06: Fearful
* 07: Disgust
* 08: Surprised 

---

## Project Structure

```
Multimodal-emotion-recognition-system
├── data_processing/
├── training/
├── testing/
├── models/
├── processed_faces/
├── processed_audio/
├── RAVDESS dataset/
├── main.py
├── README.md
└── .gitignore
```

---

## License

This project is licensed for **academic and research purposes**. For other uses, please contact the author.

---

## Contact

**Author:** Janani

For questions, suggestions, or collaboration opportunities, feel free to reach out via GitHub.

---

## Acknowledgements

* RAVDESS Dataset
* TensorFlow & Keras
* OpenCV

---

⭐ If you find this project useful, consider giving it a star!
