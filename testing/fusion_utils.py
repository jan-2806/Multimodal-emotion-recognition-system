import json
import sys
from pathlib import Path

import cv2
import numpy as np
import tensorflow as tf

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = PROJECT_ROOT / "models"
TRAINING_DIR = PROJECT_ROOT / "training"
DATA_PROCESSING_DIR = PROJECT_ROOT / "data_processing"

if str(TRAINING_DIR) not in sys.path:
    sys.path.insert(0, str(TRAINING_DIR))
if str(DATA_PROCESSING_DIR) not in sys.path:
    sys.path.insert(0, str(DATA_PROCESSING_DIR))

from audio_feature_utils import extract_audio_feature_tensor  # noqa: E402
from train_face_model import IMAGE_SIZE as FACE_IMAGE_SIZE, sparse_categorical_focal_loss  # noqa: E402

FACE_MARGIN = 0.15
DEFAULT_FACE_WEIGHT = 0.75
DEFAULT_AUDIO_WEIGHT = 0.25

CASCADE_PATHS = [
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml",
    cv2.data.haarcascades + "haarcascade_frontalface_alt2.xml",
    cv2.data.haarcascades + "haarcascade_profileface.xml",
]
FACE_CASCADES = [
    cv2.CascadeClassifier(path)
    for path in CASCADE_PATHS
    if Path(path).exists()
]


def load_class_names():
    face_path = MODEL_DIR / "face_class_names.json"
    audio_path = MODEL_DIR / "audio_class_names.json"

    with open(face_path, "r", encoding="utf-8") as file:
        face_class_names = json.load(file)

    with open(audio_path, "r", encoding="utf-8") as file:
        audio_class_names = json.load(file)

    if face_class_names != audio_class_names:
        raise ValueError(
            f"Face/audio class order mismatch: {face_class_names} vs {audio_class_names}"
        )

    return face_class_names


def load_face_model():
    model_path = MODEL_DIR / "face_emotion_model_best.keras"
    return tf.keras.models.load_model(
        model_path,
        custom_objects={"loss": sparse_categorical_focal_loss()},
        compile=False,
    )


def load_audio_model():
    model_path = MODEL_DIR / "audio_emotion_model_best.keras"
    return tf.keras.models.load_model(model_path, compile=False)


def detect_best_face(gray_frame):
    enhanced = cv2.equalizeHist(gray_frame)
    detection_attempts = [
        (gray_frame, 1.1, 4, (40, 40)),
        (enhanced, 1.1, 3, (30, 30)),
        (enhanced, 1.05, 2, (24, 24)),
    ]

    best_face = None
    best_area = 0

    for cascade in FACE_CASCADES:
        for image, scale_factor, min_neighbors, min_size in detection_attempts:
            faces = cascade.detectMultiScale(
                image,
                scaleFactor=scale_factor,
                minNeighbors=min_neighbors,
                minSize=min_size,
            )

            for (x, y, w, h) in faces:
                area = w * h
                if area > best_area:
                    best_face = (x, y, w, h)
                    best_area = area

        if best_face is not None:
            return best_face

    return None


def expand_face_box(face_box, frame_shape):
    x, y, w, h = face_box
    frame_height, frame_width = frame_shape[:2]
    pad_w = int(w * FACE_MARGIN)
    pad_h = int(h * FACE_MARGIN)

    x1 = max(0, x - pad_w)
    y1 = max(0, y - pad_h)
    x2 = min(frame_width, x + w + pad_w)
    y2 = min(frame_height, y + h + pad_h)
    return x1, y1, x2, y2


def preprocess_face(gray_face):
    equalized_face = cv2.equalizeHist(gray_face)
    resized_face = cv2.resize(equalized_face, FACE_IMAGE_SIZE)
    tensor = resized_face.astype("float32")
    return np.expand_dims(tensor, axis=(0, -1))


def predict_face_probabilities_from_crop(face_model, gray_face):
    face_tensor = preprocess_face(gray_face)
    return face_model.predict(face_tensor, verbose=0)[0]


def predict_face_probabilities_from_video(face_model, video_path, frame_step=5, max_faces=15):
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video for face fusion: {video_path}")

    collected_probabilities = []
    frame_index = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_index % frame_step == 0:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            face_box = detect_best_face(gray)

            if face_box is not None:
                x1, y1, x2, y2 = expand_face_box(face_box, frame.shape)
                gray_face = gray[y1:y2, x1:x2]
                if gray_face.size != 0:
                    collected_probabilities.append(predict_face_probabilities_from_crop(face_model, gray_face))
                    if len(collected_probabilities) >= max_faces:
                        break

        frame_index += 1

    cap.release()

    if not collected_probabilities:
        return None, 0

    return np.mean(np.stack(collected_probabilities), axis=0), len(collected_probabilities)


def predict_audio_probabilities_from_media(audio_model, media_path):
    feature = extract_audio_feature_tensor(media_path)
    return audio_model.predict(np.expand_dims(feature, axis=0), verbose=0)[0]


def fuse_probabilities(face_probabilities, audio_probabilities, face_weight=DEFAULT_FACE_WEIGHT, audio_weight=None):
    if audio_weight is None:
        audio_weight = 1.0 - face_weight

    if face_probabilities is None and audio_probabilities is None:
        raise ValueError("At least one modality must provide probabilities for fusion.")

    if face_probabilities is None:
        return audio_probabilities

    if audio_probabilities is None:
        return face_probabilities

    total_weight = face_weight + audio_weight
    normalized_face_weight = face_weight / total_weight
    normalized_audio_weight = audio_weight / total_weight
    return (normalized_face_weight * face_probabilities) + (normalized_audio_weight * audio_probabilities)


def summarize_probabilities(class_names, probabilities, top_k=3):
    sorted_indices = np.argsort(probabilities)[::-1][:top_k]
    return [(class_names[index], float(probabilities[index])) for index in sorted_indices]


def load_fusion_config():
    config_path = MODEL_DIR / "fusion_config.json"
    if not config_path.exists():
        return {
            "face_weight": DEFAULT_FACE_WEIGHT,
            "audio_weight": DEFAULT_AUDIO_WEIGHT,
        }

    with open(config_path, "r", encoding="utf-8") as file:
        return json.load(file)
