import argparse
import json
from collections import deque
from pathlib import Path

import cv2
import numpy as np
import tensorflow as tf

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = PROJECT_ROOT / "models"
BEST_MODEL_PATH = MODEL_DIR / "face_emotion_model_best.keras"
FINAL_MODEL_PATH = MODEL_DIR / "face_emotion_model_final.keras"
CLASS_NAMES_PATH = MODEL_DIR / "face_class_names.json"
FACE_MARGIN = 0.15
DEFAULT_TOP_K = 3

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


def parse_args():
    parser = argparse.ArgumentParser(
        description="Test the trained face emotion model on a raw image, cropped face, or webcam."
    )
    mode_group = parser.add_mutually_exclusive_group(required=True)
    mode_group.add_argument("--image", type=str, help="Path to a raw image for prediction.")
    mode_group.add_argument(
        "--cropped-face",
        type=str,
        help="Path to an already cropped face image such as one from processed_faces.",
    )
    mode_group.add_argument("--webcam", action="store_true", help="Use the webcam for live testing.")
    parser.add_argument(
        "--camera-index",
        type=int,
        default=0,
        help="Webcam index to use when --webcam is selected.",
    )
    parser.add_argument(
        "--save-output",
        type=str,
        help="Optional output image path for saving the annotated image result.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=DEFAULT_TOP_K,
        help="How many top probabilities to print for image-based prediction.",
    )
    parser.add_argument(
        "--smoothing-window",
        type=int,
        default=5,
        help="How many recent webcam predictions to average.",
    )
    return parser.parse_args()


def load_class_names():
    if not CLASS_NAMES_PATH.exists():
        raise FileNotFoundError(f"Class names file not found: {CLASS_NAMES_PATH}")

    with open(CLASS_NAMES_PATH, "r", encoding="utf-8") as file:
        return json.load(file)


def load_trained_model():
    if BEST_MODEL_PATH.exists():
        model_path = BEST_MODEL_PATH
    elif FINAL_MODEL_PATH.exists():
        model_path = FINAL_MODEL_PATH
    else:
        raise FileNotFoundError(
            "No trained model found. Run training/train_face_model.py first."
        )

    print(f"Loading model from: {model_path}")
    model = tf.keras.models.load_model(model_path)
    input_shape = model.input_shape
    image_size = (int(input_shape[1]), int(input_shape[2]))
    return model, image_size


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


def preprocess_face(gray_face, image_size):
    equalized_face = cv2.equalizeHist(gray_face)
    resized_face = cv2.resize(equalized_face, image_size)
    tensor = resized_face.astype("float32")
    return np.expand_dims(tensor, axis=(0, -1))


def predict_probabilities(model, gray_face, image_size):
    input_tensor = preprocess_face(gray_face, image_size)
    probabilities = model.predict(input_tensor, verbose=0)[0]
    return probabilities


def summarize_probabilities(class_names, probabilities, top_k):
    sorted_indices = np.argsort(probabilities)[::-1][:top_k]
    return [(class_names[index], float(probabilities[index])) for index in sorted_indices]


def annotate_frame(frame, face_box, label, confidence, top_predictions=None):
    x1, y1, x2, y2 = face_box
    annotated = frame.copy()
    cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)

    lines = [f"{label}: {confidence:.2%}"]
    if top_predictions:
        for class_name, probability in top_predictions[1:]:
            lines.append(f"{class_name}: {probability:.2%}")

    y_text = max(25, y1 - 10)
    for line in lines:
        cv2.putText(
            annotated,
            line,
            (x1, y_text),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )
        y_text += 24

    return annotated


def print_top_predictions(top_predictions):
    print("Top predictions:")
    for class_name, probability in top_predictions:
        print(f"  {class_name}: {probability:.4f}")


def predict_from_raw_image(image_path, model, class_names, image_size, top_k, save_output=None):
    image_path = Path(image_path)
    image = cv2.imread(str(image_path))
    if image is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    detected_face = detect_best_face(gray)

    if detected_face is None:
        print("No face detected in the image. Using the whole image as a fallback.")
        gray_face = gray
        face_box = (0, 0, image.shape[1], image.shape[0])
    else:
        x1, y1, x2, y2 = expand_face_box(detected_face, image.shape)
        gray_face = gray[y1:y2, x1:x2]
        face_box = (x1, y1, x2, y2)

    probabilities = predict_probabilities(model, gray_face, image_size)
    top_predictions = summarize_probabilities(class_names, probabilities, top_k)
    label, confidence = top_predictions[0]
    annotated = annotate_frame(image, face_box, label, confidence, top_predictions)

    print(f"Predicted emotion: {label}")
    print(f"Confidence: {confidence:.2%}")
    print_top_predictions(top_predictions)

    if save_output:
        output_path = Path(save_output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(output_path), annotated)
        print(f"Annotated output saved to: {output_path}")

    cv2.imshow("Face Emotion Prediction", annotated)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def predict_from_cropped_face(image_path, model, class_names, image_size, top_k, save_output=None):
    image_path = Path(image_path)
    image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise FileNotFoundError(f"Could not read cropped face image: {image_path}")

    probabilities = predict_probabilities(model, image, image_size)
    top_predictions = summarize_probabilities(class_names, probabilities, top_k)
    label, confidence = top_predictions[0]

    annotated = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    annotated = annotate_frame(
        annotated,
        (0, 0, annotated.shape[1], annotated.shape[0]),
        label,
        confidence,
        top_predictions,
    )

    print(f"Predicted emotion: {label}")
    print(f"Confidence: {confidence:.2%}")
    print_top_predictions(top_predictions)

    if save_output:
        output_path = Path(save_output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(output_path), annotated)
        print(f"Annotated output saved to: {output_path}")

    cv2.imshow("Cropped Face Prediction", annotated)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def predict_from_webcam(model, class_names, image_size, camera_index, smoothing_window):
    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open webcam with index {camera_index}")

    probability_window = deque(maxlen=max(1, smoothing_window))
    print("Webcam started. Press 'q' to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        detected_face = detect_best_face(gray)
        display_frame = frame.copy()

        if detected_face is not None:
            x1, y1, x2, y2 = expand_face_box(detected_face, frame.shape)
            gray_face = gray[y1:y2, x1:x2]

            if gray_face.size != 0:
                probabilities = predict_probabilities(model, gray_face, image_size)
                probability_window.append(probabilities)
                averaged_probabilities = np.mean(np.stack(probability_window), axis=0)
                top_predictions = summarize_probabilities(class_names, averaged_probabilities, top_k=3)
                label, confidence = top_predictions[0]
                display_frame = annotate_frame(frame, (x1, y1, x2, y2), label, confidence, top_predictions)

        cv2.imshow("Face Emotion Prediction", display_frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


def main():
    args = parse_args()
    class_names = load_class_names()
    model, image_size = load_trained_model()

    if args.image:
        predict_from_raw_image(args.image, model, class_names, image_size, args.top_k, args.save_output)
    elif args.cropped_face:
        predict_from_cropped_face(args.cropped_face, model, class_names, image_size, args.top_k, args.save_output)
    else:
        predict_from_webcam(model, class_names, image_size, args.camera_index, args.smoothing_window)


if __name__ == "__main__":
    main()
