import argparse
import json
import sys
from pathlib import Path

import numpy as np
import tensorflow as tf

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = PROJECT_ROOT / "models"
DATA_PROCESSING_DIR = PROJECT_ROOT / "data_processing"
BEST_MODEL_PATH = MODEL_DIR / "audio_emotion_model_best.keras"
FINAL_MODEL_PATH = MODEL_DIR / "audio_emotion_model_final.keras"
CLASS_NAMES_PATH = MODEL_DIR / "audio_class_names.json"
DEFAULT_TOP_K = 3
DEFAULT_MIN_CONFIDENCE = 0.40
DEFAULT_MIN_MARGIN = 0.08

if str(DATA_PROCESSING_DIR) not in sys.path:
    sys.path.insert(0, str(DATA_PROCESSING_DIR))

from audio_feature_utils import (  # noqa: E402
    SAMPLE_RATE,
    extract_audio_feature_from_signal,
    extract_audio_feature_tensor,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Test the trained audio emotion model on a video, audio file, or live microphone."
    )
    mode_group = parser.add_mutually_exclusive_group(required=True)
    mode_group.add_argument("--video", type=str, help="Path to an MP4 video with audio.")
    mode_group.add_argument("--audio", type=str, help="Path to an audio file such as WAV/MP3.")
    mode_group.add_argument("--microphone", action="store_true", help="Record a short live microphone clip.")
    parser.add_argument(
        "--top-k",
        type=int,
        default=DEFAULT_TOP_K,
        help="How many top probabilities to print.",
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=3.0,
        help="Recording duration in seconds for microphone mode.",
    )
    parser.add_argument(
        "--device",
        type=int,
        default=None,
        help="Optional microphone device index for microphone mode.",
    )
    parser.add_argument(
        "--min-confidence",
        type=float,
        default=DEFAULT_MIN_CONFIDENCE,
        help="Minimum top probability required before returning a concrete emotion label.",
    )
    parser.add_argument(
        "--min-margin",
        type=float,
        default=DEFAULT_MIN_MARGIN,
        help="Minimum gap between the top two probabilities before returning a concrete emotion label.",
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
            "No trained audio model found. Run training/train_audio_model.py first."
        )

    print(f"Loading audio model from: {model_path}")
    return tf.keras.models.load_model(model_path)


def summarize_probabilities(class_names, probabilities, top_k):
    sorted_indices = np.argsort(probabilities)[::-1][:top_k]
    return [(class_names[index], float(probabilities[index])) for index in sorted_indices]


def record_from_microphone(duration_seconds, device=None):
    try:
        import sounddevice as sd
    except ImportError as error:
        raise ImportError(
            "Microphone testing requires sounddevice. Install it with: "
            "python -m pip install sounddevice"
        ) from error

    total_samples = int(SAMPLE_RATE * duration_seconds)
    print(f"Recording {duration_seconds:.1f} seconds from microphone...")
    recording = sd.rec(
        frames=total_samples,
        samplerate=SAMPLE_RATE,
        channels=1,
        dtype="float32",
        device=device,
    )
    sd.wait()
    print("Recording complete.")
    return np.squeeze(recording, axis=1)


def resolve_prediction(top_predictions, min_confidence, min_margin):
    top_label, top_probability = top_predictions[0]
    runner_up_probability = top_predictions[1][1] if len(top_predictions) > 1 else 0.0
    margin = top_probability - runner_up_probability

    if top_probability < min_confidence or margin < min_margin:
        return "uncertain", top_probability, margin

    return top_label, top_probability, margin


def print_predictions(source_name, top_predictions, min_confidence, min_margin):
    resolved_label, top_probability, margin = resolve_prediction(top_predictions, min_confidence, min_margin)
    print(f"Input source: {source_name}")
    print(f"Predicted emotion: {resolved_label}")
    print(f"Top confidence: {top_probability:.2%}")
    print(f"Top-2 margin: {margin:.2%}")
    print(
        f"Decision rule: min_confidence={min_confidence:.2f}, "
        f"min_margin={min_margin:.2f}"
    )
    print("Top predictions:")
    for class_name, probability in top_predictions:
        print(f"  {class_name}: {probability:.4f}")


def main():
    args = parse_args()
    class_names = load_class_names()
    model = load_trained_model()

    if args.microphone:
        signal = record_from_microphone(args.duration, args.device)
        feature = extract_audio_feature_from_signal(signal)
        source_name = f"microphone ({args.duration:.1f}s)"
    else:
        media_path = Path(args.video or args.audio)
        if not media_path.exists():
            raise FileNotFoundError(f"Input file not found: {media_path}")

        feature = extract_audio_feature_tensor(media_path)
        source_name = str(media_path)

    probabilities = model.predict(np.expand_dims(feature, axis=0), verbose=0)[0]
    top_predictions = summarize_probabilities(class_names, probabilities, args.top_k)
    print_predictions(source_name, top_predictions, args.min_confidence, args.min_margin)


if __name__ == "__main__":
    main()
