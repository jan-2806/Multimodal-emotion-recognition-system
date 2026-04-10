import argparse
from pathlib import Path

import numpy as np

from fusion_utils import (
    DEFAULT_AUDIO_WEIGHT,
    DEFAULT_FACE_WEIGHT,
    fuse_probabilities,
    load_audio_model,
    load_class_names,
    load_face_model,
    load_fusion_config,
    predict_audio_probabilities_from_media,
    predict_face_probabilities_from_video,
    summarize_probabilities,
)

DEFAULT_TOP_K = 3


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run multimodal late-fusion emotion prediction on a video file."
    )
    parser.add_argument(
        "--video",
        type=str,
        required=True,
        help="Path to an MP4 video containing both face and audio information.",
    )
    parser.add_argument(
        "--face-weight",
        type=float,
        default=None,
        help="Optional face weight override for fusion. Defaults to the saved fusion config.",
    )
    parser.add_argument(
        "--audio-weight",
        type=float,
        default=None,
        help="Optional audio weight override for fusion. Defaults to 1 - face_weight or the saved config.",
    )
    parser.add_argument(
        "--frame-step",
        type=int,
        default=5,
        help="Process every Nth video frame for face inference.",
    )
    parser.add_argument(
        "--max-faces",
        type=int,
        default=15,
        help="Maximum number of detected face crops to use from the video.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=DEFAULT_TOP_K,
        help="How many top probabilities to print per modality.",
    )
    return parser.parse_args()


def print_prediction_block(title, top_predictions):
    print(f"\n{title}")
    for class_name, probability in top_predictions:
        print(f"  {class_name}: {probability:.4f}")


def resolve_weights(args):
    if args.face_weight is None and args.audio_weight is None:
        config = load_fusion_config()
        return float(config.get("face_weight", DEFAULT_FACE_WEIGHT)), float(
            config.get("audio_weight", DEFAULT_AUDIO_WEIGHT)
        )

    face_weight = DEFAULT_FACE_WEIGHT if args.face_weight is None else args.face_weight
    audio_weight = (1.0 - face_weight) if args.audio_weight is None else args.audio_weight
    return face_weight, audio_weight


def main():
    args = parse_args()
    video_path = Path(args.video)
    if not video_path.exists():
        raise FileNotFoundError(f"Input video not found: {video_path}")

    class_names = load_class_names()
    face_model = load_face_model()
    audio_model = load_audio_model()
    face_weight, audio_weight = resolve_weights(args)

    face_probabilities, face_count = predict_face_probabilities_from_video(
        face_model,
        video_path,
        frame_step=args.frame_step,
        max_faces=args.max_faces,
    )
    audio_probabilities = predict_audio_probabilities_from_media(audio_model, video_path)
    fused_probabilities = fuse_probabilities(
        face_probabilities,
        audio_probabilities,
        face_weight=face_weight,
        audio_weight=audio_weight,
    )

    face_top = (
        summarize_probabilities(class_names, face_probabilities, args.top_k)
        if face_probabilities is not None
        else []
    )
    audio_top = summarize_probabilities(class_names, audio_probabilities, args.top_k)
    fused_top = summarize_probabilities(class_names, fused_probabilities, args.top_k)

    final_label, final_confidence = fused_top[0]

    print(f"Input video: {video_path}")
    print(f"Face samples used: {face_count}")
    print(f"Fusion weights -> face: {face_weight:.2f}, audio: {audio_weight:.2f}")
    print(f"\nFinal fused prediction: {final_label}")
    print(f"Final confidence: {final_confidence:.2%}")

    if face_top:
        print_prediction_block("Face-only top predictions:", face_top)
    else:
        print("\nFace-only top predictions:")
        print("  No usable face detections were found. Fusion fell back to audio.")

    print_prediction_block("Audio-only top predictions:", audio_top)
    print_prediction_block("Fusion top predictions:", fused_top)


if __name__ == "__main__":
    main()
