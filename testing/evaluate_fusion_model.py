import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = PROJECT_ROOT / "models"
PROCESSED_FACES_DIR = PROJECT_ROOT / "processed_faces"
PROCESSED_AUDIO_DIR = PROJECT_ROOT / "processed_audio"
TRAINING_DIR = PROJECT_ROOT / "training"

if str(TRAINING_DIR) not in sys.path:
    sys.path.insert(0, str(TRAINING_DIR))

from audio_data_utils import collect_feature_records, split_records_by_video as split_audio_records  # noqa: E402
from face_data_utils import collect_image_records, make_dataset, split_records_by_video as split_face_records  # noqa: E402
from train_face_model import BATCH_SIZE, IMAGE_SIZE, SEED  # noqa: E402

from fusion_utils import (  # noqa: E402
    fuse_probabilities,
    load_audio_model,
    load_class_names,
    load_face_model,
)


def save_confusion_matrix_plot(matrix, class_names):
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        matrix,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
    )
    plt.title("Fusion Model Confusion Matrix")
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.tight_layout()
    plt.savefig(MODEL_DIR / "fusion_confusion_matrix.png")
    plt.close()


def build_face_video_probabilities(face_model, class_names):
    face_records, split_class_names = collect_image_records(PROCESSED_FACES_DIR)
    _, validation_records, total_videos, train_videos, validation_videos = split_face_records(face_records)
    print("Face validation split summary:")
    print(f"  Total source videos: {total_videos}")
    print(f"  Training videos: {train_videos}")
    print(f"  Validation videos: {validation_videos}")

    if split_class_names != class_names:
        raise ValueError(f"Class order mismatch for face records: {split_class_names} vs {class_names}")

    validation_dataset = make_dataset(
        validation_records,
        class_names,
        IMAGE_SIZE,
        BATCH_SIZE,
        training=False,
        shuffle_seed=SEED,
    )
    probabilities = face_model.predict(validation_dataset, verbose=1)

    video_probabilities = {}
    video_labels = {}
    for record, probability in zip(validation_records, probabilities):
        video_id = record["video_id"]
        video_probabilities.setdefault(video_id, []).append(probability)
        video_labels[video_id] = record["label"]

    video_probabilities = {
        video_id: np.mean(np.stack(probability_list), axis=0)
        for video_id, probability_list in video_probabilities.items()
    }
    return video_probabilities, video_labels


def build_audio_video_probabilities(audio_model, class_names):
    audio_records, split_class_names = collect_feature_records(PROCESSED_AUDIO_DIR)
    _, validation_records, total_videos, train_videos, validation_videos = split_audio_records(audio_records)
    print("Audio validation split summary:")
    print(f"  Total source videos: {total_videos}")
    print(f"  Training videos: {train_videos}")
    print(f"  Validation videos: {validation_videos}")

    if split_class_names != class_names:
        raise ValueError(f"Class order mismatch for audio records: {split_class_names} vs {class_names}")

    probabilities = []
    for record in validation_records:
        feature = np.load(record["path"]).astype(np.float32)
        probability = audio_model.predict(np.expand_dims(feature, axis=0), verbose=0)[0]
        probabilities.append(probability)

    video_probabilities = {}
    video_labels = {}
    for record, probability in zip(validation_records, probabilities):
        video_probabilities[record["video_id"]] = probability
        video_labels[record["video_id"]] = record["label"]

    return video_probabilities, video_labels


def choose_best_face_weight(face_video_probabilities, audio_video_probabilities, class_names):
    candidate_weights = [round(weight, 2) for weight in np.arange(0.50, 0.96, 0.05)]
    best_weight = None
    best_accuracy = -1.0

    common_video_ids = sorted(set(face_video_probabilities.keys()) & set(audio_video_probabilities.keys()))
    true_labels = np.array([class_names.index(face_video_probabilities[video_id]["label"]) for video_id in common_video_ids])

    for face_weight in candidate_weights:
        fused_predictions = []
        for video_id in common_video_ids:
            fused_probability = fuse_probabilities(
                face_video_probabilities[video_id]["probabilities"],
                audio_video_probabilities[video_id]["probabilities"],
                face_weight=face_weight,
            )
            fused_predictions.append(int(np.argmax(fused_probability)))

        accuracy = float(np.mean(np.array(fused_predictions) == true_labels))
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_weight = face_weight

    return best_weight, 1.0 - best_weight, best_accuracy


def main():
    class_names = load_class_names()
    face_model = load_face_model()
    audio_model = load_audio_model()

    face_probs, face_labels = build_face_video_probabilities(face_model, class_names)
    audio_probs, audio_labels = build_audio_video_probabilities(audio_model, class_names)

    common_video_ids = sorted(set(face_probs.keys()) & set(audio_probs.keys()))
    if not common_video_ids:
        raise ValueError("No overlapping validation videos found between face and audio splits.")

    fusion_ready_face = {
        video_id: {"probabilities": face_probs[video_id], "label": face_labels[video_id]}
        for video_id in common_video_ids
    }
    fusion_ready_audio = {
        video_id: {"probabilities": audio_probs[video_id], "label": audio_labels[video_id]}
        for video_id in common_video_ids
    }

    best_face_weight, best_audio_weight, best_accuracy = choose_best_face_weight(
        fusion_ready_face,
        fusion_ready_audio,
        class_names,
    )

    true_indices = []
    predicted_indices = []
    prediction_rows = []

    for video_id in common_video_ids:
        true_label = fusion_ready_face[video_id]["label"]
        fused_probability = fuse_probabilities(
            fusion_ready_face[video_id]["probabilities"],
            fusion_ready_audio[video_id]["probabilities"],
            face_weight=best_face_weight,
            audio_weight=best_audio_weight,
        )
        predicted_index = int(np.argmax(fused_probability))
        true_index = class_names.index(true_label)

        true_indices.append(true_index)
        predicted_indices.append(predicted_index)
        prediction_rows.append(
            {
                "video_id": video_id,
                "true_label": true_label,
                "predicted_label": class_names[predicted_index],
                "confidence": float(np.max(fused_probability)),
            }
        )

    label_indices = list(range(len(class_names)))
    matrix = confusion_matrix(true_indices, predicted_indices, labels=label_indices)
    report = classification_report(
        true_indices,
        predicted_indices,
        labels=label_indices,
        target_names=class_names,
        digits=4,
        output_dict=True,
        zero_division=0,
    )

    with open(MODEL_DIR / "fusion_config.json", "w", encoding="utf-8") as file:
        json.dump(
            {
                "face_weight": best_face_weight,
                "audio_weight": best_audio_weight,
                "validation_accuracy": best_accuracy,
                "num_validation_videos": len(common_video_ids),
            },
            file,
            indent=2,
        )

    pd.DataFrame(matrix, index=class_names, columns=class_names).to_csv(MODEL_DIR / "fusion_confusion_matrix.csv")
    save_confusion_matrix_plot(matrix, class_names)

    with open(MODEL_DIR / "fusion_classification_report.json", "w", encoding="utf-8") as file:
        json.dump(report, file, indent=2)

    pd.DataFrame(prediction_rows).to_csv(MODEL_DIR / "fusion_validation_predictions.csv", index=False)

    print(f"\nBest fusion validation accuracy: {best_accuracy:.4f}")
    print(f"Chosen fusion weights -> face: {best_face_weight:.2f}, audio: {best_audio_weight:.2f}")
    print(f"Saved fusion config to: {MODEL_DIR / 'fusion_config.json'}")


if __name__ == "__main__":
    main()
