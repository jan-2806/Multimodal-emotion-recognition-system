import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = PROJECT_ROOT / "models"
DATA_DIR = PROJECT_ROOT / "processed_faces"
TRAINING_DIR = PROJECT_ROOT / "training"

if str(TRAINING_DIR) not in sys.path:
    sys.path.insert(0, str(TRAINING_DIR))

from face_data_utils import collect_image_records, make_dataset, split_records_by_video  # noqa: E402
from train_face_model import BATCH_SIZE, IMAGE_SIZE, SEED, sparse_categorical_focal_loss  # noqa: E402


def load_class_names():
    class_names_path = MODEL_DIR / "face_class_names.json"
    if not class_names_path.exists():
        raise FileNotFoundError(f"Missing class name file: {class_names_path}")

    with open(class_names_path, "r", encoding="utf-8") as file:
        return json.load(file)


def load_best_model():
    model_path = MODEL_DIR / "face_emotion_model_best.keras"
    if not model_path.exists():
        raise FileNotFoundError(f"Missing best model checkpoint: {model_path}")

    model = tf.keras.models.load_model(
        model_path,
        custom_objects={"loss": sparse_categorical_focal_loss()},
        compile=False,
    )
    model.compile(
        optimizer=tf.keras.optimizers.Adam(),
        loss=sparse_categorical_focal_loss(),
        metrics=["accuracy"],
    )
    return model


def build_validation_records():
    records, class_names = collect_image_records(DATA_DIR)
    _, validation_records, total_videos, train_videos, validation_videos = split_records_by_video(records)
    print("Video-level split summary:")
    print(f"  Total source videos: {total_videos}")
    print(f"  Training videos: {train_videos}")
    print(f"  Validation videos: {validation_videos}")
    return validation_records, class_names


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
    plt.title("Face Model Confusion Matrix")
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.tight_layout()
    plt.savefig(MODEL_DIR / "face_confusion_matrix.png")
    plt.close()


def main():
    tf.keras.utils.set_random_seed(SEED)
    validation_records, split_class_names = build_validation_records()
    saved_class_names = load_class_names()

    if split_class_names != saved_class_names:
        raise ValueError(
            f"Class order mismatch between dataset split {split_class_names} and saved model metadata {saved_class_names}."
        )

    model = load_best_model()
    validation_dataset = make_dataset(
        validation_records,
        saved_class_names,
        IMAGE_SIZE,
        BATCH_SIZE,
        training=False,
        shuffle_seed=SEED,
    )

    true_labels = np.array([saved_class_names.index(record["label"]) for record in validation_records])
    probabilities = model.predict(validation_dataset, verbose=1)
    predicted_labels = np.argmax(probabilities, axis=1)

    matrix = confusion_matrix(true_labels, predicted_labels)
    report = classification_report(
        true_labels,
        predicted_labels,
        target_names=saved_class_names,
        digits=4,
        output_dict=True,
        zero_division=0,
    )

    matrix_df = pd.DataFrame(matrix, index=saved_class_names, columns=saved_class_names)
    matrix_df.to_csv(MODEL_DIR / "face_confusion_matrix.csv")
    save_confusion_matrix_plot(matrix, saved_class_names)

    with open(MODEL_DIR / "face_classification_report.json", "w", encoding="utf-8") as file:
        json.dump(report, file, indent=2)

    prediction_rows = []
    for record, true_index, predicted_index, probability_row in zip(
        validation_records,
        true_labels,
        predicted_labels,
        probabilities,
    ):
        prediction_rows.append(
            {
                "path": str(record["path"]),
                "true_label": saved_class_names[true_index],
                "predicted_label": saved_class_names[predicted_index],
                "confidence": float(np.max(probability_row)),
            }
        )

    pd.DataFrame(prediction_rows).to_csv(MODEL_DIR / "face_validation_predictions.csv", index=False)

    accuracy = float(np.mean(predicted_labels == true_labels))
    print(f"\nValidation accuracy from best checkpoint: {accuracy:.4f}")
    print(f"Saved confusion matrix to: {MODEL_DIR / 'face_confusion_matrix.csv'}")
    print(f"Saved confusion matrix plot to: {MODEL_DIR / 'face_confusion_matrix.png'}")
    print(f"Saved classification report to: {MODEL_DIR / 'face_classification_report.json'}")


if __name__ == "__main__":
    main()
