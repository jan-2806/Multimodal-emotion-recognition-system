import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import tensorflow as tf

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PROCESSING_DIR = PROJECT_ROOT / "data_processing"
if str(DATA_PROCESSING_DIR) not in sys.path:
    sys.path.insert(0, str(DATA_PROCESSING_DIR))

from audio_feature_utils import FEATURE_SIZE, YAMNET_HANDLE  # noqa: E402
from audio_data_utils import (
    collect_feature_records,
    count_records_per_class,
    make_dataset,
    split_records_by_video,
)

FEATURE_DIR = PROJECT_ROOT / "processed_audio"
MODEL_DIR = PROJECT_ROOT / "models"
BATCH_SIZE = 32
EPOCHS = 40
SEED = 42


def compute_class_weights(class_names, class_counts):
    total_samples = sum(class_counts.values())
    num_classes = len(class_names)
    class_weights = {}

    for index, class_name in enumerate(class_names):
        count = class_counts[class_name]
        class_weights[index] = total_samples / (num_classes * count)

    return class_weights


def build_model(num_classes):
    model = tf.keras.Sequential(
        [
            tf.keras.layers.Input(shape=(FEATURE_SIZE,)),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dense(512, activation="relu", kernel_regularizer=tf.keras.regularizers.l2(1e-4)),
            tf.keras.layers.Dropout(0.35),
            tf.keras.layers.Dense(256, activation="relu", kernel_regularizer=tf.keras.regularizers.l2(1e-4)),
            tf.keras.layers.Dropout(0.3),
            tf.keras.layers.Dense(128, activation="relu", kernel_regularizer=tf.keras.regularizers.l2(1e-4)),
            tf.keras.layers.Dropout(0.25),
            tf.keras.layers.Dense(num_classes, activation="softmax"),
        ],
        name="audio_emotion_yamnet_head",
    )

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def create_callbacks():
    return [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=8,
            restore_best_weights=True,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=4,
            min_lr=1e-6,
            verbose=1,
        ),
        tf.keras.callbacks.ModelCheckpoint(
            filepath=str(MODEL_DIR / "audio_emotion_model_best.keras"),
            monitor="val_accuracy",
            save_best_only=True,
            verbose=1,
        ),
        tf.keras.callbacks.CSVLogger(str(MODEL_DIR / "audio_training_log.csv"), append=False),
    ]


def plot_training_history(history):
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(history.history["accuracy"], label="Train Accuracy")
    plt.plot(history.history["val_accuracy"], label="Validation Accuracy")
    plt.title("Audio Model Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history.history["loss"], label="Train Loss")
    plt.plot(history.history["val_loss"], label="Validation Loss")
    plt.title("Audio Model Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()

    plt.tight_layout()
    plt.savefig(MODEL_DIR / "audio_training_curves.png")
    plt.close()


def save_training_artifacts(model, history, class_names, train_counts, validation_counts, split_summary):
    model.save(MODEL_DIR / "audio_emotion_model_final.keras")

    with open(MODEL_DIR / "audio_class_names.json", "w", encoding="utf-8") as file:
        json.dump(class_names, file, indent=2)

    with open(MODEL_DIR / "audio_train_class_counts.json", "w", encoding="utf-8") as file:
        json.dump(train_counts, file, indent=2)

    with open(MODEL_DIR / "audio_validation_class_counts.json", "w", encoding="utf-8") as file:
        json.dump(validation_counts, file, indent=2)

    with open(MODEL_DIR / "audio_split_summary.json", "w", encoding="utf-8") as file:
        json.dump(split_summary, file, indent=2)

    with open(MODEL_DIR / "audio_training_history.json", "w", encoding="utf-8") as file:
        json.dump(history.history, file, indent=2)

    plot_training_history(history)


def main():
    if not FEATURE_DIR.exists():
        raise FileNotFoundError(
            f"Processed audio directory not found: {FEATURE_DIR}. "
            f"Run data_processing/extract_audio_features.py first."
        )

    tf.keras.utils.set_random_seed(SEED)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    records, class_names = collect_feature_records(FEATURE_DIR)
    train_records, validation_records, total_videos, train_videos, validation_videos = split_records_by_video(records)
    train_counts = count_records_per_class(train_records, class_names)
    validation_counts = count_records_per_class(validation_records, class_names)
    class_weights = compute_class_weights(class_names, train_counts)

    print("Audio video-level split summary:")
    print(f"  Total source videos: {total_videos}")
    print(f"  Training videos: {train_videos}")
    print(f"  Validation videos: {validation_videos}")

    print("\nTraining class distribution:")
    for class_name, count in train_counts.items():
        print(f"  {class_name}: {count}")

    print("\nValidation class distribution:")
    for class_name, count in validation_counts.items():
        print(f"  {class_name}: {count}")

    print("\nClass names:", class_names)
    print("Class weights:", class_weights)

    train_dataset = make_dataset(train_records, class_names, BATCH_SIZE, training=True, shuffle_seed=SEED)
    validation_dataset = make_dataset(validation_records, class_names, BATCH_SIZE, training=False)

    model = build_model(num_classes=len(class_names))
    model.summary()

    history = model.fit(
        train_dataset,
        validation_data=validation_dataset,
        epochs=EPOCHS,
        class_weight=class_weights,
        callbacks=create_callbacks(),
        verbose=2,
    )

    best_model = tf.keras.models.load_model(MODEL_DIR / "audio_emotion_model_best.keras")
    validation_loss, validation_accuracy = best_model.evaluate(validation_dataset, verbose=0)
    print(f"\nBest checkpoint validation loss: {validation_loss:.4f}")
    print(f"Best checkpoint validation accuracy: {validation_accuracy:.4f}")

    split_summary = {
        "split_type": "video_level",
        "batch_size": BATCH_SIZE,
        "epochs": EPOCHS,
        "feature_extractor": "YAMNet",
        "feature_source": YAMNET_HANDLE,
        "feature_shape": [FEATURE_SIZE],
        "total_source_videos": total_videos,
        "training_videos": train_videos,
        "validation_videos": validation_videos,
        "total_training_features": len(train_records),
        "total_validation_features": len(validation_records),
    }
    save_training_artifacts(model, history, class_names, train_counts, validation_counts, split_summary)
    print(f"\nAudio training complete. Model artifacts saved in: {MODEL_DIR}")


if __name__ == "__main__":
    main()
