import json
import shutil
from pathlib import Path

import matplotlib.pyplot as plt
import tensorflow as tf

from face_data_utils import (
    collect_image_records,
    count_records_per_class,
    make_dataset,
    split_records_by_video,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "processed_faces"
MODEL_DIR = PROJECT_ROOT / "models"
IMAGE_SIZE = (96, 96)
BATCH_SIZE = 32
INITIAL_EPOCHS = 10
FINE_TUNE_EPOCHS = 8
SEED = 42
FINE_TUNE_LAYERS = 45
FOCAL_GAMMA = 1.5


def compute_class_weights(class_names, class_counts):
    total_samples = sum(class_counts.values())
    num_classes = len(class_names)
    class_weights = {}

    for index, class_name in enumerate(class_names):
        count = class_counts[class_name]
        class_weights[index] = total_samples / (num_classes * count)

    return class_weights


def build_model(num_classes):
    augmentation = tf.keras.Sequential(
        [
            tf.keras.layers.RandomFlip("horizontal"),
            tf.keras.layers.RandomRotation(0.08),
            tf.keras.layers.RandomZoom(0.12),
            tf.keras.layers.RandomTranslation(0.08, 0.08),
            tf.keras.layers.RandomContrast(0.12),
        ],
        name="augmentation",
    )

    try:
        base_model = tf.keras.applications.MobileNetV2(
            input_shape=(IMAGE_SIZE[0], IMAGE_SIZE[1], 3),
            include_top=False,
            weights="imagenet",
        )
        print("Loaded ImageNet pretrained weights for MobileNetV2.")
    except Exception as error:
        print(f"Could not load ImageNet weights ({error}). Falling back to random initialization.")
        base_model = tf.keras.applications.MobileNetV2(
            input_shape=(IMAGE_SIZE[0], IMAGE_SIZE[1], 3),
            include_top=False,
            weights=None,
        )

    base_model.trainable = False

    inputs = tf.keras.Input(shape=(IMAGE_SIZE[0], IMAGE_SIZE[1], 1), name="face_input")
    rgb = tf.keras.layers.Concatenate(name="grayscale_to_rgb")([inputs, inputs, inputs])
    augmented = augmentation(rgb)
    normalized = tf.keras.layers.Rescaling(1.0 / 127.5, offset=-1, name="mobilenet_rescaling")(augmented)
    features = base_model(normalized, training=False)
    pooled = tf.keras.layers.GlobalAveragePooling2D()(features)
    dropped = tf.keras.layers.Dropout(0.35)(pooled)
    dense = tf.keras.layers.Dense(
        192,
        activation="relu",
        kernel_regularizer=tf.keras.regularizers.l2(2e-4),
    )(dropped)
    normalized_dense = tf.keras.layers.BatchNormalization()(dense)
    dropped_dense = tf.keras.layers.Dropout(0.35)(normalized_dense)
    outputs = tf.keras.layers.Dense(num_classes, activation="softmax")(dropped_dense)

    model = tf.keras.Model(inputs=inputs, outputs=outputs, name="face_emotion_mobilenet")
    return model, base_model


def sparse_categorical_focal_loss(gamma=FOCAL_GAMMA):
    def loss(y_true, y_pred):
        y_true = tf.cast(tf.reshape(y_true, [-1]), tf.int32)
        y_true_one_hot = tf.one_hot(y_true, depth=tf.shape(y_pred)[-1])
        y_pred = tf.clip_by_value(y_pred, tf.keras.backend.epsilon(), 1.0 - tf.keras.backend.epsilon())
        cross_entropy = -y_true_one_hot * tf.math.log(y_pred)
        focal_weight = tf.pow(1.0 - y_pred, gamma)
        loss_value = tf.reduce_sum(focal_weight * cross_entropy, axis=-1)
        return tf.reduce_mean(loss_value)

    return loss


def compile_model(model, learning_rate):
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss=sparse_categorical_focal_loss(),
        metrics=["accuracy"],
    )


class BestMetricTracker(tf.keras.callbacks.Callback):
    def __init__(self, metadata_path):
        super().__init__()
        self.metadata_path = Path(metadata_path)
        if self.metadata_path.exists():
            with open(self.metadata_path, "r", encoding="utf-8") as file:
                metadata = json.load(file)
            self.best_val_accuracy = float(metadata.get("best_val_accuracy", float("-inf")))
        else:
            self.best_val_accuracy = float("-inf")

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        current_val_accuracy = logs.get("val_accuracy")
        if current_val_accuracy is None:
            return

        if current_val_accuracy > self.best_val_accuracy:
            self.best_val_accuracy = float(current_val_accuracy)
            with open(self.metadata_path, "w", encoding="utf-8") as file:
                json.dump(
                    {
                        "best_val_accuracy": self.best_val_accuracy,
                        "best_epoch_1_based": epoch + 1,
                    },
                    file,
                    indent=2,
                )


class GlobalBestModelCheckpoint(tf.keras.callbacks.Callback):
    def __init__(self, filepath, metadata_path):
        super().__init__()
        self.filepath = Path(filepath)
        self.metadata_path = Path(metadata_path)
        if self.metadata_path.exists():
            with open(self.metadata_path, "r", encoding="utf-8") as file:
                metadata = json.load(file)
            self.best_val_accuracy = float(metadata.get("best_val_accuracy", float("-inf")))
        else:
            self.best_val_accuracy = float("-inf")

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        current_val_accuracy = logs.get("val_accuracy")
        if current_val_accuracy is None:
            return

        if current_val_accuracy > self.best_val_accuracy:
            self.best_val_accuracy = float(current_val_accuracy)
            print(f"\nEpoch {epoch + 1}: val_accuracy improved to {self.best_val_accuracy:.5f}, saving global best model.")
            self.model.save(self.filepath)
            with open(self.metadata_path, "w", encoding="utf-8") as file:
                json.dump(
                    {
                        "best_val_accuracy": self.best_val_accuracy,
                        "best_epoch_1_based": epoch + 1,
                    },
                    file,
                    indent=2,
                )
        else:
            print(
                f"\nEpoch {epoch + 1}: val_accuracy did not improve from "
                f"{self.best_val_accuracy:.5f}"
            )


def create_callbacks(log_append):
    metadata_path = MODEL_DIR / "face_best_metrics.json"
    return [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=4,
            restore_best_weights=True,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=2,
            min_lr=1e-6,
            verbose=1,
        ),
        GlobalBestModelCheckpoint(
            filepath=MODEL_DIR / "face_emotion_model_best.keras",
            metadata_path=metadata_path,
        ),
        BestMetricTracker(metadata_path=metadata_path),
        tf.keras.callbacks.CSVLogger(str(MODEL_DIR / "face_training_log.csv"), append=log_append),
    ]


def merge_histories(*histories):
    merged_history = {}
    for history in histories:
        for key, values in history.history.items():
            merged_history.setdefault(key, []).extend(values)

    merged = tf.keras.callbacks.History()
    merged.history = merged_history
    return merged


def plot_training_history(history):
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(history.history["accuracy"], label="Train Accuracy")
    plt.plot(history.history["val_accuracy"], label="Validation Accuracy")
    plt.title("Face Model Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history.history["loss"], label="Train Loss")
    plt.plot(history.history["val_loss"], label="Validation Loss")
    plt.title("Face Model Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()

    plt.tight_layout()
    plt.savefig(MODEL_DIR / "face_training_curves.png")
    plt.close()


def save_training_artifacts(final_model, history, class_names, train_counts, validation_counts, split_summary):
    final_model.save(MODEL_DIR / "face_emotion_model_final.keras")

    with open(MODEL_DIR / "face_class_names.json", "w", encoding="utf-8") as file:
        json.dump(class_names, file, indent=2)

    with open(MODEL_DIR / "face_train_class_counts.json", "w", encoding="utf-8") as file:
        json.dump(train_counts, file, indent=2)

    with open(MODEL_DIR / "face_validation_class_counts.json", "w", encoding="utf-8") as file:
        json.dump(validation_counts, file, indent=2)

    with open(MODEL_DIR / "face_split_summary.json", "w", encoding="utf-8") as file:
        json.dump(split_summary, file, indent=2)

    with open(MODEL_DIR / "face_training_history.json", "w", encoding="utf-8") as file:
        json.dump(history.history, file, indent=2)

    plot_training_history(history)


def reset_best_checkpoint_state():
    for path in [
        MODEL_DIR / "face_best_metrics.json",
        MODEL_DIR / "face_emotion_model_best.keras",
        MODEL_DIR / "face_emotion_model_phase1_best.keras",
        MODEL_DIR / "face_emotion_model_phase2_best.keras",
    ]:
        if path.exists():
            path.unlink()


def main():
    if not DATA_DIR.exists():
        raise FileNotFoundError(f"Processed faces directory not found: {DATA_DIR}")

    tf.keras.utils.set_random_seed(SEED)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    reset_best_checkpoint_state()

    records, class_names = collect_image_records(DATA_DIR)
    train_records, validation_records, total_videos, train_videos, validation_videos = split_records_by_video(records)
    train_counts = count_records_per_class(train_records, class_names)
    validation_counts = count_records_per_class(validation_records, class_names)
    class_weights = compute_class_weights(class_names, train_counts)

    print("Video-level split summary:")
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

    train_dataset = make_dataset(train_records, class_names, IMAGE_SIZE, BATCH_SIZE, training=True, shuffle_seed=SEED)
    validation_dataset = make_dataset(validation_records, class_names, IMAGE_SIZE, BATCH_SIZE, training=False)

    model, base_model = build_model(num_classes=len(class_names))
    compile_model(model, learning_rate=8e-4)
    model.summary()

    print("\nPhase 1: training classification head")
    initial_history = model.fit(
        train_dataset,
        validation_data=validation_dataset,
        epochs=INITIAL_EPOCHS,
        class_weight=class_weights,
        callbacks=create_callbacks(log_append=False),
        verbose=2,
    )

    if (MODEL_DIR / "face_emotion_model_best.keras").exists():
        shutil.copy2(
            MODEL_DIR / "face_emotion_model_best.keras",
            MODEL_DIR / "face_emotion_model_phase1_best.keras",
        )

    print("\nPhase 2: fine-tuning top MobileNetV2 layers")
    base_model.trainable = True
    for layer in base_model.layers[:-FINE_TUNE_LAYERS]:
        layer.trainable = False

    compile_model(model, learning_rate=5e-6)
    fine_tune_history = model.fit(
        train_dataset,
        validation_data=validation_dataset,
        initial_epoch=initial_history.epoch[-1] + 1,
        epochs=INITIAL_EPOCHS + FINE_TUNE_EPOCHS,
        class_weight=class_weights,
        callbacks=create_callbacks(log_append=True),
        verbose=2,
    )

    if (MODEL_DIR / "face_emotion_model_best.keras").exists():
        shutil.copy2(
            MODEL_DIR / "face_emotion_model_best.keras",
            MODEL_DIR / "face_emotion_model_phase2_best.keras",
        )

    full_history = merge_histories(initial_history, fine_tune_history)
    best_model = tf.keras.models.load_model(
        MODEL_DIR / "face_emotion_model_best.keras",
        custom_objects={"loss": sparse_categorical_focal_loss()},
        compile=False,
    )
    compile_model(best_model, learning_rate=5e-6)
    validation_loss, validation_accuracy = best_model.evaluate(validation_dataset, verbose=0)
    print(f"\nBest checkpoint validation loss: {validation_loss:.4f}")
    print(f"Best checkpoint validation accuracy: {validation_accuracy:.4f}")

    split_summary = {
        "split_type": "video_level",
        "image_size": list(IMAGE_SIZE),
        "batch_size": BATCH_SIZE,
        "initial_epochs": INITIAL_EPOCHS,
        "fine_tune_epochs": FINE_TUNE_EPOCHS,
        "fine_tune_layers": FINE_TUNE_LAYERS,
        "loss": "sparse_focal_loss",
        "focal_gamma": FOCAL_GAMMA,
        "total_source_videos": total_videos,
        "training_videos": train_videos,
        "validation_videos": validation_videos,
        "total_training_images": len(train_records),
        "total_validation_images": len(validation_records),
    }
    save_training_artifacts(model, full_history, class_names, train_counts, validation_counts, split_summary)
    print(f"\nTraining complete. Model artifacts saved in: {MODEL_DIR}")


if __name__ == "__main__":
    main()
