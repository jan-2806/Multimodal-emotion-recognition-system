import json
import re
from pathlib import Path

import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.model_selection import train_test_split

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "processed_faces"
MODEL_DIR = PROJECT_ROOT / "models"
IMAGE_SIZE = (96, 96)
BATCH_SIZE = 32
INITIAL_EPOCHS = 10
FINE_TUNE_EPOCHS = 6
VALIDATION_SPLIT = 0.2
SEED = 42
AUTOTUNE = tf.data.AUTOTUNE
FINE_TUNE_LAYERS = 30
IMAGE_PATTERN = re.compile(r"^(Actor_\d+)_(\d{2}(?:-\d{2}){6})_(\d+)$")


def parse_video_id(image_path):
    match = IMAGE_PATTERN.match(image_path.stem)
    if not match:
        raise ValueError(f"Unexpected processed face filename: {image_path.name}")

    actor_id = match.group(1)
    video_id = match.group(2)
    return f"{actor_id}__{video_id}"


def collect_image_records():
    class_names = sorted(path.name for path in DATA_DIR.iterdir() if path.is_dir())
    if not class_names:
        raise ValueError(f"No emotion folders found in: {DATA_DIR}")

    records = []
    for class_name in class_names:
        class_dir = DATA_DIR / class_name
        image_paths = sorted(path for path in class_dir.iterdir() if path.is_file())
        if not image_paths:
            raise ValueError(f"Emotion folder is empty: {class_dir}")

        for image_path in image_paths:
            records.append(
                {
                    "path": image_path,
                    "label": class_name,
                    "video_id": parse_video_id(image_path),
                }
            )

    return records, class_names


def split_records_by_video(records):
    video_to_label = {}
    for record in records:
        video_id = record["video_id"]
        label = record["label"]

        if video_id in video_to_label and video_to_label[video_id] != label:
            raise ValueError(f"Video {video_id} maps to multiple labels.")

        video_to_label[video_id] = label

    video_ids = sorted(video_to_label.keys())
    video_labels = [video_to_label[video_id] for video_id in video_ids]

    train_video_ids, validation_video_ids = train_test_split(
        video_ids,
        test_size=VALIDATION_SPLIT,
        random_state=SEED,
        stratify=video_labels,
    )

    train_video_ids = set(train_video_ids)
    validation_video_ids = set(validation_video_ids)

    train_records = [record for record in records if record["video_id"] in train_video_ids]
    validation_records = [record for record in records if record["video_id"] in validation_video_ids]
    return train_records, validation_records, len(video_ids), len(train_video_ids), len(validation_video_ids)


def count_records_per_class(records, class_names):
    counts = {class_name: 0 for class_name in class_names}
    for record in records:
        counts[record["label"]] += 1
    return counts


def compute_class_weights(class_names, class_counts):
    total_samples = sum(class_counts.values())
    num_classes = len(class_names)
    class_weights = {}

    for index, class_name in enumerate(class_names):
        count = class_counts[class_name]
        class_weights[index] = total_samples / (num_classes * count)

    return class_weights


def load_image(path, label):
    image = tf.io.read_file(path)
    image = tf.image.decode_jpeg(image, channels=1)
    image = tf.image.resize(image, IMAGE_SIZE)
    image = tf.cast(image, tf.float32)
    return image, label


def make_dataset(records, class_names, training):
    class_to_index = {class_name: index for index, class_name in enumerate(class_names)}
    paths = [str(record["path"]) for record in records]
    labels = [class_to_index[record["label"]] for record in records]

    dataset = tf.data.Dataset.from_tensor_slices((paths, labels))
    if training:
        dataset = dataset.shuffle(len(paths), seed=SEED, reshuffle_each_iteration=True)

    dataset = dataset.map(load_image, num_parallel_calls=AUTOTUNE)
    dataset = dataset.batch(BATCH_SIZE).prefetch(AUTOTUNE)
    return dataset


def build_model(num_classes):
    augmentation = tf.keras.Sequential(
        [
            tf.keras.layers.RandomFlip("horizontal"),
            tf.keras.layers.RandomRotation(0.08),
            tf.keras.layers.RandomZoom(0.12),
            tf.keras.layers.RandomTranslation(0.08, 0.08),
            tf.keras.layers.RandomContrast(0.1),
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
        128,
        activation="relu",
        kernel_regularizer=tf.keras.regularizers.l2(1e-4),
    )(dropped)
    normalized_dense = tf.keras.layers.BatchNormalization()(dense)
    dropped_dense = tf.keras.layers.Dropout(0.3)(normalized_dense)
    outputs = tf.keras.layers.Dense(num_classes, activation="softmax")(dropped_dense)

    model = tf.keras.Model(inputs=inputs, outputs=outputs, name="face_emotion_mobilenet")
    return model, base_model


def compile_model(model, learning_rate):
    try:
        loss = tf.keras.losses.SparseCategoricalCrossentropy(label_smoothing=0.05)
    except TypeError:
        print("SparseCategoricalCrossentropy label_smoothing is unavailable in this TensorFlow version. Using the standard sparse categorical loss instead.")
        loss = tf.keras.losses.SparseCategoricalCrossentropy()

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss=loss,
        metrics=["accuracy"],
    )


def create_callbacks(log_append):
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
        tf.keras.callbacks.ModelCheckpoint(
            filepath=str(MODEL_DIR / "face_emotion_model_best.keras"),
            monitor="val_accuracy",
            save_best_only=True,
            verbose=1,
        ),
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


def main():
    if not DATA_DIR.exists():
        raise FileNotFoundError(f"Processed faces directory not found: {DATA_DIR}")

    tf.keras.utils.set_random_seed(SEED)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    records, class_names = collect_image_records()
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

    train_dataset = make_dataset(train_records, class_names, training=True)
    validation_dataset = make_dataset(validation_records, class_names, training=False)

    model, base_model = build_model(num_classes=len(class_names))
    compile_model(model, learning_rate=1e-3)
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

    print("\nPhase 2: fine-tuning top MobileNetV2 layers")
    base_model.trainable = True
    for layer in base_model.layers[:-FINE_TUNE_LAYERS]:
        layer.trainable = False

    compile_model(model, learning_rate=1e-5)
    fine_tune_history = model.fit(
        train_dataset,
        validation_data=validation_dataset,
        initial_epoch=initial_history.epoch[-1] + 1,
        epochs=INITIAL_EPOCHS + FINE_TUNE_EPOCHS,
        class_weight=class_weights,
        callbacks=create_callbacks(log_append=True),
        verbose=2,
    )

    full_history = merge_histories(initial_history, fine_tune_history)
    best_model = tf.keras.models.load_model(MODEL_DIR / "face_emotion_model_best.keras")
    validation_loss, validation_accuracy = best_model.evaluate(validation_dataset, verbose=0)
    print(f"\nBest checkpoint validation loss: {validation_loss:.4f}")
    print(f"Best checkpoint validation accuracy: {validation_accuracy:.4f}")

    split_summary = {
        "split_type": "video_level",
        "image_size": list(IMAGE_SIZE),
        "batch_size": BATCH_SIZE,
        "initial_epochs": INITIAL_EPOCHS,
        "fine_tune_epochs": FINE_TUNE_EPOCHS,
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
