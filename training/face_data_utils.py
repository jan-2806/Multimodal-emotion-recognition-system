import re
from pathlib import Path

import tensorflow as tf
from sklearn.model_selection import train_test_split

SEED = 42
VALIDATION_SPLIT = 0.2
IMAGE_PATTERN = re.compile(r"^(Actor_\d+)_(\d{2}(?:-\d{2}){6})_(\d+)$")


def parse_video_id(image_path):
    match = IMAGE_PATTERN.match(image_path.stem)
    if not match:
        raise ValueError(f"Unexpected processed face filename: {image_path.name}")

    actor_id = match.group(1)
    video_id = match.group(2)
    return f"{actor_id}__{video_id}"


def collect_image_records(data_dir):
    class_names = sorted(path.name for path in data_dir.iterdir() if path.is_dir())
    if not class_names:
        raise ValueError(f"No emotion folders found in: {data_dir}")

    records = []
    for class_name in class_names:
        class_dir = data_dir / class_name
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


def split_records_by_video(records, validation_split=VALIDATION_SPLIT, seed=SEED):
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
        test_size=validation_split,
        random_state=seed,
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


def make_dataset(records, class_names, image_size, batch_size, training=False, shuffle_seed=SEED):
    class_to_index = {class_name: index for index, class_name in enumerate(class_names)}
    paths = [str(record["path"]) for record in records]
    labels = [class_to_index[record["label"]] for record in records]

    dataset = tf.data.Dataset.from_tensor_slices((paths, labels))
    if training:
        dataset = dataset.shuffle(len(paths), seed=shuffle_seed, reshuffle_each_iteration=True)

    dataset = dataset.map(
        lambda path, label: load_image(path, label, image_size),
        num_parallel_calls=tf.data.AUTOTUNE,
    )
    dataset = dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return dataset


def load_image(path, label, image_size):
    image = tf.io.read_file(path)
    image = tf.image.decode_jpeg(image, channels=1)
    image = tf.image.resize(image, image_size)
    image = tf.cast(image, tf.float32)
    return image, label
