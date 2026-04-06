import csv
from pathlib import Path

import numpy as np

from audio_feature_utils import extract_audio_feature_tensor, parse_emotion_label

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASET_DIR = PROJECT_ROOT / "RAVDESS dataset"
OUTPUT_DIR = PROJECT_ROOT / "processed_audio"
METADATA_PATH = OUTPUT_DIR / "audio_feature_metadata.csv"
OVERWRITE_EXISTING = False


def iter_video_files():
    return sorted(DATASET_DIR.rglob("*.mp4"))


def build_output_path(video_path, emotion_label):
    actor_name = video_path.parent.name
    output_dir = OUTPUT_DIR / emotion_label
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir / f"{actor_name}_{video_path.stem}.npy"


def main():
    if not DATASET_DIR.exists():
        raise FileNotFoundError(f"Dataset directory not found: {DATASET_DIR}")

    video_files = iter_video_files()
    print(f"Found {len(video_files)} videos under {DATASET_DIR}")

    metadata_rows = []
    saved_count = 0
    skipped_count = 0

    for index, video_path in enumerate(video_files, start=1):
        try:
            emotion_label = parse_emotion_label(video_path.name)
            output_path = build_output_path(video_path, emotion_label)

            if output_path.exists() and not OVERWRITE_EXISTING:
                skipped_count += 1
            else:
                feature_tensor = extract_audio_feature_tensor(video_path)
                np.save(output_path, feature_tensor)
                saved_count += 1

            metadata_rows.append(
                {
                    "source_video": str(video_path),
                    "feature_path": str(output_path),
                    "emotion_label": emotion_label,
                    "actor_name": video_path.parent.name,
                    "video_stem": video_path.stem,
                }
            )

            if index % 100 == 0 or index == len(video_files):
                print(
                    f"Processed {index}/{len(video_files)} videos | "
                    f"saved={saved_count} | skipped_existing={skipped_count}"
                )
        except Exception as error:
            print(f"Skipping {video_path.name} due to error: {error}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(METADATA_PATH, "w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=["source_video", "feature_path", "emotion_label", "actor_name", "video_stem"],
        )
        writer.writeheader()
        writer.writerows(metadata_rows)

    print(f"Audio feature extraction complete. Metadata saved to: {METADATA_PATH}")


if __name__ == "__main__":
    main()
