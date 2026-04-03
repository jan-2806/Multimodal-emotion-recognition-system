import os
from pathlib import Path

import cv2

# Absolute path to dataset (recommended)
DATASET_PATH = Path(r"C:\Users\JANANI\Documents\NNDL projects\Emotion Recognition\RAVDESS dataset")

# Where processed faces will be saved
OUTPUT_PATH = Path(r"C:\Users\JANANI\Documents\NNDL projects\Emotion Recognition\processed_faces")

# Sampling and quality settings tuned for faster extraction with less duplication.
FRAME_STEP = 5
MAX_FACES_PER_VIDEO = 15
FACE_MARGIN = 0.15
MIN_FACE_AREA_RATIO = 0.015
BLUR_THRESHOLD = 20.0

# Emotion label mapping (RAVDESS codes)
emotion_dict = {
    "01": "neutral",
    "02": "calm",
    "03": "happy",
    "04": "sad",
    "05": "angry",
    "06": "fear",
    "07": "disgust",
    "08": "surprise",
}

# Try a few OpenCV cascades so we are not blocked by one detector profile.
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


def detect_best_face(gray_frame):
    """Return the largest detected face using a few detector settings."""
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


def ensure_output_folders():
    for emotion in emotion_dict.values():
        os.makedirs(OUTPUT_PATH / emotion, exist_ok=True)


def expand_face_box(face_box, frame_shape):
    x, y, w, h = face_box
    frame_height, frame_width = frame_shape
    pad_w = int(w * FACE_MARGIN)
    pad_h = int(h * FACE_MARGIN)

    x1 = max(0, x - pad_w)
    y1 = max(0, y - pad_h)
    x2 = min(frame_width, x + w + pad_w)
    y2 = min(frame_height, y + h + pad_h)
    return x1, y1, x2, y2


def is_low_quality_face(face_crop, frame_shape):
    frame_height, frame_width = frame_shape
    face_height, face_width = face_crop.shape
    face_area_ratio = (face_width * face_height) / (frame_width * frame_height)

    if face_area_ratio < MIN_FACE_AREA_RATIO:
        return True

    blur_score = cv2.Laplacian(face_crop, cv2.CV_64F).var()
    return blur_score < BLUR_THRESHOLD


def is_duplicate_face(current_face, previous_face):
    if previous_face is None:
        return False

    difference = cv2.absdiff(current_face, previous_face)
    return float(difference.mean()) < 4.0


def process_video(video_path):
    parts = video_path.stem.split("-")
    if len(parts) < 3:
        print(f"Skipping invalid filename: {video_path.name}")
        return 0, 0

    emotion_code = parts[2]
    if emotion_code not in emotion_dict:
        print(f"Skipping unknown emotion code in: {video_path.name}")
        return 0, 0

    emotion_label = emotion_dict[emotion_code]
    actor_name = video_path.parent.name
    cap = cv2.VideoCapture(str(video_path))

    if not cap.isOpened():
        print(f"Could not open video: {video_path}")
        return 0, 0

    saved_faces = 0
    frames_read = 0
    sampled_frames = 0
    previous_saved_face = None

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frames_read += 1
        sampled_frames += 1

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        face_box = detect_best_face(gray)

        if face_box is not None:
            x1, y1, x2, y2 = expand_face_box(face_box, gray.shape)
            face = gray[y1:y2, x1:x2]

            if face.size != 0 and not is_low_quality_face(face, gray.shape):
                face = cv2.equalizeHist(face)
                face = cv2.resize(face, (48, 48))

                if not is_duplicate_face(face, previous_saved_face):
                    save_path = OUTPUT_PATH / emotion_label / f"{actor_name}_{video_path.stem}_{saved_faces:04d}.jpg"
                    cv2.imwrite(str(save_path), face)
                    previous_saved_face = face
                    saved_faces += 1

                    if saved_faces >= MAX_FACES_PER_VIDEO:
                        break

        for _ in range(FRAME_STEP - 1):
            if not cap.grab():
                ret = False
                break
            frames_read += 1

        if not ret:
            break

    cap.release()
    print(
        f"{video_path.name}: frames_read={frames_read}, sampled_frames={sampled_frames}, "
        f"faces_saved={saved_faces}"
    )
    return frames_read, saved_faces


def main():
    ensure_output_folders()

    video_files = sorted(DATASET_PATH.rglob("*.mp4"))
    print(f"Found {len(video_files)} videos under {DATASET_PATH}")

    total_frames = 0
    total_faces = 0

    for video_path in video_files:
        frames_read, faces_saved = process_video(video_path)
        total_frames += frames_read
        total_faces += faces_saved

    print(f"Face extraction completed! Total frames read: {total_frames}, total faces saved: {total_faces}")


if __name__ == "__main__":
    main()
