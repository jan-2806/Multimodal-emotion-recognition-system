from functools import lru_cache
from pathlib import Path

import numpy as np

SAMPLE_RATE = 16000
CLIP_DURATION_SECONDS = 3.0
TARGET_AUDIO_SAMPLES = int(SAMPLE_RATE * CLIP_DURATION_SECONDS)
YAMNET_HANDLE = "https://tfhub.dev/google/yamnet/1"
EMBEDDING_SIZE = 1024
FEATURE_SIZE = EMBEDDING_SIZE * 3

EMOTION_MAP = {
    "01": "neutral",
    "02": "calm",
    "03": "happy",
    "04": "sad",
    "05": "angry",
    "06": "fear",
    "07": "disgust",
    "08": "surprise",
}


def get_tfhub():
    try:
        import tensorflow_hub as hub
    except ImportError as error:
        raise ImportError(
            "tensorflow-hub is required for the pretrained audio pipeline. "
            "Install it with: python -m pip install tensorflow-hub"
        ) from error
    return hub


def parse_emotion_label(file_name):
    parts = Path(file_name).stem.split("-")
    if len(parts) < 3:
        raise ValueError(f"Unexpected RAVDESS filename: {file_name}")

    emotion_code = parts[2]
    if emotion_code not in EMOTION_MAP:
        raise ValueError(f"Unknown emotion code '{emotion_code}' in {file_name}")

    return EMOTION_MAP[emotion_code]


def load_audio_from_media(media_path, target_sr=SAMPLE_RATE):
    errors = []

    try:
        signal = load_audio_with_moviepy(media_path, target_sr)
        return signal, target_sr
    except Exception as error:
        errors.append(f"moviepy failed: {error}")

    try:
        librosa = get_librosa()
        signal, sample_rate = librosa.load(str(media_path), sr=target_sr, mono=True)
        return signal.astype(np.float32), sample_rate
    except Exception as error:
        errors.append(f"librosa failed: {error}")

    raise RuntimeError(
        f"Could not load audio from {media_path}. "
        f"Tried moviepy and librosa. Details: {' | '.join(errors)}"
    )


def get_librosa():
    try:
        import librosa
    except ImportError as error:
        raise ImportError(
            "librosa is required for the audio pipeline. Install it with: "
            "python -m pip install librosa soundfile"
        ) from error
    return librosa


def load_audio_with_moviepy(media_path, target_sr=SAMPLE_RATE):
    try:
        from moviepy.video.io.VideoFileClip import VideoFileClip
    except ImportError:
        from moviepy.editor import VideoFileClip

    clip = VideoFileClip(str(media_path))
    try:
        if clip.audio is None:
            raise RuntimeError("Video does not contain an audio track.")

        audio_array = clip.audio.to_soundarray(fps=target_sr)
        if audio_array.ndim == 2:
            audio_array = audio_array.mean(axis=1)
        return audio_array.astype(np.float32)
    finally:
        if clip.audio is not None:
            clip.audio.close()
        clip.close()


def pad_or_trim_signal(signal, target_length=TARGET_AUDIO_SAMPLES):
    if len(signal) >= target_length:
        start_index = (len(signal) - target_length) // 2
        return signal[start_index:start_index + target_length]

    padding = target_length - len(signal)
    pad_left = padding // 2
    pad_right = padding - pad_left
    return np.pad(signal, (pad_left, pad_right), mode="constant")


@lru_cache(maxsize=1)
def get_yamnet_model():
    hub = get_tfhub()
    return hub.load(YAMNET_HANDLE)


def extract_audio_feature_tensor(media_path):
    signal, _ = load_audio_from_media(media_path, target_sr=SAMPLE_RATE)
    return extract_audio_feature_from_signal(signal)


def extract_audio_feature_from_signal(signal):
    signal = pad_or_trim_signal(signal)
    signal = np.asarray(signal, dtype=np.float32)

    max_abs = float(np.max(np.abs(signal)))
    if max_abs > 0:
        signal = signal / max_abs

    yamnet_model = get_yamnet_model()
    _, embeddings, _ = yamnet_model(signal)
    embeddings = embeddings.numpy().astype(np.float32)

    embedding_mean = embeddings.mean(axis=0)
    embedding_std = embeddings.std(axis=0)
    embedding_max = embeddings.max(axis=0)
    feature = np.concatenate([embedding_mean, embedding_std, embedding_max], axis=0).astype(np.float32)
    return feature
