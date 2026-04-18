from typing import Tuple

import librosa
import numpy as np

from config import SAMPLE_RATE


def load_audio(path: str) -> Tuple[np.ndarray, int]:
    """Load audio file with original sample rate."""
    audio, sr = librosa.load(path, sr=None, mono=True)
    return audio.astype(np.float32), sr


def spectral_subtract_denoise(
    audio: np.ndarray, sr: int, noise_seconds: float = 0.5
) -> np.ndarray:
    """Simple spectral subtraction using the first few frames as noise profile."""
    if len(audio) == 0:
        return audio.astype(np.float32)

    n_fft = 512
    hop_length = 128
    stft = librosa.stft(audio, n_fft=n_fft, hop_length=hop_length)
    magnitude = np.abs(stft)
    phase = np.angle(stft)

    noise_frames = max(1, int((noise_seconds * sr) / hop_length))
    noise_profile = np.mean(magnitude[:, :noise_frames], axis=1, keepdims=True)
    cleaned_magnitude = np.maximum(magnitude - noise_profile, 0.0)
    cleaned_stft = cleaned_magnitude * np.exp(1j * phase)
    cleaned = librosa.istft(cleaned_stft, hop_length=hop_length, length=len(audio))
    return cleaned.astype(np.float32)


def preprocess_audio(
    audio: np.ndarray, sr: int, apply_denoise: bool = False
) -> Tuple[np.ndarray, int]:
    """Resample to 16kHz, optionally denoise, and normalize waveform."""
    if sr != SAMPLE_RATE:
        audio = librosa.resample(audio, orig_sr=sr, target_sr=SAMPLE_RATE)
        sr = SAMPLE_RATE

    if apply_denoise:
        audio = spectral_subtract_denoise(audio, sr)

    peak = np.max(np.abs(audio)) + 1e-9
    audio = (audio / peak).astype(np.float32)
    return audio, sr
