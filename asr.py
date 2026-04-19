from typing import Dict, List, Optional

import torch
import whisper
from whisper import Whisper

from config import DEVICE, WHISPER_MODEL
from data import load_audio, preprocess_audio
from utils import slice_audio

_WHISPER_MODEL: Optional[Whisper] = None


def _get_model() -> Whisper:
    global _WHISPER_MODEL
    if _WHISPER_MODEL is None:
        _WHISPER_MODEL = whisper.load_model(WHISPER_MODEL, device=DEVICE)
    return _WHISPER_MODEL


def _extract_asr_confidence(output: Dict) -> Optional[float]:
    segment_scores = [
        seg.get("avg_logprob")
        for seg in output.get("segments", [])
        if seg.get("avg_logprob") is not None
    ]
    if not segment_scores:
        return None

    mean_logprob = sum(segment_scores) / len(segment_scores)
    confidence = float(min(1.0, max(0.0, torch.exp(torch.tensor(mean_logprob)).item())))
    return confidence


def transcribe_segments(audio_path: str, segments: List[Dict], apply_denoise: bool = False) -> List[Dict]:
    model = _get_model()
    audio, sr = load_audio(audio_path)
    audio, sr = preprocess_audio(audio, sr, apply_denoise=apply_denoise)

    results: List[Dict] = []
    with torch.no_grad():
        for seg in segments:
            start = float(seg["start"])
            end = float(seg["end"])
            speaker = seg["speaker"]

            segment_audio = slice_audio(audio, sr, start, end)
            if len(segment_audio) < int(0.1 * sr):
                text = ""
                confidence = None
            else:
                out = model.transcribe(
                    audio=segment_audio,
                    fp16=(DEVICE == "cuda"),
                    verbose=False,
                    task="transcribe",
                    language="en"
                )
                text = out.get("text", "").strip()
                confidence = _extract_asr_confidence(out)

            results.append(
                {
                    "start": start,
                    "end": end,
                    "speaker": speaker,
                    "text": text,
                    "confidence": confidence,
                }
            )

    return results
