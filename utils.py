from typing import Dict, List

import numpy as np


def format_timestamp(seconds: float) -> str:
    total_seconds = int(max(0, round(seconds)))
    hours = total_seconds // 3600
    minutes = (total_seconds % 3600) // 60
    secs = total_seconds % 60
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


def speaker_display_name(speaker_label: str) -> str:
    if isinstance(speaker_label, str) and speaker_label.startswith("SPEAKER_"):
        return speaker_label.split("_")[-1]
    return str(speaker_label)


def slice_audio(audio: np.ndarray, sr: int, start: float, end: float) -> np.ndarray:
    start_idx = max(0, int(start * sr))
    end_idx = max(start_idx, int(end * sr))
    return audio[start_idx:end_idx]


def save_transcript(lines: List[Dict], path: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for item in lines:
            line = (
                f"[{format_timestamp(item['start'])} - {format_timestamp(item['end'])}] "
                f"Speaker {speaker_display_name(item['speaker'])}: {item['text']}"
            )
            f.write(line.strip() + "\n")
