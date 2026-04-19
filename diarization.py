from typing import Dict, List, Optional

from pyannote.audio import Pipeline

from config import HF_TOKEN

_DIARIZATION_PIPELINE: Optional[Pipeline] = None


def _get_pipeline(hf_token: Optional[str] = None) -> Pipeline:
    global _DIARIZATION_PIPELINE
    if _DIARIZATION_PIPELINE is None:
        token = hf_token if hf_token else HF_TOKEN
        _DIARIZATION_PIPELINE = Pipeline.from_pretrained(
            "pyannote/speaker-diarization-3.1",
            token=token if token else None,
        )
    return _DIARIZATION_PIPELINE


def _merge_small_segments(
    segments: List[Dict], min_duration: float = 0.5, gap_tolerance: float = 0.2
) -> List[Dict]:
    if not segments:
        return segments

    merged: List[Dict] = []
    for seg in segments:
        duration = seg["end"] - seg["start"]
        if (
            merged
            and duration < min_duration
            and merged[-1]["speaker"] == seg["speaker"]
            and seg["start"] - merged[-1]["end"] <= gap_tolerance
        ):
            merged[-1]["end"] = max(merged[-1]["end"], seg["end"])
        else:
            merged.append(seg.copy())

    return merged


def run_diarization(audio_path: str, hf_token: Optional[str] = None) -> List[Dict]:
    """
    Returns:
    [
      {"start": float, "end": float, "speaker": "SPEAKER_1"},
      ...
    ]
    """
    pipeline = _get_pipeline(hf_token=hf_token)
    diarization = pipeline(audio_path)

    speaker_map: Dict[str, str] = {}
    speaker_id = 1
    segments: List[Dict] = []

    for turn, _, speaker in diarization.speaker_diarization.itertracks(yield_label=True):
        if speaker not in speaker_map:
            speaker_map[speaker] = f"SPEAKER_{speaker_id}"
            speaker_id += 1

        segments.append(
            {
                "start": float(turn.start),
                "end": float(turn.end),
                "speaker": speaker_map[speaker],
            }
        )

    # Keep overlap behavior minimal by preserving all tracks,
    # then only merging tiny same-speaker micro-segments.
    segments = sorted(segments, key=lambda x: (x["start"], x["end"]))
    return _merge_small_segments(segments)
