from typing import Dict, List, Optional

from jiwer import wer


def _label_at_time(segments: List[Dict], time_s: float) -> Optional[str]:
    for seg in segments:
        if float(seg["start"]) <= time_s < float(seg["end"]):
            return str(seg["speaker"])
    return None


def compute_wer(pred: str, gt: str) -> float:
    return float(wer(gt, pred))


def compute_der_placeholder(
    pred_segments: Optional[List[Dict]] = None,
    gt_segments: Optional[List[Dict]] = None,
    frame_step: float = 0.1,
) -> float:
    """
    Lightweight DER approximation using frame-wise speaker mismatch.
    Returns -1.0 when reference segments are unavailable.
    """
    if not pred_segments or not gt_segments:
        return -1.0

    max_time = max(
        max(float(seg["end"]) for seg in pred_segments),
        max(float(seg["end"]) for seg in gt_segments),
    )

    if max_time <= 0:
        return 0.0

    total_frames = 0
    error_frames = 0
    current_time = 0.0
    while current_time < max_time:
        pred_label = _label_at_time(pred_segments, current_time)
        gt_label = _label_at_time(gt_segments, current_time)

        if gt_label is not None:
            total_frames += 1
            if pred_label != gt_label:
                error_frames += 1

        current_time += frame_step

    if total_frames == 0:
        return 0.0
    return error_frames / total_frames


def compute_speaker_accuracy(
    pred_segments: List[Dict], gt_segments: List[Dict], frame_step: float = 0.1
) -> float:
    if not pred_segments or not gt_segments:
        return 0.0

    max_time = max(
        max(float(seg["end"]) for seg in pred_segments),
        max(float(seg["end"]) for seg in gt_segments),
    )

    total_frames = 0
    correct_frames = 0
    current_time = 0.0
    while current_time < max_time:
        pred_label = _label_at_time(pred_segments, current_time)
        gt_label = _label_at_time(gt_segments, current_time)
        if gt_label is not None:
            total_frames += 1
            if pred_label == gt_label:
                correct_frames += 1
        current_time += frame_step

    if total_frames == 0:
        return 0.0
    return correct_frames / total_frames


def compute_rtf(processing_time_s: float, audio_duration_s: float) -> float:
    if audio_duration_s <= 0:
        return 0.0
    return processing_time_s / audio_duration_s
