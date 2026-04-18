import argparse
import json
import os
from typing import Dict, List, Optional

from data import load_audio
from evaluation import compute_der_placeholder, compute_rtf, compute_speaker_accuracy, compute_wer
from main import run_pipeline_with_timing


def _load_reference_text(path: Optional[str]) -> Optional[str]:
    if not path:
        return None
    with open(path, "r", encoding="utf-8") as f:
        return f.read().strip()


def _load_reference_segments(path: Optional[str]) -> Optional[List[Dict]]:
    if not path:
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _join_predicted_text(results: List[Dict]) -> str:
    return " ".join(item.get("text", "").strip() for item in results if item.get("text")).strip()


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate meeting transcription pipeline.")
    parser.add_argument("--audio", required=True, help="Path to input audio file")
    parser.add_argument("--hf-token", default="", help="Optional Hugging Face token")
    parser.add_argument("--reference-text", default="", help="Optional reference transcript text file")
    parser.add_argument(
        "--reference-segments",
        default="",
        help="Optional JSON file with [{start, end, speaker}, ...] reference segments",
    )
    parser.add_argument(
        "--output-json",
        default=os.path.join("outputs", "evaluation_metrics.json"),
        help="Path to save computed metrics",
    )
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.output_json), exist_ok=True)

    audio, sr = load_audio(args.audio)
    audio_duration_s = len(audio) / float(sr)

    results, timings = run_pipeline_with_timing(args.audio, hf_token=args.hf_token)
    pred_text = _join_predicted_text(results)
    reference_text = _load_reference_text(args.reference_text)
    reference_segments = _load_reference_segments(args.reference_segments)

    pred_segments = [
        {"start": item["start"], "end": item["end"], "speaker": item["speaker"]}
        for item in results
    ]

    metrics = {
        "audio_duration_s": audio_duration_s,
        "timings_s": timings,
        "rtf_total": compute_rtf(timings["total"], audio_duration_s),
        "wer": compute_wer(pred_text, reference_text) if reference_text else None,
        "der_approx": compute_der_placeholder(pred_segments, reference_segments),
        "speaker_accuracy": (
            compute_speaker_accuracy(pred_segments, reference_segments)
            if reference_segments
            else None
        ),
    }

    with open(args.output_json, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    print(json.dumps(metrics, indent=2))
    print(f"\nSaved evaluation metrics to: {args.output_json}")


if __name__ == "__main__":
    main()
