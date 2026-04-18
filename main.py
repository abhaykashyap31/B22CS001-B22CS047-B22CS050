import argparse
import os
import time
from typing import Dict, List

from asr import transcribe_segments
from config import OUTPUT_DIR
from data import load_audio, preprocess_audio
from diarization import run_diarization
from utils import format_timestamp, save_transcript, speaker_display_name


def _format_lines(results: List[Dict]) -> List[str]:
    lines = []
    for item in results:
        line = (
            f"[{format_timestamp(item['start'])} - {format_timestamp(item['end'])}] "
            f"Speaker {speaker_display_name(item['speaker'])}: {item['text']}"
        )
        lines.append(line)
    return lines


def run_pipeline(audio_path: str, hf_token: str = "") -> List[Dict]:
    results, _ = run_pipeline_with_timing(audio_path, hf_token=hf_token)
    return results


def run_pipeline_with_timing(audio_path: str, hf_token: str = "") -> tuple[List[Dict], Dict[str, float]]:
    timings: Dict[str, float] = {}

    t0 = time.perf_counter()
    # 1) Load
    audio, sr = load_audio(audio_path)
    timings["load_audio"] = time.perf_counter() - t0

    t1 = time.perf_counter()
    # 2) Preprocess
    _audio, _sr = preprocess_audio(audio, sr)
    # Keep pipeline simple: diarization/asr modules re-load and preprocess internally.
    # This step preserves explicit end-to-end stage structure.
    del _audio, _sr
    timings["preprocess"] = time.perf_counter() - t1

    # 3) Diarization
    t2 = time.perf_counter()
    segments = run_diarization(audio_path, hf_token=hf_token if hf_token else None)
    timings["diarization"] = time.perf_counter() - t2
    # 4) Segment-wise ASR + speaker attribution
    t3 = time.perf_counter()
    results = transcribe_segments(audio_path, segments)
    timings["asr"] = time.perf_counter() - t3
    timings["total"] = sum(
        timings[key] for key in ["load_audio", "preprocess", "diarization", "asr"]
    )
    return results, timings


def main() -> None:
    parser = argparse.ArgumentParser(description="Meeting Transcription & Speaker Diarization")
    parser.add_argument("--audio", type=str, required=True, help="Path to input WAV/Audio file")
    parser.add_argument(
        "--hf-token",
        type=str,
        default="",
        help="Optional Hugging Face token for pyannote pipeline",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=os.path.join(OUTPUT_DIR, "result.txt"),
        help="Output transcript path",
    )
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    results, timings = run_pipeline_with_timing(args.audio, hf_token=args.hf_token)
    save_transcript(results, args.output)

    for line in _format_lines(results):
        print(line)

    print(f"\nSaved transcript to: {args.output}")
    print(f"Stage timings (s): {timings}")


if __name__ == "__main__":
    main()
