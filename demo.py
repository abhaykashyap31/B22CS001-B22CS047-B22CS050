import os

from config import OUTPUT_DIR
from main import run_pipeline
from utils import format_timestamp, save_transcript, speaker_display_name


def run_demo() -> None:
    audio_path = "sample.wav"  # Replace with your uploaded/local file path.
    output_path = os.path.join(OUTPUT_DIR, "demo_result.txt")
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    results = run_pipeline(audio_path)
    save_transcript(results, output_path)

    print("\n=== Transcript ===")
    for item in results:
        print(
            f"[{format_timestamp(item['start'])} - {format_timestamp(item['end'])}] "
            f"Speaker {speaker_display_name(item['speaker'])}: {item['text']}"
        )
    print(f"\nSaved demo transcript to: {output_path}")


if __name__ == "__main__":
    run_demo()
