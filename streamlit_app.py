import os
from pathlib import Path

import streamlit as st

from config import HF_TOKEN, OUTPUT_DIR
from main import run_pipeline_with_timing
from utils import format_timestamp, speaker_display_name


def save_uploaded_audio(uploaded_file) -> str:
    output_dir = Path(OUTPUT_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)
    temp_path = output_dir / uploaded_file.name
    with open(temp_path, "wb") as f:
        f.write(uploaded_file.read())
    return str(temp_path)


def main() -> None:
    st.set_page_config(page_title="Meeting Diarization Demo", layout="wide")

    st.title("Meeting Diarization & Transcription Demo")
    st.markdown(
        "Upload an audio file and run the shared project pipeline to get speaker-attributed transcripts and stage timing."
    )

    uploaded_audio = st.file_uploader(
        "Upload audio file",
        type=["wav", "mp3", "m4a", "flac", "ogg"],
        help="Supported audio formats: WAV, MP3, M4A, FLAC, OGG.",
    )

    hf_token = st.text_input(
        "Hugging Face token (optional)",
        value=HF_TOKEN,
        type="password",
        help="Optional token for pyannote speaker diarization if your model requires authentication.",
    )

    if uploaded_audio is not None:
        audio_path = save_uploaded_audio(uploaded_audio)
        st.success(f"Saved uploaded file to `{audio_path}`")

        if st.button("Run diarization & transcription"):
            with st.spinner("Running pipeline..."):
                results, timings = run_pipeline_with_timing(audio_path, hf_token=hf_token)

            st.subheader("Transcript")
            for item in results:
                st.write(
                    f"[{format_timestamp(item['start'])} - {format_timestamp(item['end'])}] "
                    f"Speaker {speaker_display_name(item['speaker'])}: {item['text']}"
                )

            st.subheader("Stage timings")
            st.write(timings)

            transcript_path = os.path.join(OUTPUT_DIR, "streamlit_transcript.txt")
            with open(transcript_path, "w", encoding="utf-8") as f:
                for item in results:
                    f.write(
                        f"[{format_timestamp(item['start'])} - {format_timestamp(item['end'])}] "
                        f"Speaker {speaker_display_name(item['speaker'])}: {item['text']}\n"
                    )

            st.success(f"Transcript saved to `{transcript_path}`")
            st.info("You can rerun the pipeline with a different audio upload or token.")

    else:
        st.info("Upload an audio file to begin.")


if __name__ == "__main__":
    main()
