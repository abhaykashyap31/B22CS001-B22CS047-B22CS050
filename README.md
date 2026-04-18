# Meeting Transcription & Speaker Diarization

End-to-end pipeline:

Audio -> Preprocessing -> Speaker Diarization -> Segment-wise ASR -> Speaker Attribution -> Final Transcript

The repository has two usage modes:

- `main.py` and the Python modules provide a simple CPU-friendly inference and evaluation path for demo purposes on any user audio file.
- `notebooks/demo.ipynb` is the primary GPU path: a fully automated AMI-only Colab demo that downloads a tiny AMI subset, runs Whisper `large-v2`, performs speaker attribution, computes metrics, and generates visualizations.

## Features

- Audio preprocessing to 16 kHz mono
- Optional simple spectral-subtraction denoising
- Speaker diarization with `pyannote.audio`
- Segment-wise transcription with Whisper `small` locally and Whisper `large-v2` in the notebook
- Notebook-only speaker attribution using either an MLP or cosine similarity
- Timestamped speaker-attributed transcript output
- `WER`, approximate `DER`, speaker accuracy, latency, and `RTF`
- Speaker-turn timeline visualization in the notebook

## Project Structure

```text
project/
├── main.py
├── config.py
├── data.py
├── diarization.py
├── asr.py
├── utils.py
├── evaluation.py
├── demo.py
├── requirements.txt
├── README.md
├── progress.md
├── notebooks/
│   └── demo.ipynb
└── outputs/
```

## Requirements

- Python 3.10+ recommended
- GPU recommended for the notebook
- Hugging Face token required for gated `pyannote` models

Install local dependencies:

```bash
pip install -r requirements.txt
```

## Hugging Face Token

The notebook and local pipeline both use gated `pyannote` pretrained models. You may need:

1. A Hugging Face account
2. Accepted model terms for the required `pyannote` models
3. A valid token

Local usage options:

- set `HF_TOKEN` in `config.py`
- pass `--hf-token` to `main.py`

Colab usage:

- set `os.environ["HF_TOKEN"] = "hf_..."` before the model-loading cell if the notebook cannot download `pyannote` models

## Local Usage

The local code is intended for demo/support usage and defaults to CPU.

Run the full pipeline on your own audio:

```bash
python3 main.py --audio sample.wav
```

Custom output path:

```bash
python3 main.py --audio sample.wav --output outputs/my_result.txt
```

Run the simple local demo:

```bash
python3 demo.py
```

`demo.py` expects `sample.wav` unless you edit the path.

Run local evaluation with optional references:

```bash
python3 evaluate.py --audio sample.wav --reference-text ref.txt --reference-segments ref_segments.json
```

Expected reference segment format:

```json
[
  {"start": 0.0, "end": 1.8, "speaker": "SPEAKER_1"},
  {"start": 1.8, "end": 3.5, "speaker": "SPEAKER_2"}
]
```

## Colab Notebook

Open `notebooks/demo.ipynb` in Colab and run top to bottom.

The notebook is now fully automated and AMI-only:

1. Installs dependencies
2. Downloads a tiny AMI subset via `datasets`
3. Selects a few AMI meetings for training and one for demo inference
4. Reconstructs an AMI meeting excerpt into a WAV file
5. Loads `pyannote` diarization, `pyannote` embeddings, and Whisper
6. Prepares speaker attribution from AMI embeddings using either an MLP or cosine similarity
7. Runs diarization on the reconstructed AMI excerpt
8. Runs segment-wise Whisper transcription
9. Predicts consistent speaker identities
10. Computes `WER`, approximate `DER`, speaker accuracy, latency, and `RTF`
11. Saves transcript, metrics, debug outputs, and a speaker timeline plot
12. Plays the reconstructed AMI audio

No notebook file upload is required.

Recommended Colab setup:

1. Open in Google Colab
2. Set runtime to `T4 GPU`
3. If needed, set `os.environ["HF_TOKEN"] = "hf_..."` before model loading
4. Run all cells once from top to bottom

## Notebook Attribution Modes

The notebook supports two speaker-attribution approaches:

- `mlp`: notebook-only lightweight classifier trained on AMI speaker embeddings
- `cosine`: cosine-similarity speaker prototypes built from AMI embeddings

You can switch this with:

```python
ATTRIBUTION_METHOD = "mlp"  # or "cosine"
```

The notebook does not fine-tune:

- Whisper
- `pyannote.audio` diarization
- the embedding backbone

This keeps the notebook fast and aligned with the project goal of using pretrained core models.

## Output Files

Local CLI output:

- `outputs/result.txt`

Notebook outputs:

- `outputs/result.txt`
- `outputs/attribution_debug.txt`
- `outputs/metrics.json`
- `outputs/speaker_timeline.png`
- reconstructed AMI demo WAV in `outputs/`

Transcript format:

```text
[00:00:00 - 00:00:05] Speaker 1: Hello everyone
[00:00:05 - 00:00:10] Speaker 2: Let's begin the meeting
```

## Expected Runtime on T4 GPU

Notebook on Colab T4 GPU:

- first run: roughly `5-15+ minutes`
- rerun in same runtime after model caching: often `3-8 minutes`

This includes:

- dependency install time
- AMI streaming/cache time
- first-time `pyannote` and Whisper downloads
- diarization
- segment-wise Whisper `large-v2`
- metrics and visualization

Local machine CPU/demo path:

- for a ~5 minute audio file, expect roughly `5-20+ minutes` depending on CPU
- local runtime varies a lot more than Colab because diarization and Whisper on CPU can be slow
- the local path uses Whisper `small`, so it is lighter than the notebook but still not optimized for large batch evaluation

General factors affecting runtime:

- number of speakers
- number of diarized segments
- overlap and noise level
- first-time model download/cache cost
- whether denoising is enabled

## Dataset Strategy

Local Python project:

- primary mode is user-provided audio
- no automatic dataset download

Notebook:

- strictly uses AMI only
- downloads only a small streamed subset needed for the demo
- does not download the full AMI corpus

## Evaluation

`evaluation.py` currently includes:

- `compute_wer(pred, gt)` using `jiwer`
- approximate `DER`
- speaker accuracy
- `RTF`

`evaluate.py` provides a simple experiment-style evaluation entrypoint for local runs.

The notebook reports:

- `WER`
- approximate `DER`
- speaker attribution accuracy
- stage timings
- total runtime
- `RTF`

## Current Limitations

- preprocessing is basic normalization and resampling only
- denoising is only a simple spectral-subtraction option
- local pipeline does not train a speaker-attribution model
- LibriSpeech-specific evaluation pipeline is not implemented
- notebook reconstruction uses AMI utterance chunks rather than original raw full-session audio
- notebook metric computation is lightweight and approximate, not a full benchmark protocol

## Troubleshooting

If `pyannote` fails to load:

- verify your Hugging Face token
- accept the model terms on Hugging Face
- restart the Colab runtime and rerun

If Colab is slow:

- enable `T4 GPU`
- rerun after first-time model downloads complete
- keep using the same runtime so models remain cached

If diarization produces many short turns:

- this is expected on overlapping or noisy speech
- the code already merges some very small same-speaker segments
