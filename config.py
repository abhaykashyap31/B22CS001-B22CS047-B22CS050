import os
from dotenv import load_dotenv

load_dotenv()

SAMPLE_RATE = 16000
# Keep the Python modules CPU-friendly by default.
# The Colab notebook is the only intended GPU execution path.
DEVICE = "cpu"
WHISPER_MODEL = "small"
OUTPUT_DIR = "outputs/"
HF_TOKEN = os.getenv('HF_TOKEN', '')  # Load from .env file

