"""Audio decoding and small file helpers shared by the pipeline."""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np

_CHUNK_SIZE = 1024 * 1024
SAMPLE_RATE = 16000


def load_audio_16k(path: Path) -> np.ndarray:
    """Decode any audio file (m4a/aac, wav, mp3, opus, mp4 audio) to float32 mono 16 kHz."""
    from faster_whisper.audio import decode_audio

    return decode_audio(str(path), sampling_rate=SAMPLE_RATE)


def duration_ms(samples: np.ndarray) -> int:
    """Duration of a mono 16 kHz float array, in milliseconds."""
    return int(round(len(samples) * 1000 / SAMPLE_RATE))


def sha256_file(path: Path) -> str:
    """Streamed SHA-256 of a file, reading in 1 MiB chunks."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as f:
        while chunk := f.read(_CHUNK_SIZE):
            digest.update(chunk)
    return digest.hexdigest()
