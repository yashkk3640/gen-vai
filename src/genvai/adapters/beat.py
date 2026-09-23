"""BeatDetector over ffmpeg decoding plus the pure analysis in `genvai.beats`.

ffmpeg turns any audio file into mono float samples at a fixed rate; everything after
that is arithmetic. Decoding straight to a pipe avoids writing a temporary WAV, which
for a five-minute track would be fifty megabytes for nothing.
"""

import subprocess
from pathlib import Path

import numpy as np

from genvai.beats import ANALYSIS_SECONDS, detect
from genvai.timeline import BeatMap

SAMPLE_RATE = 22050
"""Enough for onset detection. Percussive energy lives well below 11 kHz, and halving
the rate halves the work."""


class FFmpegBeatDetector:
    """Implements `BeatDetector`."""

    name = "ffmpeg"

    def __init__(self, ffmpeg: Path) -> None:
        self._ffmpeg = ffmpeg

    def is_available(self) -> bool:
        return self._ffmpeg.exists()

    def detect(self, audio: Path) -> BeatMap:
        """Read a track and return its tempo and beat grid.

        An unreadable file yields an empty map rather than raising: music is a garnish
        on the cut, and a track that cannot be analysed should cost beat alignment, not
        the render.
        """
        samples = self._decode(audio)
        if samples.size == 0:
            return BeatMap(bpm=120.0)
        return detect(samples, SAMPLE_RATE, samples.size / SAMPLE_RATE)

    def _decode(self, audio: Path) -> np.ndarray:
        command = [
            str(self._ffmpeg),
            "-hide_banner",
            "-loglevel",
            "error",
            "-t",
            str(ANALYSIS_SECONDS),
            "-i",
            str(audio),
            "-f",
            "f32le",
            "-ac",
            "1",
            "-ar",
            str(SAMPLE_RATE),
            "-",
        ]
        result = subprocess.run(command, capture_output=True, check=False)
        if result.returncode != 0 or not result.stdout:
            return np.zeros(0)
        return np.frombuffer(result.stdout, dtype=np.float32).astype(np.float64)
