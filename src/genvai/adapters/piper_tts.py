"""SpeechProvider backed by Piper.

CPU-only and fast, which matters: it leaves the whole GPU budget for the diffusion
model. Optional - requires `uv sync --extra tts`.
"""

from pathlib import Path

from genvai.config import TTSSettings


class PiperSpeechProvider:
    """Implements `SpeechProvider`."""

    name = "piper"

    def __init__(self, settings: TTSSettings, cache_dir: Path) -> None:
        self._settings = settings
        self._cache_dir = cache_dir

    def is_available(self) -> bool:
        """False when piper or the configured voice model is missing."""
        raise NotImplementedError

    def synthesise(self, text: str, *, voice: str | None = None, speed: float = 1.0) -> Path:
        raise NotImplementedError
