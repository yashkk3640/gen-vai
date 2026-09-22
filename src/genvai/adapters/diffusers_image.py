"""ImageProvider backed by a local Stable Diffusion pipeline.

Optional: requires `uv sync --extra image` (~3 GB of torch). `is_available()` returns
False when the extra is not installed, and the caller falls back to procedural cards.

Defaults to SD-Turbo at 1-4 steps, which fits 4 GB with attention slicing and
sequential CPU offload. The LLM must be unloaded before this loads.
"""

from pathlib import Path

from genvai.config import ImageSettings


class DiffusersImageProvider:
    """Implements `ImageProvider`."""

    name = "diffusers"

    def __init__(self, settings: ImageSettings, cache_dir: Path) -> None:
        self._settings = settings
        self._cache_dir = cache_dir

    def is_available(self) -> bool:
        """False when torch/diffusers are missing or no usable device exists."""
        raise NotImplementedError

    def generate(
        self,
        prompt: str,
        *,
        width: int,
        height: int,
        seed: int,
        negative_prompt: str | None = None,
    ) -> Path:
        """Generate deterministically from `seed`.

        On CUDA OOM, retries at half resolution and upscales - a smaller image beats a
        failed render. See TODO.md M4.
        """
        raise NotImplementedError

    def unload(self) -> None:
        raise NotImplementedError
