"""Choosing an image provider for the machine it is running on.

One place that knows the fallback order, so no caller has to. The rule is simple: honour
the setting when the machine can, and degrade to procedural backdrops rather than fail.
"""

from pathlib import Path

from genvai.adapters.procedural_image import ProceduralImageProvider
from genvai.config import ImageSettings
from genvai.ports import ImageProvider


def image_provider(
    settings: ImageSettings, cache_dir: Path, *, on_note: object = None
) -> ImageProvider:
    """The best available provider, given what is installed.

    `diffusers` falls back to procedural when the extra is missing or there is no usable
    device - which is the common case, and not an error. Procedural is always available,
    so this function never fails.
    """
    if settings.provider == "diffusers":
        from genvai.adapters.diffusers_image import DiffusersImageProvider

        diffusion = DiffusersImageProvider(settings, cache_dir)
        if diffusion.is_available():
            return diffusion
        if callable(on_note):
            on_note(
                "Stable Diffusion is not installed or has no usable GPU - using "
                "designed backdrops instead. Install it with: uv sync --extra image"
            )
    return ProceduralImageProvider(cache_dir)
