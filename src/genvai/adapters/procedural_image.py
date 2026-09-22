"""ImageProvider that draws typographic cards with Pillow.

The fallback that is always available: no GPU, no model download, no extra install.
Renders a gradient background and lays out the scene's text. Plain, but it means the
pipeline runs end-to-end on any machine.
"""

from pathlib import Path


class ProceduralImageProvider:
    """Implements `ImageProvider`."""

    name = "procedural"

    def __init__(self, cache_dir: Path) -> None:
        self._cache_dir = cache_dir

    def is_available(self) -> bool:
        return True

    def generate(
        self,
        prompt: str,
        *,
        width: int,
        height: int,
        seed: int,
        negative_prompt: str | None = None,
    ) -> Path:
        """Draw a card. `seed` picks the palette and gradient angle, so it is reproducible.

        `prompt` is used as the card's text rather than as a generation prompt.
        """
        raise NotImplementedError

    def unload(self) -> None:
        return None
