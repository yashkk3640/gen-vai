"""ImageProvider that composes abstract backdrops with Pillow.

The fallback that is always available: no GPU, no model download, no optional extra.

It deliberately does not attempt a photograph. A 4 GB card producing a mediocre imitation
of a photo looks worse than a deliberate graphic does, and the graphic never has six
fingers. What comes out is a gradient with a few soft shapes - a designed backdrop for
the text that sits on top, which in idea mode is where the meaning actually lives.

Output is deterministic: the same prompt and seed always give the same image, so a
timeline renders identically tomorrow.
"""

import hashlib
import math
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

Colour = tuple[int, int, int]

PALETTES: dict[str, tuple[Colour, Colour, Colour]] = {
    "warm": ((240, 148, 74), (196, 63, 78), (255, 214, 145)),
    "cool": ((38, 84, 140), (22, 138, 173), (168, 218, 220)),
    "night": ((17, 24, 43), (52, 43, 96), (120, 132, 199)),
    "forest": ((28, 74, 58), (82, 128, 74), (198, 214, 148)),
    "sunset": ((252, 116, 92), (128, 57, 105), (255, 191, 128)),
    "mono": ((28, 30, 34), (74, 78, 86), (206, 210, 216)),
    "sand": ((214, 178, 130), (150, 108, 76), (244, 231, 208)),
}

_KEYWORDS: dict[str, tuple[str, ...]] = {
    "warm": ("warm", "fire", "autumn", "coffee", "amber", "golden"),
    "cool": ("cool", "water", "ocean", "sea", "ice", "winter", "calm", "sky"),
    "night": ("night", "dark", "space", "star", "sleep", "moon", "midnight"),
    "forest": ("forest", "tree", "green", "nature", "leaf", "garden", "mountain"),
    "sunset": ("sunset", "sunrise", "dusk", "evening", "glow", "horizon"),
    "mono": ("mono", "grey", "gray", "minimal", "office", "business", "chart"),
    "sand": ("sand", "desert", "beach", "earth", "clay", "linen"),
}

GRAIN = 6.0
"""Noise amplitude. A flat gradient reads as a rendering error; a little grain reads as
a choice."""


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
        """Compose a backdrop for `prompt`, deterministically.

        The prompt steers the palette by keyword - "a calm ocean at dusk" comes out cool
        and dim - and seeds the composition, so two different scenes in one video do not
        come out identical.
        """
        rng = np.random.default_rng(_seed_of(prompt, seed))
        palette = PALETTES[_palette_for(prompt, rng)]

        canvas = _gradient(width, height, palette[0], palette[1], rng)
        _shapes(canvas, palette[2], rng)
        canvas = canvas.filter(ImageFilter.GaussianBlur(radius=max(2, min(width, height) // 120)))
        canvas = _grain(canvas, rng)

        # Dimensions belong in the name. Without them the same prompt rendered at two
        # sizes - a preview and a final, say - writes to one path, and the second silently
        # replaces the first.
        self._cache_dir.mkdir(parents=True, exist_ok=True)
        out = self._cache_dir / f"card-{_seed_of(prompt, seed):016x}-{width}x{height}.png"
        canvas.save(out)
        return out

    def unload(self) -> None:
        return None


def _palette_for(prompt: str, rng: np.random.Generator) -> str:
    """Pick a palette from the prompt's words, or at random when none of them fit."""
    words = set(prompt.lower().replace(",", " ").split())
    for name, triggers in _KEYWORDS.items():
        if words & set(triggers):
            return name
    return str(rng.choice(list(PALETTES)))


def _gradient(width: int, height: int, start: Colour, end: Colour, rng) -> Image.Image:
    """A linear gradient at an arbitrary angle.

    Built as an array rather than by drawing lines: one interpolation over a coordinate
    ramp is both faster and smoother than several thousand one-pixel rectangles.
    """
    angle = float(rng.uniform(0, math.pi))
    ys, xs = np.mgrid[0:height, 0:width]
    ramp = xs * math.cos(angle) + ys * math.sin(angle)
    ramp = (ramp - ramp.min()) / max(1e-9, ramp.max() - ramp.min())

    pixels = np.stack([ramp * (end[i] - start[i]) + start[i] for i in range(3)], axis=-1).astype(
        np.uint8
    )
    return Image.fromarray(pixels, mode="RGB")


def _shapes(canvas: Image.Image, accent: Colour, rng) -> None:
    """A few translucent discs, to give the frame something to hold the eye.

    Drawn on an overlay and composited, because Pillow's draw has no alpha of its own and
    opaque blobs would look pasted on.
    """
    width, height = canvas.size
    overlay = Image.new("RGBA", canvas.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)

    for _ in range(int(rng.integers(2, 5))):
        radius = int(rng.uniform(0.18, 0.45) * min(width, height))
        cx = int(rng.uniform(0, width))
        cy = int(rng.uniform(0, height))
        alpha = int(rng.uniform(26, 64))
        draw.ellipse([cx - radius, cy - radius, cx + radius, cy + radius], fill=(*accent, alpha))

    canvas.paste(Image.alpha_composite(canvas.convert("RGBA"), overlay).convert("RGB"), (0, 0))


def _grain(canvas: Image.Image, rng) -> Image.Image:
    pixels = np.asarray(canvas, dtype=np.float64)
    noisy = pixels + rng.normal(0, GRAIN, pixels.shape)
    return Image.fromarray(np.clip(noisy, 0, 255).astype(np.uint8), mode="RGB")


def _seed_of(prompt: str, seed: int) -> int:
    """A stable seed from the prompt and the timeline's own.

    Hashed rather than summed so two prompts differing by a word do not land on
    neighbouring seeds and produce near-identical images.
    """
    digest = hashlib.sha256(f"{seed}\x1f{prompt}".encode()).digest()
    return int.from_bytes(digest[:8], "big")
