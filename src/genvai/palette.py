"""Reading a brand's colours off its own artwork.

A promo reel has to look like it belongs to the poster it came from. Asking the user for
hex codes is asking them to do work the picture already answers, so the colours are
taken from the artwork: the deep one to darken behind text, the accent for rules and
highlights, the light one for backgrounds.

Pure numpy over an image array - k-means with a fixed seed, so the same poster always
yields the same palette and a reel re-renders identically tomorrow.
"""

import numpy as np
from numpy.typing import NDArray

from genvai.timeline import Frozen

Colour = tuple[int, int, int]

SAMPLE = 120
"""Longest edge the image is reduced to before clustering. Colour has no fine detail."""

CLUSTERS = 6
ITERATIONS = 12

MIN_SATURATION = 0.18
"""Below this a colour is effectively grey, and a grey accent is not an accent."""

DEEP_LUMA = 52.0
"""Brightness the scrim colour is scaled to.

Dark enough that white text sits on it comfortably, light enough that the hue still
reads - a near-black scrim belongs to no brand in particular.
"""


class Brand(Frozen):
    """Three colours that describe a piece of artwork well enough to build around."""

    deep: Colour
    accent: Colour
    light: Colour

    def hex(self, which: str = "accent") -> str:
        value: Colour = getattr(self, which)
        return "#{:02X}{:02X}{:02X}".format(*value)


def brand(image: NDArray[np.uint8]) -> Brand:
    """The three working colours of a picture.

    `deep` is the darkest colour with real hue in it, which is what a scrim and a text
    outline want - a neutral black would look borrowed from somewhere else. `accent` is
    the most saturated, which on a poster is almost always the colour it was designed
    around. `light` is the palest, for the background of a card.

    Falls back to sensible neutrals rather than failing on a picture with no colour in
    it at all.
    """
    colours = dominant(image)
    if not colours:
        return Brand(deep=(32, 28, 34), accent=(190, 60, 90), light=(246, 240, 240))

    scored = [(c, _luma(c), _saturation(c)) for c in colours]
    coloured = [s for s in scored if s[2] >= MIN_SATURATION] or scored

    deep = min(coloured, key=lambda s: s[1])[0]
    accent = max(coloured, key=lambda s: s[2] * (0.4 + 0.6 * min(1.0, s[1] / 140)))[0]
    light = max(scored, key=lambda s: s[1])[0]

    # A scrim only works if it is actually dark, and a card background only if it is
    # actually pale, so both are pushed into a usable band rather than trusted as found.
    # `deep` is scaled toward a target brightness instead of darkened by a fixed amount:
    # darkening a colour that was already dark gives near-black, which is a scrim that
    # could have come from any brand at all.
    return Brand(deep=_at_luma(deep, DEEP_LUMA), accent=accent, light=_lighten(light, 0.55))


def dominant(image: NDArray[np.uint8], count: int = CLUSTERS) -> list[Colour]:
    """The main colours in a picture, most common first.

    k-means rather than a histogram: a histogram of a photograph returns a hundred near
    identical beiges, while clustering returns the handful a person would actually name.
    """
    pixels = _sample(image)
    if pixels.size == 0:
        return []

    centres = _seed(pixels, count)
    for _ in range(ITERATIONS):
        distances = ((pixels[:, None, :] - centres[None, :, :]) ** 2).sum(axis=2)
        labels = distances.argmin(axis=1)
        moved = np.array(
            [
                pixels[labels == k].mean(axis=0) if np.any(labels == k) else centres[k]
                for k in range(len(centres))
            ]
        )
        if np.allclose(moved, centres, atol=0.5):
            centres = moved
            break
        centres = moved

    sizes = [int((labels == k).sum()) for k in range(len(centres))]
    order = np.argsort(sizes)[::-1]
    return [tuple(int(v) for v in centres[k]) for k in order if sizes[k] > 0]  # type: ignore[misc]


def _sample(image: NDArray[np.uint8]) -> NDArray[np.float64]:
    """Shrink to a manageable number of pixels by plain striding.

    Averaging would invent colours that are not in the picture - the mean of red and
    white is pink, which may appear nowhere on the poster.
    """
    if image.ndim != 3 or image.shape[2] < 3 or image.size == 0:
        return np.zeros((0, 3))
    step = max(1, max(image.shape[0], image.shape[1]) // SAMPLE)
    return image[::step, ::step, :3].reshape(-1, 3).astype(np.float64)


def _seed(pixels: NDArray[np.float64], count: int) -> NDArray[np.float64]:
    """Spread the starting centres out, deterministically.

    Picking at random would make the palette differ between runs, and a reel that
    re-renders in different colours is not reproducible.
    """
    chosen = [pixels[0]]
    for _ in range(1, min(count, len(pixels))):
        far = ((pixels[:, None, :] - np.array(chosen)[None, :, :]) ** 2).sum(axis=2).min(axis=1)
        chosen.append(pixels[int(far.argmax())])
    return np.array(chosen, dtype=np.float64)


def _luma(colour: Colour) -> float:
    r, g, b = colour
    return 0.299 * r + 0.587 * g + 0.114 * b


def _saturation(colour: Colour) -> float:
    high, low = max(colour), min(colour)
    return (high - low) / high if high else 0.0


def _at_luma(colour: Colour, target: float) -> Colour:
    """Scale a colour to a given brightness, keeping its hue.

    Multiplying every channel by one factor preserves their ratios, so a maroon stays a
    maroon; adding or subtracting equally would wash it toward grey.
    """
    current = _luma(colour)
    if current <= 1.0:
        return (int(target), int(target * 0.35), int(target * 0.55))
    factor = target / current
    return tuple(min(255, int(c * factor)) for c in colour)  # type: ignore[return-value]


def _lighten(colour: Colour, amount: float) -> Colour:
    return tuple(int(c + (255 - c) * amount) for c in colour)  # type: ignore[return-value]
