"""Finding the parts of a designed layout that are pictures rather than words.

A poster is mostly type. Type makes a terrible backdrop - it is unreadable behind the
reel's own text and it is not what anyone wants to look at - so the photographs and
illustrations have to be found and the price list left alone.

The discriminator is simple and holds well on print layouts: a text block is mostly
paper, so most of its pixels are near-white and nearly colourless, while a photograph
fills its rectangle with varied colour. Scoring on that separates the two without
anything resembling layout analysis.

Pure numpy over an image array.
"""

import numpy as np
from numpy.typing import NDArray

from genvai.timeline import Rect

GRID = 7
"""How finely the picture is divided when looking. Windows overlap, so the effective
step is half a cell and a photograph need not line up with the grid."""

PAPER_LUMA = 218.0
PAPER_SATURATION = 0.16
"""Above one and below the other is background: the colour of an empty page."""

MIN_COLOUR = 14.0
"""Channel spread below which a region has nothing to look at, however dark it is."""


def photo_regions(
    image: NDArray[np.uint8], *, count: int = 4, aspect: float | None = None
) -> tuple[Rect, ...]:
    """The most picture-like rectangles in a layout, best first.

    Returns rects in fractions of the frame, non-overlapping, so each gives a distinct
    backdrop rather than four views of the same photograph.

    `aspect` widens or narrows the window to the shape it will eventually be cropped to,
    which matters: a region that scores well as a square may be half type once it has
    been squeezed into 9:16.
    """
    if image.ndim != 3 or image.size == 0:
        return ()

    height, width = image.shape[:2]
    window_h = height / GRID * 1.6
    window_w = window_h * aspect if aspect else width / GRID * 1.6
    window_w = min(window_w, width)
    window_h = min(window_h, height)

    step_y = max(1, int(window_h / 2))
    step_x = max(1, int(window_w / 2))

    scored: list[tuple[float, Rect]] = []
    for top in range(0, max(1, height - int(window_h) + 1), step_y):
        for left in range(0, max(1, width - int(window_w) + 1), step_x):
            patch = image[top : top + int(window_h), left : left + int(window_w), :3]
            value = score(patch)
            if value <= 0:
                continue
            scored.append(
                (
                    value,
                    (left / width, top / height, window_w / width, window_h / height),
                )
            )

    scored.sort(key=lambda pair: -pair[0])
    return tuple(_spread_out(scored, count))


def score(patch: NDArray[np.uint8]) -> float:
    """How much of a picture, rather than a page, this rectangle is.

    Two things have to be true at once. It must have colour in it - a region whose
    channels barely differ is a grey box or a rule. And it must not be mostly paper - a
    price list is ninety percent empty background with ink on top, and averaging over
    that gives a light, flat, useless backdrop.
    """
    if patch.size == 0:
        return 0.0

    pixels = patch.reshape(-1, 3).astype(np.float64)
    luma = pixels @ np.array([0.299, 0.587, 0.114])
    high = pixels.max(axis=1)
    low = pixels.min(axis=1)
    saturation = np.where(high > 0, (high - low) / np.maximum(high, 1e-6), 0.0)

    paper = float(np.mean((luma > PAPER_LUMA) & (saturation < PAPER_SATURATION)))
    colour = float(pixels.std(axis=0).mean())
    if colour < MIN_COLOUR:
        return 0.0

    # Squared, because a region that is two thirds paper is far worse than one that is
    # a third paper - it is a caption, not a picture.
    return colour * (1.0 - paper) ** 2


def _spread_out(scored: list[tuple[float, Rect]], count: int) -> list[Rect]:
    """Keep the best, then only rects that do not overlap one already kept."""
    kept: list[Rect] = []
    for _, rect in scored:
        if any(_overlaps(rect, taken) for taken in kept):
            continue
        kept.append(rect)
        if len(kept) == count:
            break
    return kept


def _overlaps(a: Rect, b: Rect, *, allowed: float = 0.25) -> bool:
    """Whether two rects share more than a little of their area."""
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    wide = max(0.0, min(ax + aw, bx + bw) - max(ax, bx))
    tall = max(0.0, min(ay + ah, by + bh) - max(ay, by))
    shared = wide * tall
    smaller = min(aw * ah, bw * bh)
    return smaller > 0 and shared / smaller > allowed
