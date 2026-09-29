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

from genvai.media import Face
from genvai.timeline import Rect

GRID = 7
"""How finely the picture is divided when looking. Windows overlap, so the effective
step is half a cell and a photograph need not line up with the grid."""

PAPER_LUMA = 218.0
PAPER_SATURATION = 0.16
"""Above one and below the other is background: the colour of an empty page."""

MIN_COLOUR = 14.0
"""Channel spread below which a region has nothing to look at, however dark it is."""

FACE_WEIGHT = 1.5
"""How much a whole face inside a window multiplies its score.

Picture-likeness cannot tell a face from a flower, and a backdrop with a person in it
holds a viewer where a flower does not - the client picked exactly those out of the
hand-built reel. Enough to beat a busier patch of ornament, not enough to lift a face
drawn in line art on bare paper over a real photograph.
"""

TEXT_WEIGHT = 5.0
"""How hard printed text inside a window counts against it, per unit of area.

OCR boxes are tight around the glyphs, so a banner line crossing a window covers only a
tenth of it - and still reads as a mistake behind a price. A tenth costs half the score;
a fifth rules the window out.
"""

ERASE_PAD = 0.3
"""Padding around a word box before it is painted out, as a fraction of its height.
OCR boxes hug the glyphs; descenders, accents and outlines sit just outside them."""

ERASE_MAX_HEIGHT = 0.045
"""Word boxes taller than this, as a fraction of the picture, are left alone.

Display script - the occasion's title, a slogan set across an illustration - comes back
as one huge box that sweeps up whatever it overlaps. Erasing it cut blocks out of the
dancer on both Navratri posters. Body type and prices are well under this."""

RING = 4
"""Pixels of surround sampled for the fill colour."""

FLAT_GROUND = 7.0
"""Median deviation of that surround above which it is not flat ground, and the word stays.

Measured on the client posters: type on paper, pills and banners 0.7-6; script over
ornament 8-14; script across the dancer 37."""

FACE_HEIGHT = 0.34
"""Where a face sits in a window built around it, from the top.

Above centre, because the centre of every promo beat is taken by the price.
"""


def photo_regions(
    image: NDArray[np.uint8],
    *,
    count: int = 4,
    aspect: float | None = None,
    faces: tuple[Face, ...] = (),
    text: tuple[Rect, ...] = (),
) -> tuple[Rect, ...]:
    """The most picture-like rectangles in a layout, best first.

    Returns rects in fractions of the frame, non-overlapping, so each gives a distinct
    backdrop rather than four views of the same photograph.

    `aspect` widens or narrows the window to the shape it will eventually be cropped to,
    which matters: a region that scores well as a square may be half type once it has
    been squeezed into 9:16.

    `faces` adds a window placed around each one to the candidates, and weights any
    window holding a whole face above one that does not. `text` - where OCR found
    words - counts against any window it falls in, so a backdrop does not carry half a
    heading behind the reel's own type.
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

    origins = [
        (left, top)
        for top in range(0, max(1, height - int(window_h) + 1), step_y)
        for left in range(0, max(1, width - int(window_w) + 1), step_x)
    ]
    origins += [_around(face, width, height, window_w, window_h) for face in faces]

    scored: list[tuple[float, Rect]] = []
    for left, top in origins:
        patch = image[top : top + int(window_h), left : left + int(window_w), :3]
        value = score(patch)
        if value <= 0:
            continue
        rect = (left / width, top / height, window_w / width, window_h / height)
        value *= 1.0 + FACE_WEIGHT * _faces_held(rect, faces)
        value *= max(0.0, 1.0 - TEXT_WEIGHT * _text_cover(rect, text))
        if value > 0:
            scored.append((value, rect))

    scored.sort(key=lambda pair: -pair[0])
    return tuple(_spread_out(scored, count))


def erase_text(
    image: NDArray[np.uint8], text: tuple[Rect, ...]
) -> tuple[NDArray[np.uint8], tuple[Rect, ...]]:
    """The picture with its printed words painted out, for cutting backdrops from.

    Each word box, padded, is filled with the median colour of a thin ring just outside
    it. Print sets type on flat ground - paper, a pill, a banner - so the ring is that
    ground and the fill is invisible. Where the ring is not flat - script lettering over
    ornament, a caption across a photograph - the word is left alone: a flat block cut
    into a picture is worse than the word.

    Returns the picture and the body-sized words that could not be erased, which are the
    ones still worth keeping out of a crop.

    Deliberately not an inpainting model: this is a few lines of numpy and deterministic.
    """
    if image.ndim != 3 or not text:
        return image, ()
    height, width = image.shape[:2]
    out = image.copy()
    stubborn: list[Rect] = []
    for rect in text:
        x, y, w, h = rect
        if h > ERASE_MAX_HEIGHT:
            continue
        pad = max(3, round(h * height * ERASE_PAD))
        left, top = max(0, round(x * width) - pad), max(0, round(y * height) - pad)
        right = min(width, round((x + w) * width) + pad)
        bottom = min(height, round((y + h) * height) + pad)
        if right <= left or bottom <= top:
            continue
        ring = _ring(image, left, top, right, bottom, RING)
        if not ring.size:
            continue
        ground = np.median(ring, axis=0)
        # Median deviation, not spread: the ring catches strokes of the next word over,
        # which inflate a standard deviation on perfectly flat paper.
        if float(np.median(np.abs(ring - ground).mean(axis=1))) <= FLAT_GROUND:
            out[top:bottom, left:right] = ground.astype(np.uint8)
        else:
            stubborn.append(rect)
    return out, tuple(stubborn)


def _ring(
    image: NDArray[np.uint8], left: int, top: int, right: int, bottom: int, thickness: int
) -> NDArray[np.uint8]:
    """The pixels in a band just outside a rectangle, as rows of RGB."""
    height, width = image.shape[:2]
    outer = image[
        max(0, top - thickness) : min(height, bottom + thickness),
        max(0, left - thickness) : min(width, right + thickness),
        :3,
    ]
    mask = np.ones(outer.shape[:2], dtype=bool)
    mask[
        top - max(0, top - thickness) : bottom - max(0, top - thickness),
        left - max(0, left - thickness) : right - max(0, left - thickness),
    ] = False
    return outer[mask]


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


def _around(
    face: Face, width: int, height: int, window_w: float, window_h: float
) -> tuple[int, int]:
    """The top-left of a window centred across a face and holding it above centre."""
    cx, cy = face.centre
    left = cx * width - window_w / 2
    top = cy * height - window_h * FACE_HEIGHT
    return (
        int(min(max(0.0, left), width - window_w)),
        int(min(max(0.0, top), height - window_h)),
    )


def _faces_held(rect: Rect, faces: tuple[Face, ...]) -> float:
    """How much face a window holds, 0-1: each face counted by how much of it is inside.

    A face cut in half by the window edge counts for half, which is generous - but a
    crop through someone's face is caught later by the inset anyway.
    """
    held = 0.0
    for face in faces:
        if face.area <= 0:
            continue
        held += face.confidence * _shared(rect, face.rect) / face.area
    return min(1.0, held)


def _text_cover(rect: Rect, text: tuple[Rect, ...]) -> float:
    """What fraction of a window printed text covers."""
    area = rect[2] * rect[3]
    if area <= 0 or not text:
        return 0.0
    return min(1.0, sum(_shared(rect, box) for box in text) / area)


def _shared(a: Rect, b: Rect) -> float:
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    wide = max(0.0, min(ax + aw, bx + bw) - max(ax, bx))
    tall = max(0.0, min(ay + ah, by + bh) - max(ay, by))
    return wide * tall


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
    shared = _shared(a, b)
    smaller = min(a[2] * a[3], b[2] * b[3])
    return smaller > 0 and shared / smaller > allowed
