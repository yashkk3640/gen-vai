"""The pictures on a poster, found as objects - the cast a storyboard can call on.

`regions.py` finds 9:16 windows worth cropping. A storyboard needs something else: each
picture on its own - this photo tile, that dancer, the logo portrait - with its bounds,
whether a face is in it, and what it shows. "Makeup model, push in on the face" needs
the makeup photo as a thing, not a window that happens to contain it.

The method is the same observation `regions.py` rests on: a picture is not paper. Cells
of a coarse grid are marked picture where they are mostly not paper and have colour in
them; touching cells are grouped; overlapping groups are merged. What each shows is read
off the poster itself - the word printed beside a tile is its name.

Pure numpy over arrays that callers have already made: the poster with its words
erased, the OCR boxes, the faces.
"""

from typing import Literal

import numpy as np
from numpy.typing import NDArray
from pydantic import Field

from genvai.media import Face
from genvai.ocr import Box
from genvai.regions import PAPER_LUMA, PAPER_SATURATION, score
from genvai.timeline import Frozen, Rect

CELL = 8
"""Grid cell in pixels. Fine enough to separate tiles set 16 px apart."""

MIN_CELLS = 40
"""Smallest group kept, in cells - about a 50 px square on a 1024 px poster."""

MIN_SPREAD = 12.0
"""Channel spread under which a cell is flat colour, not picture."""

PORTRAIT = 3.2
"""How much bigger than a face its portrait is taken to be, around it."""

NAME_REACH = 0.25
"""How far left of a picture, as a fraction of the poster's width, its name may sit."""

ORNAMENT = 16.0
"""Picture strength under which it is decoration - mandala, a faint pattern. The dancer
on the Navratri poster scores 23 for all the pale sky around her; mandalas 5-14."""

Kind = Literal["photo", "portrait", "illustration", "ornament"]


class Picture(Frozen):
    """One picture on a poster."""

    id: str
    poster: int = Field(default=0, description="Which poster it is on, by position.")
    rect: Rect = Field(description="Fractions of the poster.")
    kind: Kind
    name: str = Field(default="", description="The word printed beside it, lowercased.")
    face: Rect | None = Field(default=None, description="Largest face, in poster fractions.")
    strength: float = Field(default=0.0, description="How much of a picture it is.")
    pixels: tuple[int, int] = Field(default=(0, 0), description="Width, height in pixels.")

    @property
    def aspect(self) -> float:
        return self.pixels[0] / self.pixels[1] if self.pixels[1] else 1.0


def find_cast(
    image: NDArray[np.uint8],
    *,
    boxes: tuple[Box, ...] = (),
    faces: tuple[Face, ...] = (),
    stubborn: tuple[Rect, ...] = (),
) -> tuple[Picture, ...]:
    """Every picture on the poster, strongest first.

    `image` should already have its printed words erased, or a price list reads as one
    long picture. `boxes` are the OCR boxes in pixels, used for names; `faces` come from
    a detector and add portraits the grid cannot see - line art on paper has no colour.
    `stubborn` are words that could not be erased; one running along a picture's top or
    bottom edge - a banner under an illustration - is trimmed off it.
    """
    if image.ndim != 3 or image.size == 0:
        return ()
    height, width = image.shape[:2]

    groups = _merge([_to_pixels(g) for g in _groups(_picture_cells(image))])
    # A face inside a found picture belongs to it. Only a face on bare paper - line art,
    # which the grid cannot see - gets a portrait of its own, and that portrait is not
    # merged: a box three faces wide swallowed the neighbouring tiles when it was.
    groups += [
        _portrait(face, width, height)
        for face in faces
        if not any(_holds(group, face, width, height) for group in groups)
    ]

    groups = [_trimmed(g, stubborn, width, height) for g in groups]

    pictures: list[Picture] = []
    for left, top, right, bottom in groups:
        patch = image[top:bottom, left:right, :3]
        strength = score(patch)
        rect = (left / width, top / height, (right - left) / width, (bottom - top) / height)
        face = _largest_face_in(rect, faces)
        pictures.append(
            Picture(
                id="",
                rect=rect,
                kind=_kind(patch, strength, face),
                name=_name(rect, boxes, width, height),
                face=face,
                strength=strength,
                pixels=(right - left, bottom - top),
            )
        )

    ranked = sorted(pictures, key=lambda p: -_rank(p))
    return tuple(p.model_copy(update={"id": f"p{i + 1}"}) for i, p in enumerate(ranked))


# ------------------------------------------------------------------------ finding


def _picture_cells(image: NDArray[np.uint8]) -> NDArray[np.bool_]:
    height, width = image.shape[:2]
    rows, columns = height // CELL, width // CELL
    cells = image[: rows * CELL, : columns * CELL, :3].astype(np.float64)
    cells = cells.reshape(rows, CELL, columns, CELL, 3)
    luma = cells @ np.array([0.299, 0.587, 0.114])
    high, low = cells.max(axis=-1), cells.min(axis=-1)
    saturation = (high - low) / np.maximum(high, 1.0)
    paper = ((luma > PAPER_LUMA) & (saturation < PAPER_SATURATION)).mean(axis=(1, 3))
    spread = cells.std(axis=(1, 3)).mean(axis=-1)
    return (paper < 0.5) & (spread > MIN_SPREAD)


def _groups(mask: NDArray[np.bool_]) -> list[tuple[int, int, int, int]]:
    """Bounding boxes, in cells, of 4-connected groups of picture cells."""
    seen = np.zeros_like(mask)
    found: list[tuple[int, int, int, int]] = []
    rows, columns = mask.shape
    for y, x in zip(*np.nonzero(mask), strict=True):
        if seen[y, x]:
            continue
        seen[y, x] = True
        stack, ys, xs = [(y, x)], [], []
        while stack:
            cy, cx = stack.pop()
            ys.append(cy)
            xs.append(cx)
            for ny, nx in ((cy + 1, cx), (cy - 1, cx), (cy, cx + 1), (cy, cx - 1)):
                if 0 <= ny < rows and 0 <= nx < columns and mask[ny, nx] and not seen[ny, nx]:
                    seen[ny, nx] = True
                    stack.append((ny, nx))
        if len(ys) >= MIN_CELLS:
            found.append((min(xs), min(ys), max(xs) + 1, max(ys) + 1))
    return found


def _to_pixels(group: tuple[int, int, int, int]) -> tuple[int, int, int, int]:
    return tuple(v * CELL for v in group)  # type: ignore[return-value]


def _portrait(face: Face, width: int, height: int) -> tuple[int, int, int, int]:
    """A box around a face big enough to hold the head and shoulders."""
    cx, cy = face.centre
    half_w = face.rect[2] * width * PORTRAIT / 2
    half_h = face.rect[3] * height * PORTRAIT / 2
    return (
        max(0, int(cx * width - half_w)),
        max(0, int(cy * height - half_h)),
        min(width, int(cx * width + half_w)),
        min(height, int(cy * height + half_h * 1.2)),
    )


def _trimmed(
    group: tuple[int, int, int, int], stubborn: tuple[Rect, ...], width: int, height: int
) -> tuple[int, int, int, int]:
    """Cut a banner of type off the top or bottom edge of a picture."""
    left, top, right, bottom = group
    tall = bottom - top
    for x, y, w, h in stubborn:
        bx0, by0, bx1, by1 = x * width, y * height, (x + w) * width, (y + h) * height
        across = max(0.0, min(right, bx1) - max(left, bx0))
        if across < (right - left) * 0.4:
            continue
        if by0 > top + tall * 0.7 and by0 < bottom:
            bottom = min(bottom, int(by0 - h * height * 0.6))
        elif by1 < top + tall * 0.3 and by1 > top:
            top = max(top, int(by1 + h * height * 0.6))
    return (left, top, right, max(top + CELL, bottom))


def _holds(group: tuple[int, int, int, int], face: Face, width: int, height: int) -> bool:
    cx, cy = face.centre
    return group[0] <= cx * width <= group[2] and group[1] <= cy * height <= group[3]


def _merge(boxes: list[tuple[int, int, int, int]]) -> list[tuple[int, int, int, int]]:
    """Join boxes that overlap until none do. A photo split by a pale stripe is one."""
    merged = list(boxes)
    changed = True
    while changed:
        changed = False
        for i in range(len(merged)):
            for j in range(i + 1, len(merged)):
                a, b = merged[i], merged[j]
                if a[0] < b[2] and b[0] < a[2] and a[1] < b[3] and b[1] < a[3]:
                    merged[i] = (min(a[0], b[0]), min(a[1], b[1]), max(a[2], b[2]), max(a[3], b[3]))
                    del merged[j]
                    changed = True
                    break
            if changed:
                break
    return merged


# ------------------------------------------------------------------- describing


def _kind(patch: NDArray[np.uint8], strength: float, face: Rect | None) -> Kind:
    if face is not None:
        return "portrait"
    if strength < ORNAMENT:
        return "ornament"
    height, width = patch.shape[:2]
    # A photograph fills its rectangle; an illustration on a poster floats on paper.
    edge = np.concatenate([patch[0], patch[-1], patch[:, 0], patch[:, -1]]).astype(np.float64)
    luma = edge @ np.array([0.299, 0.587, 0.114])
    return "illustration" if float(np.mean(luma > PAPER_LUMA)) > 0.5 else "photo"


def _largest_face_in(rect: Rect, faces: tuple[Face, ...]) -> Rect | None:
    x, y, w, h = rect
    inside = [f for f in faces if x <= f.centre[0] <= x + w and y <= f.centre[1] <= y + h]
    return max(inside, key=lambda f: f.area).rect if inside else None


def _name(rect: Rect, boxes: tuple[Box, ...], width: int, height: int) -> str:
    """The word printed beside a picture - level with it on the left, or else above it."""
    x, y, w, h = rect
    beside = [
        b
        for b in boxes
        if y * height <= b.middle <= (y + h) * height
        and b.right <= x * width + 4
        and x * width - b.right <= NAME_REACH * width
        # A tile's label is set in capitals; a sentence beside it is copy, not a name.
        and _capitals(b.cleaned)
    ]
    if beside:
        nearest = min(beside, key=lambda b: x * width - b.right)
        return " ".join(nearest.cleaned.split()).lower()

    # A grid sets its labels above each picture, often on two lines: "1 FINGER" / "ART".
    # Only for grid-sized pictures: above a big illustration is whatever copy was there.
    if w > 0.2:
        return ""
    above = [
        b
        for b in boxes
        if b.y + b.height <= y * height + 4
        and y * height - (b.y + b.height) <= h * height * 0.6
        and abs((b.x + b.width / 2) - (x + w / 2) * width) <= w * width * 0.6
        and _capitals(b.cleaned)
    ]
    return " ".join(b.cleaned for b in sorted(above, key=lambda b: b.y)).lower()


def _capitals(text: str) -> bool:
    letters = [c for c in text if c.isalpha()]
    return len(letters) >= 3 and all(c.isupper() for c in letters)


def _rank(picture: Picture) -> float:
    """People first, then photographs, then the rest; big and colourful within each."""
    weight = {"portrait": 3.0, "photo": 2.0, "illustration": 1.6, "ornament": 0.4}[picture.kind]
    area = picture.rect[2] * picture.rect[3]
    return weight * picture.strength * (0.3 + area) ** 0.5
