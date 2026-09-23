"""Deciding where to crop when a shot does not fit the canvas.

Phone clips are landscape, reels are vertical, and a centre crop cuts the subject in
half about as often as it works. This finds where the interesting part of the frame is
and crops around that instead.

Saliency rather than face detection, and the choice is not only about avoiding a
dependency. A face detector answers one question well and everything else not at all -
it has nothing to say about a plate of food, a dog, a building or a sunset, which is
most of what a camera roll holds. Faces would be a genuine improvement *on top* of
this, and are noted in the backlog.

Saliency here means local detail density. Spectral residual, the textbook choice, was
tried first and measured: it located a centred subject well and then failed badly near
the frame edges, because the FFT wraps and a subject against the border aliases to the
opposite side. A subject at 15% across the frame was reported at 60%. Detail energy has
no wrap, no spectrum, and placed the same subject within 0.2% everywhere across the
frame - so that is what is here.

All of it is numpy over greyscale frames - no model, no GPU.
"""

import numpy as np
from numpy.typing import NDArray

from genvai.timeline import Rect

Frame = NDArray[np.float64]

MAP_SIZE = 64
"""Saliency is computed at this resolution.

Deliberately coarse. The output is one crop rectangle, so resolution beyond "which
part of the frame" is wasted.
"""

POOL_WIDTH = 5
"""Neighbourhood, in map cells, that detail is pooled over.

A lone hard edge is not a subject; a region dense with edges is.
"""

BACKGROUND_PERCENTILE = 60.0
"""What counts as background and is subtracted away entirely.

Load-bearing: without it the flat parts of a frame contribute an even pedestal, and a
centre of mass over a pedestal sits near the middle wherever the subject actually is.
"""

DRIFT_THRESHOLD = 0.06
"""How far the subject must travel, as a fraction of frame width, to earn a moving crop.

Below this the crop is held still. A frame that creeps by two percent reads as a fault
in the render rather than as camera work.
"""

SMOOTHING = 5
"""Frames of moving average over the subject track.

Saliency jitters shot to shot; without smoothing the crop would chase it and the result
would be less watchable than a centre crop.
"""


def saliency(frame: Frame) -> Frame:
    """Where the detail is - which, for framing, is where the subject is.

    Gradient magnitude pooled over small neighbourhoods. A subject carries texture and
    edges; sky, walls, tables and bokeh do not, so detail density locates the thing
    worth keeping in shot without knowing what it is.

    The percentile floor is the part that matters. Without it a flat background still
    contributes a small, even pedestal across the whole frame, and a centre of mass over
    that pedestal sits near the middle no matter where the subject is. Subtracting it
    makes the background contribute nothing at all.
    """
    small = _resize(frame, MAP_SIZE)
    if small.size == 0:
        return small

    horizontal = np.zeros_like(small)
    vertical = np.zeros_like(small)
    horizontal[:, 1:-1] = small[:, 2:] - small[:, :-2]
    vertical[1:-1, :] = small[2:, :] - small[:-2, :]

    pooled = _box_blur(np.hypot(horizontal, vertical), POOL_WIDTH, pad="edge")
    lit = np.maximum(pooled - np.percentile(pooled, BACKGROUND_PERCENTILE), 0.0)
    return _normalise(lit)


def focus(frame: Frame) -> tuple[float, float]:
    """The subject's centre, as fractions of width and height.

    A saliency-weighted centre of mass rather than the single brightest point: one
    specular highlight should not drag the frame off the subject.
    """
    heat = saliency(frame)
    total = heat.sum()
    if total <= 0:
        return 0.5, 0.5

    rows, columns = heat.shape
    x = float((heat.sum(axis=0) @ (np.arange(columns) + 0.5)) / total / columns)
    y = float((heat.sum(axis=1) @ (np.arange(rows) + 0.5)) / total / rows)
    return _clamp(x), _clamp(y)


def crop_for(centre: tuple[float, float], source_aspect: float, target_aspect: float) -> Rect:
    """The largest rectangle of the target shape that fits, centred on the subject.

    Largest, so nothing is thrown away that did not have to be, and clamped to the frame
    so a subject near the edge yields an edge crop rather than one hanging off it.
    """
    if source_aspect <= 0 or target_aspect <= 0:
        return (0.0, 0.0, 1.0, 1.0)

    x, y = centre
    if source_aspect > target_aspect:
        width = min(1.0, target_aspect / source_aspect)
        return (_clamp(x - width / 2, 1.0 - width), 0.0, width, 1.0)

    height = min(1.0, source_aspect / target_aspect)
    return (0.0, _clamp(y - height / 2, 1.0 - height), 1.0, height)


def solve_path(
    frames: list[Frame], source_aspect: float, target_aspect: float
) -> tuple[Rect, Rect | None]:
    """A crop for the shot: one rectangle, or two when the subject actually moves.

    Returns `(crop, None)` for a still frame and `(start, end)` for a pan. The second is
    None far more often than not, which is the intended outcome - a moving crop is only
    worth its cost when the subject genuinely travels.
    """
    if not frames:
        return (0.0, 0.0, 1.0, 1.0), None

    track = [focus(frame) for frame in frames]
    xs = _smooth([p[0] for p in track])
    ys = _smooth([p[1] for p in track])

    drift = max(max(xs) - min(xs), max(ys) - min(ys))
    if drift < DRIFT_THRESHOLD or len(frames) < 2:
        middle = (float(np.median(xs)), float(np.median(ys)))
        return crop_for(middle, source_aspect, target_aspect), None

    start = crop_for((xs[0], ys[0]), source_aspect, target_aspect)
    end = crop_for((xs[-1], ys[-1]), source_aspect, target_aspect)
    return start, (None if start == end else end)


def needs_reframing(source_aspect: float, target_aspect: float, *, tolerance: float = 0.02) -> bool:
    """Whether the shapes differ enough that a crop changes anything."""
    if source_aspect <= 0 or target_aspect <= 0:
        return False
    return abs(source_aspect - target_aspect) / max(source_aspect, target_aspect) > tolerance


# ---------------------------------------------------------------------------- helpers


def _smooth(values: list[float]) -> list[float]:
    """Moving average, with the ends held rather than tapered to zero."""
    if len(values) < 3:
        return values
    width = min(SMOOTHING, len(values))
    padded = np.pad(np.asarray(values, dtype=np.float64), width // 2, mode="edge")
    kernel = np.ones(width) / width
    return list(np.convolve(padded, kernel, mode="valid")[: len(values)])


def _resize(frame: Frame, size: int) -> Frame:
    """Area-average down to a square. Averaging, so noise does not survive the shrink."""
    if frame.ndim != 2 or frame.size == 0:
        return np.zeros((0, 0))
    rows = np.linspace(0, frame.shape[0], size + 1).astype(int)
    columns = np.linspace(0, frame.shape[1], size + 1).astype(int)
    out = np.zeros((size, size))
    for i in range(size):
        for j in range(size):
            block = frame[
                rows[i] : max(rows[i] + 1, rows[i + 1]),
                columns[j] : max(columns[j] + 1, columns[j + 1]),
            ]
            out[i, j] = block.mean() if block.size else 0.0
    return out


def _box_blur(values: Frame, width: int, *, pad: str = "edge") -> Frame:
    """Separable box blur - two 1-D passes, the same result far more cheaply.

    `pad` matters more than it looks. A spectrum wraps, so it averages with `wrap`; an
    image does not, so it averages with `edge`. Zero padding would be wrong for both,
    and silently so.
    """
    if width < 2 or values.size == 0:
        return values
    kernel = np.ones(width) / width
    half = width // 2
    out = values
    for axis in (1, 0):
        padded = np.pad(out, [(half, half) if a == axis else (0, 0) for a in (0, 1)], mode=pad)
        out = np.apply_along_axis(
            lambda line: np.convolve(line, kernel, mode="valid"), axis, padded
        )[: values.shape[0], : values.shape[1]]
    return out


def _normalise(values: Frame) -> Frame:
    peak = values.max() if values.size else 0.0
    return values / peak if peak > 0 else values


def _clamp(value: float, high: float = 1.0) -> float:
    return float(min(max(value, 0.0), max(high, 0.0)))
