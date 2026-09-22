"""Measuring picture quality. Pure arithmetic over greyscale frames.

Every function here takes numbers and returns numbers - no files, no ffmpeg, no model.
That keeps the judgements testable against synthetic images whose answer is known, which
matters because these scores decide what ends up in the video.

All scores are normalised to 0-1 so they can be weighted and summed. The raw quantities
are unbounded and have no natural ceiling, so each is mapped through `_saturate`, whose
midpoint constant is the only real tuning knob.
"""

from collections.abc import Sequence

import numpy as np
from numpy.typing import NDArray

Frame = NDArray[np.float64]
"""A greyscale frame, 0-255."""

SHARPNESS_MIDPOINT = 300.0
"""Laplacian variance that scores 0.5.

Handheld phone footage runs roughly: badly out of focus below 50, acceptable around
200-400, crisp above 1000.
"""

MOTION_MIDPOINT = 12.0
"""Mean absolute frame difference that scores 0.5, on a 0-255 scale."""

_LAPLACIAN_MIN_SIZE = 3


def sharpness(frame: Frame) -> float:
    """How much high-frequency detail the frame has, as variance of its Laplacian.

    The standard focus measure: a blurred image has little rapid intensity change, so
    the second derivative is small everywhere and its variance collapses.

    Note what this cannot tell you - a sharp photo of a boring wall scores well. It
    answers "is this usable", never "is this interesting".
    """
    if frame.shape[0] < _LAPLACIAN_MIN_SIZE or frame.shape[1] < _LAPLACIAN_MIN_SIZE:
        return 0.0
    laplacian = (
        frame[:-2, 1:-1]
        + frame[2:, 1:-1]
        + frame[1:-1, :-2]
        + frame[1:-1, 2:]
        - 4.0 * frame[1:-1, 1:-1]
    )
    return _saturate(float(laplacian.var()), SHARPNESS_MIDPOINT)


def exposure(frame: Frame) -> float:
    """Mean brightness, 0-1. Around 0.5 is well exposed; the ends are crushed or blown.

    Deliberately not a "good/bad" score - a night shot is legitimately dark. Callers
    decide what range they want, which is why this returns the measurement rather than
    a verdict.
    """
    if frame.size == 0:
        return 0.0
    return float(np.clip(frame.mean() / 255.0, 0.0, 1.0))


def clipping(frame: Frame, *, low: float = 4.0, high: float = 251.0) -> float:
    """Fraction of pixels stuck at pure black or pure white.

    Distinguishes a dark shot, which is fine, from a crushed one where the detail is
    gone and no grade will bring it back.
    """
    if frame.size == 0:
        return 0.0
    lost = np.count_nonzero((frame <= low) | (frame >= high))
    return float(lost / frame.size)


def motion(before: Frame, after: Frame) -> float:
    """How much changed between two frames, 0-1.

    Covers subject movement and camera movement alike; separating them needs optical
    flow, and for choosing a span the total is what matters. Near-zero means a static
    shot, which is calm but can be dull; very high usually means a whip pan nobody
    wants in the cut.
    """
    if before.shape != after.shape or before.size == 0:
        return 0.0
    return _saturate(float(np.abs(after - before).mean()), MOTION_MIDPOINT)


def shake(motions: Sequence[float]) -> float:
    """Handheld instability, 0-1, from a run of consecutive motion scores.

    The signal is *irregularity*, not amount. A deliberate pan moves a lot but smoothly,
    so its frame-to-frame motion is near constant; a shaky hand jitters, so the motion
    trace jumps around. Measuring the swing between consecutive readings separates the
    two, where mean motion alone would punish every pan.
    """
    if len(motions) < 3:
        return 0.0
    jerk = np.abs(np.diff(np.asarray(motions, dtype=np.float64)))
    return float(np.clip(jerk.mean() * 4.0, 0.0, 1.0))


def to_greyscale(rgb: NDArray[np.uint8]) -> Frame:
    """Rec. 601 luma. Perceptual weighting matters: a plain channel average would rate
    a saturated blue frame as far brighter than the eye sees it."""
    if rgb.ndim == 2:
        return rgb.astype(np.float64)
    weights = np.array([0.299, 0.587, 0.114])
    return rgb[..., :3].astype(np.float64) @ weights


def _saturate(value: float, midpoint: float) -> float:
    """Map an unbounded positive quantity onto 0-1, reaching 0.5 at `midpoint`.

    A hard cap would flatten every good frame to 1.0 and lose the ordering between
    them. This keeps the whole range distinguishable while never exceeding 1.
    """
    if value <= 0.0 or midpoint <= 0.0:
        return 0.0
    return float(value / (value + midpoint))
