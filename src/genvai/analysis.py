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

from genvai.media import Span
from genvai.timeline import Frozen

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


# --------------------------------------------------------------------------- spans


class FrameScore(Frozen):
    """One sampled frame's measurements, with the time it was taken from."""

    at: float
    sharpness: float
    exposure: float
    motion: float


SHARPNESS_WEIGHT = 0.55
EXPOSURE_WEIGHT = 0.30
MOTION_WEIGHT = 0.15

HIGH_MOTION = 0.55
"""Above this, movement stops reading as life and starts reading as a whip pan."""

EDGE_SECONDS = 0.6
"""How much of a clip's head and tail to treat as suspect.

Phone clips are shakiest while a thumb is still on the button, and the last moment is
usually the hand travelling to stop the recording. Both ends are penalised rather than
cut outright, so a short clip that is *all* edge can still contribute.
"""

SHAKE_PENALTY = 0.35


def frame_score(sample: FrameScore) -> float:
    """How usable one frame is, 0-1.

    Focus dominates, because nothing rescues a blurred frame. Exposure is scored by
    distance from the middle rather than by brightness, so an overexposed frame loses as
    much as a crushed one. Motion only subtracts, and only when it is high enough to
    smear - a still moment is a perfectly good shot.
    """
    exposure_fit = max(0.0, 1.0 - abs(sample.exposure - 0.5) * 1.8)
    motion_fit = (
        1.0 if sample.motion <= HIGH_MOTION else max(0.0, 1.0 - (sample.motion - HIGH_MOTION) * 2.5)
    )
    return (
        SHARPNESS_WEIGHT * sample.sharpness
        + EXPOSURE_WEIGHT * exposure_fit
        + MOTION_WEIGHT * motion_fit
    )


def find_spans(
    samples: Sequence[FrameScore],
    *,
    duration: float,
    target: float = 2.2,
    max_spans: int = 3,
    minimum: float = 0.8,
) -> tuple[Span, ...]:
    """Find the best few stretches of a clip, best first.

    This is where most of the value of camera-roll editing sits: a twenty-second phone
    clip usually holds two seconds worth keeping, and finding them by hand forty times
    over is the job people give up on.

    Returns several candidates rather than one, so selection can take a different span
    when the best one collides with something already chosen.
    """
    if len(samples) < 2 or duration <= 0.0:
        return ()

    window = max(2, round(target / _spacing(samples)))
    if window > len(samples):
        # A clip shorter than the target beat: offer the whole thing.
        score = sum(frame_score(s) for s in samples) / len(samples)
        return (Span(start=0.0, end=duration, score=_clamp(score), reason=_reason(samples)),)

    scored = [
        (_window_score(samples[i : i + window], duration), i)
        for i in range(len(samples) - window + 1)
    ]
    scored.sort(key=lambda pair: pair[0], reverse=True)

    chosen: list[Span] = []
    taken: list[tuple[float, float]] = []
    for score, index in scored:
        block = samples[index : index + window]
        start, end = block[0].at, min(duration, block[-1].at)
        if end - start < minimum or any(_overlaps((start, end), r) for r in taken):
            continue
        chosen.append(Span(start=start, end=end, score=_clamp(score), reason=_reason(block)))
        taken.append((start, end))
        if len(chosen) == max_spans:
            break
    return tuple(chosen)


def _window_score(block: Sequence[FrameScore], duration: float) -> float:
    """Mean frame quality, penalised for jitter and for sitting at the clip's edge."""
    base = sum(frame_score(s) for s in block) / len(block)
    steadiness = 1.0 - SHAKE_PENALTY * shake([s.motion for s in block])
    return base * steadiness * _edge_taper(block[0].at, block[-1].at, duration)


def _edge_taper(start: float, end: float, duration: float) -> float:
    """Scale down windows that reach into the suspect head or tail of a clip."""
    head = min(1.0, start / EDGE_SECONDS) if EDGE_SECONDS > 0 else 1.0
    tail = min(1.0, (duration - end) / EDGE_SECONDS) if EDGE_SECONDS > 0 else 1.0
    return 0.7 + 0.3 * min(head, tail)


def _reason(block: Sequence[FrameScore]) -> str:
    """A short, honest explanation. The user sees this when asking why a clip was used."""
    focus = sum(s.sharpness for s in block) / len(block)
    movement = sum(s.motion for s in block) / len(block)
    jitter = shake([s.motion for s in block])

    parts = ["sharp" if focus > 0.5 else "soft focus"]
    if jitter > 0.5:
        parts.append("unsteady")
    elif movement > HIGH_MOTION:
        parts.append("fast movement")
    elif movement > 0.15:
        parts.append("some movement")
    else:
        parts.append("steady")
    return ", ".join(parts)


def _spacing(samples: Sequence[FrameScore]) -> float:
    gaps = [b.at - a.at for a, b in zip(samples, samples[1:], strict=False) if b.at > a.at]
    return sum(gaps) / len(gaps) if gaps else 1.0


def _overlaps(a: tuple[float, float], b: tuple[float, float]) -> bool:
    return a[0] < b[1] and b[0] < a[1]


def _clamp(value: float) -> float:
    return float(min(1.0, max(0.0, value)))
