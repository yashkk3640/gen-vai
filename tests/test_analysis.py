"""Measurement tests, against synthetic frames whose right answer is known.

These scores decide what ends up in the video, so each one is checked for *ordering*
- sharp beats blurry, jitter beats a smooth pan - rather than against magic numbers that
would break the moment a constant is retuned.
"""

import numpy as np
import pytest

from genvai.analysis import (
    clipping,
    exposure,
    motion,
    shake,
    sharpness,
    to_greyscale,
)

RNG = np.random.default_rng(20260922)


def _flat(value: float = 128.0, size: int = 64) -> np.ndarray:
    return np.full((size, size), value, dtype=np.float64)


def _checkerboard(size: int = 64, cell: int = 4) -> np.ndarray:
    """Maximum high-frequency detail: the sharpest thing an image can be."""
    grid = np.indices((size, size)).sum(axis=0) // cell
    return np.where(grid % 2 == 0, 0.0, 255.0)


def _blurred(frame: np.ndarray, passes: int = 6) -> np.ndarray:
    """Repeated box blur, using only the primitives under test."""
    out = frame.copy()
    for _ in range(passes):
        padded = np.pad(out, 1, mode="edge")
        out = (padded[:-2, 1:-1] + padded[2:, 1:-1] + padded[1:-1, :-2] + padded[1:-1, 2:]) / 4.0
    return out


# ---------------------------------------------------------------------- sharpness


def test_detail_scores_higher_than_blur() -> None:
    assert sharpness(_checkerboard()) > sharpness(_blurred(_checkerboard()))


def test_flat_frame_has_no_sharpness() -> None:
    assert sharpness(_flat()) == pytest.approx(0.0, abs=1e-9)


def test_sharpness_is_bounded() -> None:
    assert 0.0 <= sharpness(_checkerboard()) < 1.0


def test_tiny_frame_does_not_crash() -> None:
    """Degenerate input must return a score, not raise, or ingest dies on one odd file."""
    assert sharpness(np.zeros((2, 2))) == 0.0


def test_sharpness_ignores_brightness() -> None:
    """A dark but detailed shot is still in focus."""
    bright = _checkerboard()
    dark = bright * 0.4
    assert sharpness(dark) > 0.3


# ----------------------------------------------------------------------- exposure


def test_exposure_tracks_brightness() -> None:
    assert exposure(_flat(0.0)) < exposure(_flat(128.0)) < exposure(_flat(255.0))


def test_mid_grey_is_mid_exposure() -> None:
    assert exposure(_flat(128.0)) == pytest.approx(0.5, abs=0.01)


def test_exposure_is_bounded() -> None:
    assert exposure(_flat(255.0)) == pytest.approx(1.0)
    assert exposure(_flat(0.0)) == pytest.approx(0.0)


# ----------------------------------------------------------------------- clipping


def test_clipping_finds_crushed_pixels() -> None:
    frame = _flat(128.0)
    frame[:32] = 0.0
    assert clipping(frame) == pytest.approx(0.5, abs=0.01)


def test_a_dark_but_intact_frame_is_not_clipped() -> None:
    """Night footage is legitimate; only lost detail counts."""
    assert clipping(_flat(30.0)) == 0.0


def test_blown_highlights_count() -> None:
    assert clipping(_flat(255.0)) == pytest.approx(1.0)


# ------------------------------------------------------------------------- motion


def test_identical_frames_have_no_motion() -> None:
    frame = _checkerboard()
    assert motion(frame, frame) == 0.0


def test_bigger_change_scores_higher() -> None:
    base = _flat(100.0)
    small = _flat(105.0)
    large = _flat(200.0)
    assert motion(base, small) < motion(base, large)


def test_motion_is_bounded() -> None:
    assert 0.0 <= motion(_flat(0.0), _flat(255.0)) < 1.0


def test_mismatched_shapes_are_survivable() -> None:
    assert motion(_flat(size=32), _flat(size=64)) == 0.0


# -------------------------------------------------------------------------- shake


def test_smooth_pan_is_not_shake() -> None:
    """The whole point: a deliberate pan moves a lot, but evenly."""
    steady = [0.4] * 12
    assert shake(steady) == pytest.approx(0.0, abs=1e-9)


def test_jitter_is_shake() -> None:
    jittery = [0.05, 0.6, 0.1, 0.7, 0.05, 0.65, 0.1, 0.72]
    assert shake(jittery) > 0.5


def test_a_fast_smooth_pan_beats_a_slow_jittery_one() -> None:
    fast_smooth = [0.8] * 10
    slow_jittery = [0.0, 0.3, 0.0, 0.3, 0.0, 0.3, 0.0, 0.3]
    assert shake(fast_smooth) < shake(slow_jittery)


def test_shake_needs_a_run_of_frames() -> None:
    assert shake([0.5, 0.1]) == 0.0


def test_shake_is_bounded() -> None:
    wild = list(RNG.uniform(0, 1, 40))
    assert 0.0 <= shake(wild) <= 1.0


# ---------------------------------------------------------------------- greyscale


def test_greyscale_uses_perceptual_weights() -> None:
    """A flat channel average would call saturated blue as bright as green."""
    green = to_greyscale(np.full((8, 8, 3), [0, 255, 0], dtype=np.uint8))
    blue = to_greyscale(np.full((8, 8, 3), [0, 0, 255], dtype=np.uint8))
    assert green.mean() > blue.mean()


def test_greyscale_passes_through_single_channel() -> None:
    already = np.full((8, 8), 120, dtype=np.uint8)
    assert to_greyscale(already).mean() == pytest.approx(120.0)


def test_greyscale_ignores_alpha() -> None:
    rgba = np.full((8, 8, 4), [255, 255, 255, 0], dtype=np.uint8)
    assert to_greyscale(rgba).mean() == pytest.approx(255.0, abs=0.5)
