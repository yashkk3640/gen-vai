"""Reframing: finding the subject, and cropping around it.

Synthetic frames with the subject in a known place, so "did it find the subject" has an
answer rather than an opinion.
"""

import numpy as np
import pytest

from genvai.reframe import (
    DRIFT_THRESHOLD,
    crop_for,
    focus,
    needs_reframing,
    saliency,
    solve_path,
)

WIDE = 16 / 9
TALL = 9 / 16


def _frame(width: int = 320, height: int = 180, background: float = 40.0) -> np.ndarray:
    return np.full((height, width), background)


def _subject(at_x: float, at_y: float = 0.5, *, size: int = 40, width: int = 320) -> np.ndarray:
    """A textured patch on a plain background - something that stands out."""
    frame = _frame(width=width)
    rows, columns = frame.shape
    cx, cy = int(at_x * columns), int(at_y * rows)
    half = size // 2
    patch = np.indices((size, size)).sum(axis=0) % 2 * 200.0 + 40.0
    top, left = max(0, cy - half), max(0, cx - half)
    frame[top : top + size, left : left + size] = patch[: rows - top, : columns - left]
    return frame


# -------------------------------------------------------------------------- saliency


def test_saliency_lights_up_the_subject() -> None:
    heat = saliency(_subject(0.25))
    left, right = heat[:, : heat.shape[1] // 2], heat[:, heat.shape[1] // 2 :]
    assert left.sum() > right.sum()


def test_a_blank_frame_has_no_focus() -> None:
    """Nothing stands out, so nothing should be preferred."""
    x, y = focus(_frame())
    assert x == pytest.approx(0.5, abs=0.12)


def test_saliency_is_bounded() -> None:
    heat = saliency(_subject(0.5))
    assert heat.min() >= 0.0
    assert heat.max() == pytest.approx(1.0)


def test_an_empty_frame_is_survivable() -> None:
    assert saliency(np.zeros((0, 0))).size == 0


# ----------------------------------------------------------------------------- focus


@pytest.mark.parametrize("position", [0.2, 0.35, 0.5, 0.65, 0.8])
def test_focus_follows_the_subject(position: float) -> None:
    x, _ = focus(_subject(position))
    assert x == pytest.approx(position, abs=0.05)


@pytest.mark.parametrize("position", [0.10, 0.15, 0.85, 0.90])
def test_focus_works_at_the_frame_edges(position: float) -> None:
    """The failure that ruled out spectral residual.

    Its FFT wraps, so a subject against the border aliases to the opposite side - one at
    15% across the frame was reported at 60%, which would have cropped to exactly the
    wrong half.
    """
    x, _ = focus(_subject(position))
    assert x == pytest.approx(position, abs=0.05)


def test_a_flat_background_contributes_nothing() -> None:
    """Without the background floor, an even pedestal drags every answer to the middle."""
    from genvai.reframe import saliency

    heat = saliency(_subject(0.2))
    assert float(np.median(heat)) == 0.0


def test_focus_is_weighted_not_winner_takes_all() -> None:
    """One specular highlight must not drag the frame off the subject."""
    frame = _subject(0.3, size=70)
    frame[10:13, 300:303] = 255.0
    x, _ = focus(frame)
    assert x < 0.5


def test_focus_stays_inside_the_frame() -> None:
    for position in (0.02, 0.98):
        x, y = focus(_subject(position))
        assert 0.0 <= x <= 1.0
        assert 0.0 <= y <= 1.0


# ------------------------------------------------------------------------ cropping


def test_a_wide_source_is_cropped_at_the_sides() -> None:
    x, y, w, h = crop_for((0.5, 0.5), WIDE, TALL)
    assert h == 1.0, "full height is kept"
    assert w < 1.0
    assert x == pytest.approx((1 - w) / 2)


def test_a_tall_source_is_cropped_top_and_bottom() -> None:
    x, y, w, h = crop_for((0.5, 0.5), TALL, WIDE)
    assert w == 1.0
    assert h < 1.0


def test_the_crop_follows_the_subject() -> None:
    left = crop_for((0.2, 0.5), WIDE, TALL)
    right = crop_for((0.8, 0.5), WIDE, TALL)
    assert left[0] < right[0]


def test_a_subject_at_the_edge_yields_an_edge_crop() -> None:
    """Clamped to the frame, rather than hanging off it."""
    x, _, w, _ = crop_for((0.02, 0.5), WIDE, TALL)
    assert x == 0.0
    x, _, w, _ = crop_for((0.98, 0.5), WIDE, TALL)
    assert x == pytest.approx(1.0 - w)


def test_matching_shapes_keep_the_whole_frame() -> None:
    assert crop_for((0.5, 0.5), TALL, TALL) == (0.0, 0.0, 1.0, 1.0)


def test_the_crop_is_the_largest_that_fits() -> None:
    """Nothing thrown away that did not have to be."""
    _, _, w, h = crop_for((0.5, 0.5), WIDE, TALL)
    assert w == pytest.approx(TALL / WIDE)
    assert h == 1.0


def test_degenerate_aspects_are_survivable() -> None:
    assert crop_for((0.5, 0.5), 0.0, TALL) == (0.0, 0.0, 1.0, 1.0)


# ------------------------------------------------------------------------- the path


def test_a_still_subject_gets_a_still_crop() -> None:
    frames = [_subject(0.3) for _ in range(8)]
    start, end = solve_path(frames, WIDE, TALL)
    assert end is None, "a moving crop is only worth its cost when the subject moves"
    assert start[0] < 0.2


def test_a_travelling_subject_earns_a_pan() -> None:
    frames = [_subject(0.15 + i * 0.09) for i in range(8)]
    start, end = solve_path(frames, WIDE, TALL)
    assert end is not None
    assert end[0] > start[0], "the crop follows the subject across the frame"


def test_a_small_wobble_does_not_move_the_frame() -> None:
    """A frame that creeps reads as a fault in the render, not as camera work."""
    frames = [_subject(0.5 + (0.01 if i % 2 else -0.01)) for i in range(8)]
    assert solve_path(frames, WIDE, TALL)[1] is None


def test_the_drift_threshold_is_what_decides() -> None:
    below = [_subject(0.4 + i * DRIFT_THRESHOLD / 20) for i in range(8)]
    above = [_subject(0.2 + i * DRIFT_THRESHOLD / 2) for i in range(8)]
    assert solve_path(below, WIDE, TALL)[1] is None
    assert solve_path(above, WIDE, TALL)[1] is not None


def test_no_frames_yields_the_whole_picture() -> None:
    assert solve_path([], WIDE, TALL) == ((0.0, 0.0, 1.0, 1.0), None)


def test_a_single_frame_is_a_still_crop() -> None:
    assert solve_path([_subject(0.3)], WIDE, TALL)[1] is None


def test_the_path_stays_inside_the_frame() -> None:
    frames = [_subject(0.05 + i * 0.12) for i in range(8)]
    start, end = solve_path(frames, WIDE, TALL)
    for rect in (start, end):
        if rect is None:
            continue
        x, y, w, h = rect
        assert x >= 0.0 and x + w <= 1.0 + 1e-9
        assert y >= 0.0 and y + h <= 1.0 + 1e-9


# ---------------------------------------------------------------------- shortcuts


def test_reframing_is_skipped_when_shapes_match() -> None:
    assert needs_reframing(TALL, TALL) is False
    assert needs_reframing(1.777, 1.778) is False


def test_reframing_is_needed_for_landscape_into_vertical() -> None:
    assert needs_reframing(WIDE, TALL) is True


# ----------------------------------------------------------------- the crop filter


def test_a_still_crop_is_a_plain_filter() -> None:
    from genvai.adapters.ffmpeg import _crop

    assert _crop((0.2, 0.0, 0.6, 1.0)) == "crop=iw*0.6:ih*1.0:iw*0.2:ih*0.0"


def test_no_crop_is_a_passthrough() -> None:
    from genvai.adapters.ffmpeg import _crop

    assert _crop(None) == "null"


def test_a_moving_crop_varies_with_time() -> None:
    from genvai.adapters.ffmpeg import _crop

    built = _crop((0.1, 0.0, 0.6, 1.0), (0.3, 0.0, 0.6, 1.0), 2.0)
    assert "t/2.0000" in built
    assert "eval=frame" not in built, "crop has no eval option and needs none"


def test_a_crop_that_would_resize_falls_back_to_still() -> None:
    """ffmpeg fixes the output size once; only the position may move."""
    from genvai.adapters.ffmpeg import _crop

    built = _crop((0.1, 0.0, 0.6, 1.0), (0.3, 0.0, 0.4, 1.0), 2.0)
    assert "t/" not in built


def test_a_moving_crop_with_no_duration_falls_back() -> None:
    from genvai.adapters.ffmpeg import _crop

    assert "t/" not in _crop((0.1, 0.0, 0.6, 1.0), (0.3, 0.0, 0.6, 1.0), 0.0)


# ------------------------------------------------------------------- the pipeline


def _clip_timeline(canvas_w: int, canvas_h: int, crop=None):
    from genvai.timeline import (
        Asset,
        AssetProvenance,
        Canvas,
        ClipVisual,
        Scene,
        Timeline,
    )

    return Timeline(
        intent="t",
        canvas=Canvas(width=canvas_w, height=canvas_h),
        scenes=(
            Scene(
                id="s1",
                duration=2.0,
                visual=ClipVisual(asset_id="a", source_start=0.2, source_end=2.2, crop=crop),
            ),
        ),
        assets={
            "a": Asset(
                kind="video",
                path="a.mp4",
                sha256="a",
                provenance=AssetProvenance(provider="test"),
            )
        },
    )


def test_a_matching_shape_is_left_alone(tmp_path, sample_media) -> None:
    """A vertical clip in a vertical reel must cost nothing at all."""
    from genvai.pipeline.reframe import reframe

    before = _clip_timeline(1280, 720)
    after = reframe(
        before,
        lambda _: sample_media["clip_a"],
        tmp_path / "no-ffmpeg-needed",
        aspects={"a": 1280 / 720},
    )
    assert after == before


def test_a_landscape_clip_in_a_vertical_reel_is_cropped(sample_media, ffmpeg_binary) -> None:
    from genvai.pipeline.reframe import reframe

    after = reframe(
        _clip_timeline(1080, 1920),
        lambda _: sample_media["clip_a"],
        ffmpeg_binary,
        aspects={"a": 1280 / 720},
    )
    crop = after.scenes[0].visual.crop
    assert crop is not None
    assert crop[2] == pytest.approx((1080 / 1920) / (1280 / 720), rel=0.01)


def test_an_existing_crop_is_not_overwritten(sample_media, ffmpeg_binary) -> None:
    """A crop the user chose is a decision, not a default."""
    from genvai.pipeline.reframe import reframe

    chosen = (0.0, 0.0, 0.5, 1.0)
    after = reframe(
        _clip_timeline(1080, 1920, crop=chosen),
        lambda _: sample_media["clip_a"],
        ffmpeg_binary,
        aspects={"a": 1280 / 720},
    )
    assert after.scenes[0].visual.crop == chosen


def test_an_unknown_aspect_is_left_alone(sample_media, ffmpeg_binary) -> None:
    from genvai.pipeline.reframe import reframe

    before = _clip_timeline(1080, 1920)
    assert reframe(before, lambda _: sample_media["clip_a"], ffmpeg_binary, aspects={}) == before


def test_an_unreadable_asset_does_not_break_the_reel(ffmpeg_binary) -> None:
    """Reframing improves a cut; it must never be why one fails to render."""
    from pathlib import Path

    from genvai.pipeline.reframe import reframe

    before = _clip_timeline(1080, 1920)
    after = reframe(
        before, lambda _: Path("does-not-exist.mp4"), ffmpeg_binary, aspects={"a": 1280 / 720}
    )
    assert after == before
