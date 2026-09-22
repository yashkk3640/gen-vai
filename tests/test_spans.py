"""Span detection: finding the good seconds inside a clip.

The single thing that decides whether the output is watchable. Tests build synthetic
frame traces with a known good stretch and check it gets found.
"""

import pytest

from genvai.analysis import FrameScore, find_spans, frame_score


def _trace(pattern: list[tuple[float, float, float]], *, spacing: float = 0.25) -> list[FrameScore]:
    """Build a frame trace from (sharpness, exposure, motion) triples."""
    return [
        FrameScore(at=i * spacing, sharpness=s, exposure=e, motion=m)
        for i, (s, e, m) in enumerate(pattern)
    ]


def _uniform(count: int, sharpness: float, exposure: float, motion: float) -> list:
    return [(sharpness, exposure, motion)] * count


# ------------------------------------------------------------------- frame scoring


def test_sharp_beats_blurry() -> None:
    sharp = FrameScore(at=0, sharpness=0.9, exposure=0.5, motion=0.1)
    blurry = FrameScore(at=0, sharpness=0.1, exposure=0.5, motion=0.1)
    assert frame_score(sharp) > frame_score(blurry)


def test_well_exposed_beats_blown_out() -> None:
    good = FrameScore(at=0, sharpness=0.7, exposure=0.5, motion=0.1)
    blown = FrameScore(at=0, sharpness=0.7, exposure=1.0, motion=0.1)
    assert frame_score(good) > frame_score(blown)


def test_crushed_loses_as_much_as_blown() -> None:
    dark = FrameScore(at=0, sharpness=0.7, exposure=0.0, motion=0.1)
    blown = FrameScore(at=0, sharpness=0.7, exposure=1.0, motion=0.1)
    assert frame_score(dark) == pytest.approx(frame_score(blown))


def test_a_still_moment_is_not_penalised() -> None:
    """Low motion is a calm shot, not a defect."""
    still = FrameScore(at=0, sharpness=0.8, exposure=0.5, motion=0.0)
    moving = FrameScore(at=0, sharpness=0.8, exposure=0.5, motion=0.3)
    assert frame_score(still) == pytest.approx(frame_score(moving))


def test_whip_pan_is_penalised() -> None:
    smeared = FrameScore(at=0, sharpness=0.8, exposure=0.5, motion=0.95)
    calm = FrameScore(at=0, sharpness=0.8, exposure=0.5, motion=0.2)
    assert frame_score(smeared) < frame_score(calm)


# --------------------------------------------------------------------- span picking


def test_finds_the_good_stretch_in_a_mostly_bad_clip() -> None:
    """The real shape of a phone clip: fumbling, then something worth keeping."""
    pattern = (
        _uniform(16, 0.1, 0.5, 0.8)  # 0.0-4.0s  blurry and thrashing
        + _uniform(12, 0.9, 0.5, 0.2)  # 4.0-7.0s  sharp and steady
        + _uniform(12, 0.1, 0.5, 0.8)  # 7.0-10.0s blurry again
    )
    spans = find_spans(_trace(pattern), duration=10.0, target=2.0)

    assert spans, "a clearly good stretch must be found"
    best = spans[0]
    assert best.start >= 3.5
    assert best.end <= 7.5


def test_best_span_comes_first() -> None:
    pattern = (
        _uniform(10, 0.4, 0.5, 0.2) + _uniform(10, 0.95, 0.5, 0.2) + _uniform(10, 0.4, 0.5, 0.2)
    )
    spans = find_spans(_trace(pattern), duration=7.5, target=2.0, max_spans=3)
    assert spans[0].score == max(s.score for s in spans)


def test_candidates_do_not_overlap() -> None:
    """Selection needs genuine alternatives, not three views of the same moment."""
    spans = find_spans(_trace(_uniform(40, 0.8, 0.5, 0.2)), duration=10.0, target=2.0)
    ordered = sorted(spans, key=lambda s: s.start)
    for earlier, later in zip(ordered, ordered[1:], strict=False):
        assert earlier.end <= later.start


def test_clip_edges_are_avoided_when_the_middle_is_equally_good() -> None:
    """Phone clips are shakiest with a thumb still on the button."""
    spans = find_spans(_trace(_uniform(40, 0.8, 0.5, 0.2)), duration=10.0, target=2.0)
    assert spans[0].start > 0.0


def test_steady_beats_jittery_at_equal_sharpness() -> None:
    jittery = [(0.8, 0.5, 0.05 if i % 2 else 0.7) for i in range(14)]
    steady = _uniform(14, 0.8, 0.5, 0.35)
    spans = find_spans(_trace(jittery + steady), duration=7.0, target=1.5)
    assert spans[0].start > 3.0, "the steady half should win"


def test_short_clip_offers_itself_whole() -> None:
    """A clip shorter than the target beat is still usable."""
    spans = find_spans(_trace(_uniform(4, 0.8, 0.5, 0.2)), duration=1.0, target=2.5)
    assert len(spans) == 1
    assert spans[0].start == 0.0
    assert spans[0].end == pytest.approx(1.0)


def test_max_spans_is_respected() -> None:
    spans = find_spans(_trace(_uniform(60, 0.8, 0.5, 0.2)), duration=15.0, target=2.0, max_spans=2)
    assert len(spans) <= 2


def test_empty_trace_yields_nothing() -> None:
    assert find_spans([], duration=5.0) == ()


def test_zero_duration_yields_nothing() -> None:
    assert find_spans(_trace(_uniform(8, 0.8, 0.5, 0.2)), duration=0.0) == ()


def test_spans_stay_inside_the_clip() -> None:
    spans = find_spans(_trace(_uniform(40, 0.8, 0.5, 0.2)), duration=10.0, target=2.0)
    for span in spans:
        assert 0.0 <= span.start < span.end <= 10.0


def test_scores_are_bounded() -> None:
    spans = find_spans(_trace(_uniform(40, 1.0, 0.5, 0.0)), duration=10.0, target=2.0)
    assert all(0.0 <= s.score <= 1.0 for s in spans)


# ------------------------------------------------------------------------- reasons


def test_reason_describes_a_steady_sharp_shot() -> None:
    spans = find_spans(_trace(_uniform(40, 0.9, 0.5, 0.05)), duration=10.0, target=2.0)
    assert "sharp" in spans[0].reason
    assert "steady" in spans[0].reason


def test_reason_flags_soft_focus() -> None:
    spans = find_spans(_trace(_uniform(40, 0.15, 0.5, 0.05)), duration=10.0, target=2.0)
    assert "soft focus" in spans[0].reason


def test_reason_flags_unsteady_footage() -> None:
    jittery = [(0.8, 0.5, 0.0 if i % 2 else 0.9) for i in range(40)]
    spans = find_spans(_trace(jittery), duration=10.0, target=2.0)
    assert "unsteady" in spans[0].reason
