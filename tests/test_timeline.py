"""Tests for the pure core. These are the correctness-critical parts of the project,
and they run with no models, no GPU and no ffmpeg installed.
"""

import pytest
from pydantic import ValidationError

from genvai.timeline import (
    Canvas,
    GeneratedVisual,
    Music,
    Scene,
    Timeline,
    Transition,
)


def _scene(scene_id: str, duration: float = 4.0, transition_in: Transition | None = None) -> Scene:
    return Scene(
        id=scene_id,
        duration=duration,
        visual=GeneratedVisual(prompt=f"prompt for {scene_id}"),
        transition_in=transition_in or Transition(),
    )


def test_timeline_is_immutable() -> None:
    timeline = Timeline(intent="test", scenes=(_scene("s1"),))
    with pytest.raises(ValidationError):
        timeline.intent = "changed"  # type: ignore[misc]


def test_duration_sums_scenes() -> None:
    timeline = Timeline(intent="test", scenes=(_scene("s1", 3.0), _scene("s2", 2.5)))
    assert timeline.duration == pytest.approx(5.5)


def test_transitions_overlap_rather_than_extend() -> None:
    """A transition blends into the previous scene, so it shortens total runtime."""
    timeline = Timeline(
        intent="test",
        scenes=(
            _scene("s1", 3.0),
            _scene("s2", 3.0, Transition(kind="fade", duration=0.5)),
        ),
    )
    assert timeline.duration == pytest.approx(5.5)


def test_leading_transition_does_not_subtract() -> None:
    """There is nothing before the first scene to overlap with."""
    timeline = Timeline(
        intent="test",
        scenes=(_scene("s1", 3.0, Transition(kind="fade", duration=0.5)),),
    )
    assert timeline.duration == pytest.approx(3.0)


def test_unresolved_visuals_excludes_generated_assets() -> None:
    timeline = Timeline(
        intent="test",
        scenes=(
            _scene("s1"),
            Scene(id="s2", visual=GeneratedVisual(prompt="done", asset_id="img_abc")),
        ),
    )
    assert tuple(s.id for s in timeline.unresolved_visuals) == ("s1",)


def test_scene_lookup_returns_none_when_absent() -> None:
    timeline = Timeline(intent="test", scenes=(_scene("s1"),))
    assert timeline.scene("s1") is not None
    assert timeline.scene("nope") is None


def test_canvas_rejects_odd_dimensions() -> None:
    """H.264 requires even dimensions; catching it here beats an ffmpeg failure."""
    with pytest.raises(ValidationError):
        Canvas(width=1081, height=1920)


@pytest.mark.parametrize("state", ["none", "declined", "resolved"])
def test_renderable_music_states(state: str) -> None:
    assert Music(state=state).is_renderable  # type: ignore[arg-type]


@pytest.mark.parametrize("state", ["suggested", "candidates_ready", "approved"])
def test_pending_music_states_block_rendering(state: str) -> None:
    """Rendering mid-negotiation means the user was never asked. See docs D4."""
    assert not Music(state=state).is_renderable  # type: ignore[arg-type]


def test_unknown_field_is_rejected() -> None:
    """extra='forbid' stops an LLM inventing fields that silently do nothing."""
    with pytest.raises(ValidationError):
        Timeline(intent="test", bogus_field=1)  # type: ignore[call-arg]


def test_roundtrips_through_json() -> None:
    timeline = Timeline(
        intent="test", scenes=(_scene("s1"),), canvas=Canvas(width=1920, height=1080)
    )
    assert Timeline.model_validate_json(timeline.model_dump_json()) == timeline
