"""Tests for the pure core. These are the correctness-critical parts of the project,
and they run with no models, no GPU and no ffmpeg installed.
"""

import pytest
from pydantic import ValidationError

from genvai.timeline import (
    AssetVisual,
    Canvas,
    Captions,
    ClipVisual,
    Export,
    GeneratedVisual,
    Music,
    SafeArea,
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


# --------------------------------------------------------------- short-form contract


def test_default_pacing_targets_short_form() -> None:
    """A default beat should be short-form length, not essay length."""
    assert 1.5 <= Scene(id="s1", visual=GeneratedVisual(prompt="x")).duration <= 2.5


def test_hook_is_discoverable() -> None:
    timeline = Timeline(
        intent="test",
        scenes=(
            Scene(id="s1", role="hook", visual=GeneratedVisual(prompt="x")),
            Scene(id="s2", visual=GeneratedVisual(prompt="y")),
        ),
    )
    assert timeline.hook is not None
    assert timeline.hook.id == "s1"


def test_hook_is_none_when_unmarked() -> None:
    """Planning that never chose an opening is visible rather than silently fine."""
    assert Timeline(intent="test", scenes=(_scene("s1"),)).hook is None


def test_scenes_default_to_body_role() -> None:
    assert Scene(id="s1", visual=GeneratedVisual(prompt="x")).role == "body"


def test_mean_scene_duration_reports_pacing() -> None:
    timeline = Timeline(intent="test", scenes=(_scene("s1", 2.0), _scene("s2", 3.0)))
    assert timeline.mean_scene_duration == pytest.approx(2.5)


def test_mean_scene_duration_is_zero_when_empty() -> None:
    assert Timeline(intent="test").mean_scene_duration == 0.0


def test_captions_default_to_word_level() -> None:
    """Kinetic captions are the short-form convention, so they are the default."""
    assert Captions().mode == "word"


def test_export_emits_a_narration_only_cut_by_default() -> None:
    """The cut to upload when a trending sound is attached in-app. See docs D4 / B4."""
    assert "narration_only" in Export().audio_variants


def test_export_does_not_loop_by_default() -> None:
    assert Export().seamless_loop is False


def test_safe_area_leaves_a_usable_band() -> None:
    """Top and bottom insets must not consume the whole frame."""
    area = SafeArea()
    assert area.top + area.bottom < 1.0


def test_style_suffix_defaults_to_empty() -> None:
    assert Timeline(intent="test").style_suffix == ""


def test_canvas_orientation() -> None:
    assert Canvas().is_vertical
    assert not Canvas(width=1920, height=1080).is_vertical


# ------------------------------------------------------------- camera roll contract


def test_clip_span_is_the_edit() -> None:
    clip = ClipVisual(asset_id="v1", source_start=3.0, source_end=5.4)
    assert clip.source_duration == pytest.approx(2.4)
    assert clip.output_duration == pytest.approx(2.4)


def test_slow_motion_stretches_output() -> None:
    clip = ClipVisual(asset_id="v", source_end=2.0, speed=0.5)
    assert clip.output_duration == pytest.approx(4.0)


def test_speed_up_shortens_output() -> None:
    clip = ClipVisual(asset_id="v", source_end=6.0, speed=2.0)
    assert clip.output_duration == pytest.approx(3.0)


def test_inverted_span_has_no_duration() -> None:
    """A malformed trim yields zero, not a negative that would corrupt the timeline."""
    clip = ClipVisual(asset_id="v", source_start=5.0, source_end=2.0)
    assert clip.source_duration == 0.0


def test_clip_audio_is_muted_by_default() -> None:
    """Camera-roll audio is usually wind and chatter."""
    assert ClipVisual(asset_id="v", source_end=2.0).mute is True


def test_clips_and_photos_share_one_timeline() -> None:
    timeline = Timeline(
        intent="trip reel",
        scenes=(
            Scene(id="s1", duration=2.4, visual=ClipVisual(asset_id="v1", source_end=2.4)),
            Scene(id="s2", duration=2.0, visual=AssetVisual(asset_id="p1")),
        ),
    )
    assert Timeline.model_validate_json(timeline.model_dump_json()) == timeline
    assert timeline.duration == pytest.approx(4.4)


def test_cuts_snap_to_beat_by_default() -> None:
    assert Export().snap_cuts_to_beat is True
