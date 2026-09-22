"""Edit operations: validation, application, and the diff the user reads.

The safety property that matters is atomicity. A list containing one bad op must leave
the timeline completely untouched, because a half-applied edit puts the project in a
state nobody asked for and nobody can reason about.
"""

import pytest

from genvai.errors import InvalidEditOp
from genvai.ops import (
    EditOp,
    InsertScene,
    NudgeClipSpan,
    RemoveMusic,
    RemoveScene,
    ReorderScenes,
    ReplaceSource,
    ScaleAllDurations,
    SelectMusicCandidate,
    SetCanvas,
    SetCaptions,
    SetClipMuted,
    SetClipSpan,
    SetClipSpeed,
    SetCrop,
    SetMotion,
    SetMusicGain,
    SetMusicQuery,
    SetOverlayText,
    SetSceneDuration,
    SetSceneRole,
    SetTransition,
    apply,
    describe,
    validate,
)
from genvai.timeline import (
    Asset,
    AssetProvenance,
    AssetVisual,
    Canvas,
    ClipVisual,
    KenBurns,
    Music,
    MusicCandidate,
    MusicQuery,
    Scene,
    TextOverlay,
    Timeline,
    Transition,
)


def _asset(name: str) -> Asset:
    return Asset(
        kind="video",
        path=f"assets/{name}.mp4",
        sha256=name,
        provenance=AssetProvenance(provider="test"),
    )


def _timeline() -> Timeline:
    return Timeline(
        intent="a reel",
        scenes=(
            Scene(
                id="s1",
                duration=2.0,
                role="hook",
                visual=ClipVisual(asset_id="aaa", source_start=3.0, source_end=5.0),
                overlays=(TextOverlay(content="first"),),
            ),
            Scene(id="s2", duration=2.2, visual=AssetVisual(asset_id="bbb")),
            Scene(
                id="s3",
                duration=1.5,
                role="payoff",
                visual=ClipVisual(asset_id="ccc", source_end=1.5),
            ),
        ),
        assets={n: _asset(n) for n in ("aaa", "bbb", "ccc", "ddd")},
    )


def _apply(*ops: EditOp) -> Timeline:
    return apply(_timeline(), ops)


# ------------------------------------------------------------------------ atomicity


def test_version_increments_once_per_edit() -> None:
    assert _apply(SetSceneDuration(scene_id="s1", seconds=3.0)).version == 2


def test_a_bad_op_leaves_everything_untouched() -> None:
    """The property the whole design exists for."""
    before = _timeline()
    with pytest.raises(InvalidEditOp):
        apply(
            before,
            (
                SetSceneDuration(scene_id="s1", seconds=4.0),
                RemoveScene(scene_id="does-not-exist"),
            ),
        )
    assert before.scenes[0].duration == 2.0, "the valid op did not sneak through"
    assert before.version == 1


def test_validation_names_the_op_and_the_reason() -> None:
    with pytest.raises(InvalidEditOp, match="no scene 'nope'") as caught:
        validate(_timeline(), (RemoveScene(scene_id="nope"),))
    assert caught.value.op == "remove_scene"


def test_ops_apply_in_order() -> None:
    result = _apply(
        SetSceneDuration(scene_id="s1", seconds=3.0),
        SetSceneDuration(scene_id="s1", seconds=5.0),
    )
    assert result.scene("s1").duration == 5.0  # type: ignore[union-attr]


def test_an_empty_list_is_a_no_op_but_still_versions() -> None:
    result = _apply()
    assert result.version == 2
    assert result.scenes == _timeline().scenes


# --------------------------------------------------------------------------- timing


def test_retime_one_scene() -> None:
    assert _apply(SetSceneDuration(scene_id="s1", seconds=3.5)).scene("s1").duration == 3.5  # type: ignore[union-attr]


def test_out_of_range_duration_is_refused() -> None:
    with pytest.raises(InvalidEditOp, match="outside"):
        _apply(SetSceneDuration(scene_id="s1", seconds=0.05))


def test_scaling_touches_every_scene() -> None:
    result = _apply(ScaleAllDurations(factor=0.5))
    assert [s.duration for s in result.scenes] == [1.0, 1.1, 0.75]


def test_scaling_clamps_rather_than_producing_an_invalid_scene() -> None:
    result = _apply(ScaleAllDurations(factor=0.01))
    assert all(s.duration >= 0.3 for s in result.scenes)


# ---------------------------------------------------------------------------- clips


def test_retrim_moves_the_scene_duration_with_it() -> None:
    """Span and duration must stay consistent or the render desynchronises."""
    result = _apply(SetClipSpan(scene_id="s1", source_start=6.0, source_end=8.5))
    scene = result.scene("s1")
    assert scene.visual.source_start == 6.0  # type: ignore[union-attr]
    assert scene.duration == pytest.approx(2.5)  # type: ignore[union-attr]


def test_a_backwards_span_is_refused() -> None:
    with pytest.raises(InvalidEditOp, match="at or before"):
        _apply(SetClipSpan(scene_id="s1", source_start=8.0, source_end=4.0))


def test_nudge_keeps_the_length() -> None:
    """What users actually ask for: 'show a bit more before that'."""
    result = _apply(NudgeClipSpan(scene_id="s1", seconds=-1.0))
    visual = result.scene("s1").visual  # type: ignore[union-attr]
    assert (visual.source_start, visual.source_end) == (2.0, 4.0)
    assert result.scene("s1").duration == pytest.approx(2.0)  # type: ignore[union-attr]


def test_nudge_never_goes_before_the_clip_starts() -> None:
    visual = _apply(NudgeClipSpan(scene_id="s1", seconds=-99.0)).scene("s1").visual  # type: ignore[union-attr]
    assert visual.source_start == 0.0
    assert visual.source_duration == pytest.approx(2.0), "the length survives the clamp"


def test_slow_motion_stretches_the_scene() -> None:
    result = _apply(SetClipSpeed(scene_id="s1", speed=0.5))
    assert result.scene("s1").duration == pytest.approx(4.0)  # type: ignore[union-attr]


def test_clip_ops_refuse_a_photo() -> None:
    with pytest.raises(InvalidEditOp, match="not a video clip"):
        _apply(SetClipSpeed(scene_id="s2", speed=2.0))


def test_unmute_a_clip() -> None:
    assert _apply(SetClipMuted(scene_id="s1", mute=False)).scene("s1").visual.mute is False  # type: ignore[union-attr]


def test_reframe_a_photo() -> None:
    result = _apply(SetCrop(scene_id="s2", crop=(0.2, 0.0, 0.6, 1.0), fit="cover"))
    assert result.scene("s2").visual.crop == (0.2, 0.0, 0.6, 1.0)  # type: ignore[union-attr]


def test_crop_can_be_reset() -> None:
    assert _apply(SetCrop(scene_id="s2", crop=None)).scene("s2").visual.crop is None  # type: ignore[union-attr]


def test_swap_the_take_keeps_the_timing() -> None:
    result = _apply(ReplaceSource(scene_id="s1", asset_id="ddd"))
    assert result.scene("s1").visual.asset_id == "ddd"  # type: ignore[union-attr]
    assert result.scene("s1").duration == 2.0  # type: ignore[union-attr]


def test_swapping_to_an_unknown_asset_is_refused() -> None:
    with pytest.raises(InvalidEditOp, match="no asset"):
        _apply(ReplaceSource(scene_id="s1", asset_id="zzz"))


# ------------------------------------------------------------------------ structure


def test_remove_a_scene() -> None:
    result = _apply(RemoveScene(scene_id="s2"))
    assert [s.id for s in result.scenes] == ["s1", "s3"]


def test_the_last_scene_cannot_be_removed() -> None:
    single = Timeline(intent="t", scenes=(Scene(id="only", visual=AssetVisual(asset_id="a")),))
    with pytest.raises(InvalidEditOp, match="only scene"):
        apply(single, (RemoveScene(scene_id="only"),))


def test_insert_after_a_scene() -> None:
    new = Scene(id="s9", visual=AssetVisual(asset_id="ddd"))
    result = _apply(InsertScene(scene=new, after_scene_id="s1"))
    assert [s.id for s in result.scenes] == ["s1", "s9", "s2", "s3"]


def test_insert_at_the_start() -> None:
    new = Scene(id="s0", visual=AssetVisual(asset_id="ddd"))
    assert _apply(InsertScene(scene=new)).scenes[0].id == "s0"


def test_a_duplicate_scene_id_is_refused() -> None:
    clash = Scene(id="s1", visual=AssetVisual(asset_id="ddd"))
    with pytest.raises(InvalidEditOp, match="already taken"):
        _apply(InsertScene(scene=clash))


def test_reorder() -> None:
    result = _apply(ReorderScenes(scene_ids=("s3", "s1", "s2")))
    assert [s.id for s in result.scenes] == ["s3", "s1", "s2"]


def test_a_reorder_that_drops_a_scene_is_refused() -> None:
    """Otherwise a typo silently deletes footage."""
    with pytest.raises(InvalidEditOp, match="every scene exactly once"):
        _apply(ReorderScenes(scene_ids=("s1", "s2")))


def test_a_reorder_that_repeats_a_scene_is_refused() -> None:
    with pytest.raises(InvalidEditOp, match="every scene exactly once"):
        _apply(ReorderScenes(scene_ids=("s1", "s1", "s2")))


# ---------------------------------------------------------------------------- roles


def test_promoting_a_hook_demotes_the_old_one() -> None:
    """There can only be one opening shot."""
    result = _apply(SetSceneRole(scene_id="s2", role="hook"))
    roles = {s.id: s.role for s in result.scenes}
    assert roles["s2"] == "hook"
    assert roles["s1"] != "hook"
    assert sum(1 for r in roles.values() if r == "hook") == 1


def test_other_roles_do_not_displace_anyone() -> None:
    result = _apply(SetSceneRole(scene_id="s2", role="cta"))
    assert result.scene("s1").role == "hook"  # type: ignore[union-attr]


# ----------------------------------------------------------------------- appearance


def test_edit_a_caption() -> None:
    result = _apply(SetOverlayText(scene_id="s1", index=0, content="changed"))
    assert result.scene("s1").overlays[0].content == "changed"  # type: ignore[union-attr]


def test_editing_a_missing_overlay_is_refused() -> None:
    with pytest.raises(InvalidEditOp, match="no overlay"):
        _apply(SetOverlayText(scene_id="s2", index=0, content="x"))


def test_reframe_the_canvas() -> None:
    result = _apply(SetCanvas(canvas=Canvas(width=1920, height=1080)))
    assert (result.canvas.width, result.canvas.height) == (1920, 1080)


def test_toggle_captions_without_touching_the_mode() -> None:
    result = _apply(SetCaptions(enabled=False))
    assert result.captions.enabled is False
    assert result.captions.mode == "word", "an unset field is left alone"


def test_change_the_transition() -> None:
    result = _apply(SetTransition(scene_id="s2", transition=Transition(kind="fade", duration=0.4)))
    assert result.scene("s2").transition_in.kind == "fade"  # type: ignore[union-attr]


def test_change_the_camera_move() -> None:
    result = _apply(SetMotion(scene_id="s2", motion=KenBurns()))
    assert result.scene("s2").motion.kind == "ken_burns"  # type: ignore[union-attr]


# ---------------------------------------------------------------------------- music


def test_asking_for_different_music_resets_the_negotiation() -> None:
    """New query means the old candidates and choice are stale."""
    started = _timeline().model_copy(
        update={
            "music": Music(
                state="resolved",
                asset_id="old",
                candidates=(
                    MusicCandidate(id="c1", title="t", source_url="u", licence="cc", duration=60.0),
                ),
            )
        }
    )
    result = apply(started, (SetMusicQuery(query=MusicQuery(mood="calm")),))
    assert result.music.state == "suggested"
    assert result.music.candidates == ()
    assert result.music.asset_id is None


def test_approving_a_candidate_does_not_download_it() -> None:
    """`approved` permits the fetch; it is not the fetch. See docs/decisions.md."""
    with_candidates = _timeline().model_copy(
        update={
            "music": Music(
                state="candidates_ready",
                candidates=(
                    MusicCandidate(id="c1", title="t", source_url="u", licence="cc", duration=60.0),
                ),
            )
        }
    )
    result = apply(with_candidates, (SelectMusicCandidate(candidate_id="c1"),))
    assert result.music.state == "approved"
    assert result.music.asset_id is None, "nothing has been downloaded"


def test_approving_an_unsuggested_track_is_refused() -> None:
    with pytest.raises(InvalidEditOp, match="no candidate"):
        _apply(SelectMusicCandidate(candidate_id="c9"))


def test_dropping_music_is_renderable() -> None:
    result = _apply(RemoveMusic())
    assert result.music.state == "declined"
    assert result.music.is_renderable


def test_change_the_music_level() -> None:
    assert _apply(SetMusicGain(gain_db=-6.0)).music.gain_db == -6.0


# ----------------------------------------------------------------------- describing


def test_scenes_are_described_by_position_not_id() -> None:
    """ "shot 2" means something to someone who just watched it; "s2" does not."""
    lines = describe(_timeline(), (SetSceneDuration(scene_id="s2", seconds=3.0),))
    assert lines == ("shot 2: hold for 3.0s",)


def test_every_op_produces_a_line() -> None:
    ops: tuple[EditOp, ...] = (
        RemoveScene(scene_id="s2"),
        SetClipSpeed(scene_id="s1", speed=0.5),
        RemoveMusic(),
    )
    lines = describe(_timeline(), ops)
    assert len(lines) == 3
    assert all(line and not line.startswith("set_") for line in lines)


def test_a_reorder_reads_as_positions() -> None:
    (line,) = describe(_timeline(), (ReorderScenes(scene_ids=("s3", "s1", "s2")),))
    assert line == "reorder to shot 3 then shot 1 then shot 2"


def test_describing_does_not_mutate() -> None:
    before = _timeline()
    describe(before, (RemoveScene(scene_id="s2"),))
    assert len(before.scenes) == 3
