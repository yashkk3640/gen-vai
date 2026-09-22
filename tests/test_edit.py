"""Translating a change request into operations.

The model speaks in shot numbers and plain verbs; `to_ops` turns that into the typed
vocabulary. These tests pin the translation, which is where a misread request is caught
- no model needed to run them.
"""

import pytest

from genvai.errors import GenvaiError
from genvai.ops import (
    NudgeClipSpan,
    RemoveMusic,
    RemoveScene,
    ReorderScenes,
    ScaleAllDurations,
    SetClipMuted,
    SetClipSpeed,
    SetMusicGain,
    SetSceneCaption,
    SetSceneDuration,
    SetSceneRole,
    apply,
)
from genvai.pipeline.edit import EditCommand, EditRequest, edit, summarise, to_ops
from genvai.timeline import (
    AssetVisual,
    ClipVisual,
    Music,
    Scene,
    TextOverlay,
    Timeline,
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
                note="sharp, steady",
            ),
            Scene(
                id="s2",
                duration=2.2,
                visual=AssetVisual(asset_id="bbb"),
                overlays=(TextOverlay(content="day one"),),
            ),
            Scene(
                id="s3",
                duration=1.5,
                role="payoff",
                visual=ClipVisual(asset_id="ccc", source_end=1.5),
            ),
        ),
    )


def _request(*commands: EditCommand) -> EditRequest:
    return EditRequest(commands=commands)


def _ops(*commands: EditCommand):
    return to_ops(_request(*commands), _timeline())


# ----------------------------------------------------------------------- summary


def test_summary_numbers_the_shots() -> None:
    """The model answers a request phrased as 'the third clip', so it sees positions."""
    text = summarise(_timeline())
    assert "shot 1" in text and "shot 3" in text
    assert "s1" not in text, "ids would be an extra mapping to get wrong"


def test_summary_distinguishes_clips_from_photos() -> None:
    text = summarise(_timeline())
    assert "clip" in text and "photo" in text


def test_summary_shows_existing_captions_and_music_state() -> None:
    text = summarise(_timeline())
    assert 'caption "day one"' in text
    assert "no music" in text


def test_summary_stays_short() -> None:
    assert len(summarise(_timeline())) < 500


# -------------------------------------------------------------------- translation


def test_remove_a_shot_by_position() -> None:
    (op,) = _ops(EditCommand(action="remove", shot=2))
    assert isinstance(op, RemoveScene)
    assert op.scene_id == "s2"


def test_retime_a_shot() -> None:
    (op,) = _ops(EditCommand(action="retime", shot=1, value=4.0))
    assert isinstance(op, SetSceneDuration)
    assert (op.scene_id, op.seconds) == ("s1", 4.0)


def test_nudge_a_clip_earlier() -> None:
    (op,) = _ops(EditCommand(action="nudge", shot=1, value=-0.8))
    assert isinstance(op, NudgeClipSpan)
    assert op.seconds == -0.8


def test_slow_a_clip_down() -> None:
    (op,) = _ops(EditCommand(action="speed", shot=1, value=0.5))
    assert isinstance(op, SetClipSpeed)
    assert op.speed == 0.5


def test_caption_a_shot() -> None:
    (op,) = _ops(EditCommand(action="caption", shot=2, text="golden hour"))
    assert isinstance(op, SetSceneCaption)
    assert op.content == "golden hour"


def test_an_empty_caption_clears_it() -> None:
    (op,) = _ops(EditCommand(action="caption", shot=2, text=""))
    assert isinstance(op, SetSceneCaption)
    assert op.content == ""


def test_promote_a_hook() -> None:
    (op,) = _ops(EditCommand(action="hook", shot=3))
    assert isinstance(op, SetSceneRole)
    assert (op.scene_id, op.role) == ("s3", "hook")


def test_mute_and_unmute() -> None:
    (muted,) = _ops(EditCommand(action="mute", shot=1))
    (unmuted,) = _ops(EditCommand(action="unmute", shot=1))
    assert isinstance(muted, SetClipMuted) and muted.mute is True
    assert isinstance(unmuted, SetClipMuted) and unmuted.mute is False


def test_pace_the_whole_reel() -> None:
    (op,) = _ops(EditCommand(action="pace", value=0.8))
    assert isinstance(op, ScaleAllDurations)
    assert op.factor == 0.8


def test_drop_the_music() -> None:
    (op,) = _ops(EditCommand(action="music_off"))
    assert isinstance(op, RemoveMusic)


def test_change_music_level() -> None:
    (op,) = _ops(EditCommand(action="music_level", value=-6.0))
    assert isinstance(op, SetMusicGain)
    assert op.gain_db == -6.0


def test_reorder_by_position() -> None:
    (op,) = _ops(EditCommand(action="reorder", order=(3, 1, 2)))
    assert isinstance(op, ReorderScenes)
    assert op.scene_ids == ("s3", "s1", "s2")


# ------------------------------------------------------------- forgiving, not lossy


def test_a_shot_that_does_not_exist_is_skipped() -> None:
    """One misread instruction must not cost the user the others."""
    ops = _ops(
        EditCommand(action="remove", shot=9),
        EditCommand(action="retime", shot=1, value=3.0),
    )
    assert len(ops) == 1
    assert isinstance(ops[0], SetSceneDuration)


def test_a_clip_only_action_on_a_photo_is_skipped() -> None:
    assert _ops(EditCommand(action="speed", shot=2, value=0.5)) == ()


def test_a_partial_reorder_is_refused() -> None:
    """Applying it would silently delete a shot."""
    assert _ops(EditCommand(action="reorder", order=(1, 2))) == ()


def test_a_reorder_naming_a_missing_shot_is_refused() -> None:
    assert _ops(EditCommand(action="reorder", order=(1, 2, 7))) == ()


def test_a_zero_duration_retime_is_skipped() -> None:
    assert _ops(EditCommand(action="retime", shot=1, value=0.0)) == ()


def test_an_empty_request_yields_no_ops() -> None:
    assert to_ops(EditRequest(commands=()), _timeline()) == ()


def test_several_commands_translate_together() -> None:
    ops = _ops(
        EditCommand(action="remove", shot=3),
        EditCommand(action="caption", shot=2, text="hi"),
        EditCommand(action="pace", value=0.9),
    )
    assert len(ops) == 3


# --------------------------------------------------------------- the round trip


def test_translated_ops_actually_apply() -> None:
    """The translation is only useful if what comes out is applicable."""
    before = _timeline()
    ops = to_ops(_request(EditCommand(action="remove", shot=2)), before)
    after = apply(before, ops)
    assert [s.id for s in after.scenes] == ["s1", "s3"]
    assert after.version == before.version + 1


def test_captioning_a_shot_with_no_overlay_adds_one() -> None:
    before = _timeline()
    ops = to_ops(_request(EditCommand(action="caption", shot=1, text="opening")), before)
    after = apply(before, ops)
    assert after.scenes[0].overlays[0].content == "opening"  # type: ignore[union-attr]


def test_clearing_a_caption_removes_the_overlay() -> None:
    before = _timeline()
    ops = to_ops(_request(EditCommand(action="caption", shot=2, text="")), before)
    assert apply(before, ops).scenes[1].overlays == ()


class _FakeLLM:
    def __init__(self, reply: EditRequest) -> None:
        self._reply = reply

    def structured(self, prompt: str, schema: type, **kw: object) -> EditRequest:
        return self._reply

    def complete(self, prompt: str, **kw: object) -> str:
        return ""

    def unload(self) -> None:
        return None


def test_edit_returns_a_new_timeline_and_a_diff() -> None:
    llm = _FakeLLM(_request(EditCommand(action="retime", shot=2, value=3.5)))
    after, lines = edit("make the second one longer", _timeline(), llm)  # type: ignore[arg-type]
    assert after.scene("s2").duration == 3.5  # type: ignore[union-attr]
    assert lines == ("shot 2: hold for 3.5s",)


def test_an_unusable_request_says_so_rather_than_doing_nothing() -> None:
    """A silent no-op version would look like the tool ignored the user."""
    llm = _FakeLLM(_request(EditCommand(action="remove", shot=99)))
    with pytest.raises(GenvaiError, match="naming a shot number"):
        edit("delete the last bit", _timeline(), llm)  # type: ignore[arg-type]


def test_music_state_survives_an_unrelated_edit() -> None:
    started = _timeline().model_copy(update={"music": Music(state="resolved", asset_id="m")})
    ops = to_ops(_request(EditCommand(action="retime", shot=1, value=3.0)), started)
    assert apply(started, ops).music.asset_id == "m"
