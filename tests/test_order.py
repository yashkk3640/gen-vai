"""Reconciling what the model returns.

A 3B model produces schema-valid-but-wrong answers often enough that its output is
treated as a suggestion, not a instruction. These tests pin the repair rules, which is
where the safety actually lives - no model required to run them.
"""

import pytest

from genvai.media import ClipQuality, MediaItem, Span
from genvai.pipeline.select import (
    CAPTION_LIMIT,
    ReelPlan,
    ShotChoice,
    build_timeline,
    describe,
    reconcile,
)
from genvai.timeline import Asset, AssetProvenance, TextOverlay


def _item(asset_id: str, *, kind: str = "image", at: str | None = None) -> MediaItem:
    return MediaItem(
        asset_id=asset_id,
        kind=kind,  # type: ignore[arg-type]
        source_name=f"{asset_id}.jpg",
        width=1920,
        height=1080,
        duration=8.0 if kind == "video" else None,
        captured_at=at,
        quality=ClipQuality(sharpness=0.8, exposure=0.5, motion=0.2, shake=0.1),
        spans=(Span(start=1.0, end=3.0, score=0.8, reason="sharp, steady"),)
        if kind == "video"
        else (),
    )


def _plan(*pairs: tuple[str, str], hook: str = "") -> ReelPlan:
    return ReelPlan(
        order=tuple(ShotChoice(asset_id=a, caption=c) for a, c in pairs),
        hook_asset_id=hook,
    )


def _assets(items: tuple[MediaItem, ...]) -> dict:
    return {
        i.asset_id: Asset(
            kind="image",
            path=f"assets/{i.asset_id}",
            sha256=i.asset_id,
            provenance=AssetProvenance(provider="test"),
        )
        for i in items
    }


# ------------------------------------------------------------------- reconcile


def test_honours_the_models_order() -> None:
    items = (_item("aaa"), _item("bbb"), _item("ccc"))
    ordered, _, _ = reconcile(_plan(("ccc", ""), ("aaa", ""), ("bbb", "")), items)
    assert [i.asset_id for i in ordered] == ["ccc", "aaa", "bbb"]


def test_invented_ids_are_dropped() -> None:
    """A hallucinated id would fail the render outright; a smaller reel is recoverable."""
    items = (_item("aaa"), _item("bbb"))
    ordered, _, _ = reconcile(_plan(("aaa", ""), ("nonsense", ""), ("bbb", "")), items)
    assert [i.asset_id for i in ordered] == ["aaa", "bbb"]


def test_forgotten_shots_are_appended_not_lost() -> None:
    items = (_item("aaa", at="2026-01-01T09:00"), _item("bbb", at="2026-01-01T08:00"))
    ordered, _, _ = reconcile(_plan(("aaa", "")), items)
    assert [i.asset_id for i in ordered] == ["aaa", "bbb"]


def test_forgotten_shots_come_back_in_capture_order() -> None:
    items = (
        _item("late", at="2026-01-01T18:00"),
        _item("early", at="2026-01-01T06:00"),
        _item("kept"),
    )
    ordered, _, _ = reconcile(_plan(("kept", "")), items)
    assert [i.asset_id for i in ordered] == ["kept", "early", "late"]


def test_repeated_ids_are_used_once() -> None:
    items = (_item("aaa"), _item("bbb"))
    ordered, _, _ = reconcile(_plan(("aaa", ""), ("aaa", ""), ("bbb", "")), items)
    assert [i.asset_id for i in ordered] == ["aaa", "bbb"]


def test_truncated_ids_are_matched() -> None:
    """The prompt shows a shortened id to save context, so the model echoes it back."""
    items = (_item("abcdef123456789"), _item("999888777666555"))
    ordered, _, _ = reconcile(_plan(("abcdef12", "")), items)
    assert ordered[0].asset_id == "abcdef123456789"


def test_an_ambiguous_prefix_is_refused() -> None:
    """Two candidates and no way to choose - better to append both than guess wrong."""
    items = (_item("abc111"), _item("abc222"))
    ordered, _, _ = reconcile(_plan(("abc", "")), items)
    assert len(ordered) == 2


def test_captions_are_collected() -> None:
    items = (_item("aaa"), _item("bbb"))
    _, captions, _ = reconcile(_plan(("aaa", "Day one"), ("bbb", "")), items)
    assert captions == {"aaa": "Day one"}


def test_overlong_captions_are_trimmed() -> None:
    items = (_item("aaa"),)
    _, captions, _ = reconcile(_plan(("aaa", "x" * 200)), items)
    assert len(captions["aaa"]) == CAPTION_LIMIT


def test_blank_captions_are_dropped() -> None:
    items = (_item("aaa"),)
    _, captions, _ = reconcile(_plan(("aaa", "   ")), items)
    assert captions == {}


def test_hook_is_carried_through() -> None:
    items = (_item("aaa"), _item("bbb"))
    _, _, hook = reconcile(_plan(("aaa", ""), ("bbb", ""), hook="bbb"), items)
    assert hook == "bbb"


def test_an_invalid_hook_is_discarded() -> None:
    items = (_item("aaa"),)
    _, _, hook = reconcile(_plan(("aaa", ""), hook="does-not-exist"), items)
    assert hook is None


def test_an_empty_plan_still_yields_every_shot() -> None:
    """The worst case the model can produce must still make a reel."""
    items = (_item("aaa"), _item("bbb"))
    ordered, captions, hook = reconcile(_plan(), items)
    assert len(ordered) == 2
    assert captions == {}
    assert hook is None


# --------------------------------------------------------------------- prompt


def test_description_names_every_shot() -> None:
    items = (_item("aaa"), _item("bbb", kind="video"))
    text = describe(items)
    assert "aaa" in text and "bbb" in text
    assert "photo" in text and "clip" in text


def test_description_includes_the_span_reason() -> None:
    """The model cannot see pixels, so the measurement is what it reasons over."""
    assert "sharp, steady" in describe((_item("v", kind="video"),))


def test_description_stays_compact() -> None:
    """Twenty shots must still fit comfortably in a small model's attention."""
    items = tuple(_item(f"id{i:03d}") for i in range(20))
    assert len(describe(items)) < 2000


# ------------------------------------------------------------------ into scenes


def test_captions_become_overlays() -> None:
    items = (_item("aaa"), _item("bbb"))
    timeline = build_timeline(items, _assets(items), intent="t", captions={"aaa": "Day one"})
    assert timeline.scenes[0].overlays == (TextOverlay(content="Day one"),)
    assert timeline.scenes[1].overlays == ()


def test_the_nominated_hook_gets_the_role() -> None:
    items = (_item("aaa"), _item("bbb"), _item("ccc"))
    timeline = build_timeline(items, _assets(items), intent="t", hook_asset_id="bbb")
    roles = {s.visual.asset_id: s.role for s in timeline.scenes}  # type: ignore[union-attr]
    assert roles["bbb"] == "hook"
    assert sum(1 for r in roles.values() if r == "hook") == 1, "exactly one hook"


def test_without_a_nominated_hook_the_first_shot_opens() -> None:
    items = (_item("aaa"), _item("bbb"))
    timeline = build_timeline(items, _assets(items), intent="t")
    assert timeline.scenes[0].role == "hook"


def test_durations_come_from_measurement_not_the_model() -> None:
    """The model is never asked for a number that was already measured."""
    items = (_item("v", kind="video"),)
    timeline = build_timeline(items, _assets(items), intent="t", captions={"v": "hi"})
    assert timeline.scenes[0].duration == pytest.approx(2.0), "the span's length"
