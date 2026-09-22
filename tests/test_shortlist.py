"""Shortlisting: the pure, deterministic filter that runs before any model."""

import pytest

from genvai.media import ClipQuality, MediaItem, MediaLibrary, Span
from genvai.pipeline.select import contribution, item_score, shortlist


def _quality(sharpness: float = 0.8, exposure: float = 0.5, **kw: float) -> ClipQuality:
    base = {"sharpness": sharpness, "exposure": exposure, "motion": 0.3, "shake": 0.1}
    return ClipQuality(**{**base, **kw})  # type: ignore[arg-type]


def _clip(
    name: str,
    *,
    span_score: float = 0.8,
    sharpness: float = 0.8,
    at: str | None = None,
    group: str | None = None,
    span_duration: float = 2.0,
    **quality: float,
) -> MediaItem:
    return MediaItem(
        asset_id=name,
        kind="video",
        source_name=f"{name}.mp4",
        width=1920,
        height=1080,
        duration=10.0,
        captured_at=at,
        dedup_group=group,
        quality=_quality(sharpness, **quality),
        spans=(Span(start=3.0, end=3.0 + span_duration, score=span_score),),
    )


def _photo(name: str, *, sharpness: float = 0.8, at: str | None = None, **quality: float):
    return MediaItem(
        asset_id=name,
        kind="image",
        source_name=f"{name}.jpg",
        width=4032,
        height=3024,
        captured_at=at,
        quality=_quality(sharpness, **quality),
    )


# -------------------------------------------------------------------------- scoring


def test_a_better_span_scores_higher() -> None:
    assert item_score(_clip("a", span_score=0.9)) > item_score(_clip("b", span_score=0.3))


def test_sharper_photo_scores_higher() -> None:
    assert item_score(_photo("a", sharpness=0.9)) > item_score(_photo("b", sharpness=0.2))


def test_badly_exposed_photo_loses() -> None:
    assert item_score(_photo("a", exposure=0.5)) > item_score(_photo("b", exposure=0.05))


def test_unusable_is_demoted_not_excluded() -> None:
    """Better a mediocre reel than no reel - a bad shot stays eligible, just unlikely."""
    bad = _clip("bad", sharpness=0.05, shake=0.95)
    assert bad.quality is not None and not bad.quality.usable
    assert 0.0 < item_score(bad) < item_score(_clip("good"))


def test_unmeasured_items_score_zero() -> None:
    raw = MediaItem(asset_id="x", kind="image", source_name="x.jpg", width=10, height=10)
    assert item_score(raw) == 0.0


def test_intent_promotes_a_match() -> None:
    plain = _clip("a")
    tagged = plain.model_copy(update={"tags": ("food", "table")})
    assert item_score(tagged, intent="focus on the food") > item_score(plain, intent="food")


def test_intent_without_tags_changes_nothing() -> None:
    """Without a tagger the library has no labels, so intent must not silently reorder."""
    item = _clip("a")
    assert item_score(item, intent="sunset") == item_score(item)


# --------------------------------------------------------------------- contribution


def test_photo_contributes_one_beat() -> None:
    assert contribution(_photo("a")) == pytest.approx(2.2)


def test_clip_contributes_its_span() -> None:
    assert contribution(_clip("a", span_duration=1.8)) == pytest.approx(1.8)


def test_a_very_long_span_is_capped() -> None:
    """No single shot should hold the screen long enough to stop feeling like a reel."""
    assert contribution(_clip("a", span_duration=30.0)) <= 4.0


# ----------------------------------------------------------------------- shortlist


def test_picks_the_best_shots() -> None:
    library = MediaLibrary(
        items=(
            _clip("weak", span_score=0.1, sharpness=0.1),
            _clip("strong", span_score=0.95),
            _clip("middling", span_score=0.5),
        )
    )
    chosen = {i.asset_id for i in shortlist(library, target_duration=2.0)}
    assert "strong" in chosen
    assert "weak" not in chosen


def test_returns_enough_material_for_the_target() -> None:
    library = MediaLibrary(items=tuple(_photo(f"p{i}") for i in range(30)))
    picked = shortlist(library, target_duration=20.0)
    assert sum(contribution(i) for i in picked) >= 20.0


def test_does_not_return_everything() -> None:
    """Handing a model a hundred clips produces worse ordering than handing it twenty."""
    library = MediaLibrary(items=tuple(_photo(f"p{i}") for i in range(60)))
    assert len(shortlist(library, target_duration=15.0)) < 60


def test_a_short_library_is_used_whole() -> None:
    library = MediaLibrary(items=(_photo("a"), _photo("b")))
    assert len(shortlist(library, target_duration=60.0)) == 2


def test_duplicates_are_collapsed() -> None:
    library = MediaLibrary(
        items=(
            _clip("take1", group="g", sharpness=0.4),
            _clip("take2", group="g", sharpness=0.9),
            _clip("other"),
        )
    )
    chosen = {i.asset_id for i in shortlist(library, target_duration=10.0)}
    assert "take1" not in chosen, "the softer take loses"
    assert chosen == {"take2", "other"}


def test_result_is_in_capture_order() -> None:
    """Scores decide who is in; chronology is the sane default for when."""
    library = MediaLibrary(
        items=(
            _photo("late", at="2026-09-22T18:00:00", sharpness=0.95),
            _photo("early", at="2026-09-22T09:00:00", sharpness=0.60),
            _photo("noon", at="2026-09-22T13:00:00", sharpness=0.75),
        )
    )
    order = [i.asset_id for i in shortlist(library, target_duration=10.0)]
    assert order == ["early", "noon", "late"]


def test_undated_items_sort_after_dated_ones() -> None:
    """A dated holiday should not be interleaved with undated screenshots."""
    library = MediaLibrary(
        items=(_photo("screenshot"), _photo("holiday", at="2026-01-01T10:00:00"))
    )
    order = [i.asset_id for i in shortlist(library, target_duration=10.0)]
    assert order == ["holiday", "screenshot"]


def test_max_items_is_respected() -> None:
    library = MediaLibrary(items=tuple(_photo(f"p{i}") for i in range(20)))
    assert len(shortlist(library, target_duration=60.0, max_items=5)) == 5


def test_is_deterministic() -> None:
    library = MediaLibrary(items=tuple(_photo(f"p{i}", sharpness=0.5) for i in range(12)))
    first = shortlist(library, target_duration=8.0)
    second = shortlist(library, target_duration=8.0)
    assert [i.asset_id for i in first] == [i.asset_id for i in second]


def test_empty_library_yields_nothing() -> None:
    assert shortlist(MediaLibrary(), target_duration=30.0) == ()


def test_unanalysed_items_are_skipped() -> None:
    raw = MediaItem(asset_id="x", kind="image", source_name="x.jpg", width=10, height=10)
    library = MediaLibrary(items=(raw, _photo("good")))
    assert [i.asset_id for i in shortlist(library, target_duration=5.0)] == ["good"]


def test_all_bad_material_still_produces_a_shortlist() -> None:
    """Refusing to make a reel because the footage is poor is the wrong failure."""
    library = MediaLibrary(items=tuple(_photo(f"p{i}", sharpness=0.02) for i in range(6)))
    assert shortlist(library, target_duration=10.0) != ()
