"""Tests for the ingest/analysis layer. No files, no OpenCV, no GPU - pure data."""

import pytest

from genvai.media import ClipQuality, MediaItem, MediaLibrary, Span


def _quality(sharpness: float = 0.8, **kw: float) -> ClipQuality:
    base = {"sharpness": sharpness, "exposure": 0.5, "motion": 0.3, "shake": 0.2}
    return ClipQuality(**{**base, **kw})  # type: ignore[arg-type]


def _clip(asset_id: str, *, group: str | None = None, sharpness: float = 0.8) -> MediaItem:
    return MediaItem(
        asset_id=asset_id,
        kind="video",
        source_name=f"{asset_id}.mov",
        width=1920,
        height=1080,
        duration=14.0,
        dedup_group=group,
        quality=_quality(sharpness),
    )


def _photo(asset_id: str, width: int = 3024, height: int = 4032) -> MediaItem:
    return MediaItem(
        asset_id=asset_id,
        kind="image",
        source_name=f"{asset_id}.jpg",
        width=width,
        height=height,
    )


def test_blurry_clip_is_not_usable() -> None:
    assert not _quality(sharpness=0.1).usable


def test_blown_out_clip_is_not_usable() -> None:
    assert not _quality(exposure=0.95).usable


def test_shaky_clip_is_not_usable() -> None:
    assert not _quality(shake=0.9).usable


def test_good_clip_is_usable() -> None:
    assert _quality().usable


def test_best_span_wins_on_score() -> None:
    item = _clip("a").model_copy(
        update={
            "spans": (
                Span(start=0.0, end=2.0, score=0.4),
                Span(start=6.0, end=8.5, score=0.9, reason="steady, subject centred"),
            )
        }
    )
    assert item.best_span is not None
    assert item.best_span.start == pytest.approx(6.0)
    assert item.best_span.duration == pytest.approx(2.5)


def test_still_has_no_best_span() -> None:
    assert _photo("p").best_span is None


def test_dedup_keeps_the_sharpest_take() -> None:
    library = MediaLibrary(
        items=(
            _clip("a", group="g1", sharpness=0.4),
            _clip("b", group="g1", sharpness=0.9),
            _clip("c"),
        )
    )
    assert {i.asset_id for i in library.deduplicated()} == {"b", "c"}


def test_dedup_discards_nothing_from_the_library() -> None:
    """Losing takes stay available - the user can always ask for the other one."""
    library = MediaLibrary(items=(_clip("a", group="g1"), _clip("b", group="g1")))
    assert len(library.items) == 2
    assert len(library.deduplicated()) == 1


def test_unanalysed_reports_what_still_needs_work() -> None:
    library = MediaLibrary(items=(_clip("a"), _photo("z")))
    assert tuple(i.asset_id for i in library.unanalysed) == ("z",)


def test_clips_and_photos_partition_the_library() -> None:
    library = MediaLibrary(items=(_clip("a"), _photo("p")))
    assert len(library.clips) == 1
    assert len(library.photos) == 1


def test_lookup_by_asset_id() -> None:
    library = MediaLibrary(items=(_clip("a"),))
    assert library.item("a") is not None
    assert library.item("nope") is None


def test_portrait_detection() -> None:
    assert _photo("p", width=1080, height=1920).is_vertical
    assert not _clip("a").is_vertical


def test_library_roundtrips_through_json() -> None:
    library = MediaLibrary(items=(_clip("a", group="g1"), _photo("p")))
    assert MediaLibrary.model_validate_json(library.model_dump_json()) == library
