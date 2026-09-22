"""Assembly: an ordered set of shots becomes a renderable timeline.

No model involved. This is the fallback that keeps the tool working with Ollama down,
and the structure the LLM step later rearranges rather than replaces.
"""

from pathlib import Path

import pytest

from genvai.adapters.fs_store import FilesystemStore
from genvai.errors import GenvaiError
from genvai.media import ClipQuality, MediaItem, MediaLibrary, Span
from genvai.pipeline.select import build_timeline, fit_to_duration, make_reel
from genvai.timeline import AssetProvenance, AssetVisual, Canvas, ClipVisual, KenBurns


def _quality(sharpness: float = 0.8) -> ClipQuality:
    return ClipQuality(sharpness=sharpness, exposure=0.5, motion=0.3, shake=0.1)


def _clip(name: str, *, start: float = 3.0, end: float = 5.0, at: str | None = None) -> MediaItem:
    return MediaItem(
        asset_id=name,
        kind="video",
        source_name=f"{name}.mp4",
        width=1920,
        height=1080,
        duration=10.0,
        captured_at=at,
        quality=_quality(),
        spans=(Span(start=start, end=end, score=0.85, reason="sharp, steady"),),
    )


def _photo(name: str, *, at: str | None = None, sharpness: float = 0.8) -> MediaItem:
    return MediaItem(
        asset_id=name,
        kind="image",
        source_name=f"{name}.jpg",
        width=4032,
        height=3024,
        captured_at=at,
        quality=_quality(sharpness),
    )


def _assets(items: tuple[MediaItem, ...]) -> dict:
    from genvai.timeline import Asset

    return {
        i.asset_id: Asset(
            kind="video" if i.kind == "video" else "image",
            path=f"assets/{i.asset_id}",
            sha256=i.asset_id,
            provenance=AssetProvenance(provider="test"),
        )
        for i in items
    }


# ------------------------------------------------------------------------- fitting


def test_trims_to_the_target() -> None:
    items = tuple(_photo(f"p{i}") for i in range(20))
    kept = fit_to_duration(items, 8.0)
    assert sum(2.2 for _ in kept) <= 8.0 + 2.2


def test_keeps_the_strongest_shots() -> None:
    items = (_photo("weak", sharpness=0.05), _photo("strong", sharpness=0.95))
    assert [i.asset_id for i in fit_to_duration(items, 2.2)] == ["strong"]


def test_preserves_capture_order_after_trimming() -> None:
    items = (
        _photo("c", at="2026-01-03T00:00:00"),
        _photo("a", at="2026-01-01T00:00:00"),
        _photo("b", at="2026-01-02T00:00:00"),
    )
    assert [i.asset_id for i in fit_to_duration(items, 10.0)] == ["a", "b", "c"]


def test_zero_target_yields_nothing() -> None:
    assert fit_to_duration((_photo("a"),), 0.0) == ()


def test_a_short_library_is_never_padded() -> None:
    """Running short is fine; inventing material is not."""
    kept = fit_to_duration((_photo("a"),), 60.0)
    assert len(kept) == 1


# ------------------------------------------------------------------------ building


def test_clips_carry_their_span_as_the_trim() -> None:
    items = (_clip("v", start=6.2, end=8.6),)
    timeline = build_timeline(items, _assets(items), intent="t")
    visual = timeline.scenes[0].visual
    assert isinstance(visual, ClipVisual)
    assert visual.source_start == pytest.approx(6.2)
    assert visual.source_end == pytest.approx(8.6)
    assert timeline.scenes[0].duration == pytest.approx(2.4)


def test_photos_get_camera_movement() -> None:
    items = (_photo("p"),)
    timeline = build_timeline(items, _assets(items), intent="t")
    assert isinstance(timeline.scenes[0].visual, AssetVisual)
    assert isinstance(timeline.scenes[0].motion, KenBurns)


def test_camera_moves_vary_between_photos() -> None:
    """One uniform push on every photo is what makes a reel read as a slideshow."""
    items = tuple(_photo(f"p{i}") for i in range(4))
    timeline = build_timeline(items, _assets(items), intent="t")
    moves = {(s.motion.start_rect, s.motion.end_rect) for s in timeline.scenes}  # type: ignore[union-attr]
    assert len(moves) > 1


def test_first_shot_is_the_hook_and_last_is_the_payoff() -> None:
    items = (_photo("a"), _photo("b"), _photo("c"))
    timeline = build_timeline(items, _assets(items), intent="t")
    assert [s.role for s in timeline.scenes] == ["hook", "body", "payoff"]
    assert timeline.hook is not None


def test_span_reason_is_carried_through_for_the_user() -> None:
    items = (_clip("v"),)
    timeline = build_timeline(items, _assets(items), intent="t")
    assert timeline.scenes[0].note == "sharp, steady"


def test_canvas_and_seed_are_honoured() -> None:
    items = (_photo("p"),)
    timeline = build_timeline(
        items, _assets(items), intent="t", canvas=Canvas(width=1920, height=1080), seed=42
    )
    assert timeline.canvas.width == 1920
    assert timeline.seed == 42


def test_scene_ids_are_sequential() -> None:
    items = tuple(_photo(f"p{i}") for i in range(3))
    timeline = build_timeline(items, _assets(items), intent="t")
    assert [s.id for s in timeline.scenes] == ["s1", "s2", "s3"]


def test_timeline_roundtrips() -> None:
    from genvai.timeline import Timeline

    items = (_clip("v"), _photo("p"))
    timeline = build_timeline(items, _assets(items), intent="t")
    assert Timeline.model_validate_json(timeline.model_dump_json()) == timeline


# --------------------------------------------------------------------- make_reel


@pytest.fixture
def stocked(tmp_path: Path, sample_media: dict[str, Path]) -> FilesystemStore:
    """A project with real assets stored and a matching media library."""
    store = FilesystemStore(tmp_path / "projects")
    store.create("trip", "trip")
    items = []
    for key, kind in (("photo", "image"), ("clip_a", "video"), ("clip_b", "video")):
        asset = store.store_asset("trip", sample_media[key], kind, AssetProvenance(provider="test"))
        items.append(_clip(asset.sha256) if kind == "video" else _photo(asset.sha256))
    store.save_media("trip", MediaLibrary(items=tuple(items)))
    return store


def test_make_reel_saves_a_timeline(stocked: FilesystemStore) -> None:
    timeline = make_reel("trip", stocked, intent="a trip", target_duration=8.0)
    assert timeline.scenes
    assert stocked.load_timeline("trip") == timeline


def test_make_reel_versions_each_attempt(stocked: FilesystemStore) -> None:
    """Every reel is reversible, so a second attempt never overwrites the first."""
    first = make_reel("trip", stocked, target_duration=6.0)
    second = make_reel("trip", stocked, target_duration=4.0, seed=9)
    assert second.version == first.version + 1
    assert stocked.load_timeline("trip", first.version) == first


def test_make_reel_references_real_assets(stocked: FilesystemStore) -> None:
    timeline = make_reel("trip", stocked, target_duration=8.0)
    for scene in timeline.scenes:
        asset_id = scene.visual.asset_id  # type: ignore[union-attr]
        assert asset_id in timeline.assets
        assert stocked.asset_path("trip", asset_id).exists()


def test_make_reel_respects_the_target(stocked: FilesystemStore) -> None:
    timeline = make_reel("trip", stocked, target_duration=4.0)
    assert timeline.duration <= 8.0


def test_make_reel_without_media_says_what_to_do(tmp_path: Path) -> None:
    store = FilesystemStore(tmp_path / "projects")
    store.create("empty", "empty")
    with pytest.raises(GenvaiError, match="genvai add"):
        make_reel("empty", store)


def test_make_reel_is_deterministic_for_a_seed(stocked: FilesystemStore) -> None:
    first = make_reel("trip", stocked, target_duration=8.0, seed=7)
    second = make_reel("trip", stocked, target_duration=8.0, seed=7)
    assert [s.model_dump() for s in first.scenes] == [s.model_dump() for s in second.scenes]
