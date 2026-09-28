"""Building a promo reel from artwork and a confirmed brief."""

from pathlib import Path

import numpy as np
import pytest
from PIL import Image, ImageDraw

from genvai.pipeline.promo import build, choose
from genvai.promo import Brief, Offer
from genvai.timeline import Asset, AssetProvenance, AssetVisual, Canvas, Timeline


def _offer(service: str, price: str, note: str = "") -> Offer:
    return Offer(service=service, price=price, note=note)


@pytest.fixture
def poster(tmp_path: Path) -> Path:
    """Artwork shaped like a real one: a coloured photo block, a pale price column."""
    image = Image.new("RGB", (1024, 1536), (250, 240, 238))
    draw = ImageDraw.Draw(image)
    rng = np.random.default_rng(4)
    patch = rng.integers(40, 220, (420, 300, 3), dtype=np.uint8)
    image.paste(Image.fromarray(patch, "RGB"), (700, 400))
    for y in range(500, 1200, 90):
        draw.text((120, y), "SERVICE ............ 250", fill=(90, 20, 50))
    draw.rectangle([60, 60, 500, 260], fill=(150, 20, 60))
    path = tmp_path / "poster.png"
    image.save(path)
    return path


@pytest.fixture
def keeper(tmp_path: Path):
    """A store_asset stand-in that records what was stored."""
    kept: list[Path] = []

    def store(image: Path, provenance: AssetProvenance) -> Asset:
        kept.append(image)
        return Asset(
            kind="image",
            path=f"assets/{image.stem}.png",
            sha256=f"sha-{image.stem}",
            provenance=provenance,
        )

    store.kept = kept  # type: ignore[attr-defined]
    return store


def _brief(count: int = 6) -> Brief:
    prices = ["40", "150", "250", "350", "600", "750"]
    return Brief(
        business="Gracy Khatri",
        occasion="Raksha Bandhan offer",
        phone="7043641428",
        offers=tuple(_offer(f"SERVICE {i}", f"₹{p}") for i, p in enumerate(prices[:count])),
    )


# ------------------------------------------------------------------------ choosing


def test_the_cheapest_leads() -> None:
    """A low number is what stops a thumb."""
    chosen = choose(_brief().offers, 4)
    assert chosen[0].price == "₹40"


def test_the_rest_are_spread_not_sorted() -> None:
    """Four ascending prices read as a list; four spread read as a range."""
    chosen = choose(_brief().offers, 4)
    assert len(chosen) == 4
    assert [o.price for o in chosen] != sorted(o.price for o in chosen)[:4]


def test_a_short_list_is_used_whole() -> None:
    assert len(choose(_brief(3).offers, 4)) == 3


def test_no_offers_chooses_nothing() -> None:
    assert choose((), 4) == ()


def test_a_price_that_is_not_a_number_sorts_last_rather_than_crashing() -> None:
    offers = (_offer("A", "free"), _offer("B", "₹10"))
    assert choose(offers, 2)[0].service == "B"


# ------------------------------------------------------------------------ building


def test_a_reel_is_built_from_artwork(poster: Path, keeper, tmp_path: Path) -> None:
    timeline = build(_brief(), (poster,), tmp_path / "work", store_asset=keeper, canvas=Canvas())
    assert isinstance(timeline, Timeline)
    assert timeline.scenes
    assert all(isinstance(s.visual, AssetVisual) for s in timeline.scenes)


def test_it_opens_on_the_occasion_and_closes_on_the_phone(
    poster: Path, keeper, tmp_path: Path
) -> None:
    timeline = build(_brief(), (poster,), tmp_path / "work", store_asset=keeper)
    first, last = timeline.scenes[0], timeline.scenes[-1]
    assert first.role == "hook"
    assert "RAKSHA BANDHAN" in first.overlays[0].content
    assert last.role == "cta"
    assert "70436" in last.overlays[0].content, "grouped so it fits and can be read aloud"


def test_every_featured_price_reaches_the_screen(poster: Path, keeper, tmp_path: Path) -> None:
    brief = _brief()
    timeline = build(brief, (poster,), tmp_path / "work", store_asset=keeper, featured=3)
    shown = {o.content for s in timeline.scenes for o in s.overlays}
    for offer in choose(brief.offers, 3):
        assert offer.price in shown


def test_the_whole_poster_appears_as_the_save_this_moment(
    poster: Path, keeper, tmp_path: Path
) -> None:
    timeline = build(_brief(), (poster,), tmp_path / "work", store_asset=keeper)
    footers = {o.content for s in timeline.scenes for o in s.overlays}
    assert "SAVE THIS" in footers


def test_two_posters_each_get_a_card(poster: Path, keeper, tmp_path: Path) -> None:
    timeline = build(_brief(), (poster, poster), tmp_path / "work", store_asset=keeper)
    cards = [s for s in timeline.scenes if any(o.content == "SAVE THIS" for o in s.overlays)]
    assert len(cards) == 2


def test_without_a_phone_there_is_no_cta(poster: Path, keeper, tmp_path: Path) -> None:
    """An invented number would be worse than no closing card."""
    brief = _brief().model_copy(update={"phone": ""})
    timeline = build(brief, (poster,), tmp_path / "work", store_asset=keeper)
    assert all(s.role != "cta" for s in timeline.scenes)


def test_the_colours_come_from_the_artwork(poster: Path, keeper, tmp_path: Path) -> None:
    """Nobody should have to type a hex code."""
    timeline = build(_brief(), (poster,), tmp_path / "work", store_asset=keeper)
    assert timeline.canvas.background.startswith("#")
    assert timeline.styles["headline"].stroke_color.startswith("#")
    assert timeline.styles["headline"].stroke_color != "#000000"


def test_a_silent_cut_is_always_exported(poster: Path, keeper, tmp_path: Path) -> None:
    """The reach comes from attaching a sound in the app."""
    timeline = build(_brief(), (poster,), tmp_path / "work", store_asset=keeper)
    assert "silent" in timeline.export.audio_variants


def test_consecutive_beats_do_not_move_identically(poster: Path, keeper, tmp_path: Path) -> None:
    timeline = build(_brief(), (poster,), tmp_path / "work", store_asset=keeper)
    moves = [(s.motion.start_rect, s.motion.end_rect) for s in timeline.scenes[:3]]  # type: ignore[union-attr]
    assert len(set(moves)) > 1


def test_artwork_with_no_pictures_still_builds(tmp_path: Path, keeper) -> None:
    """A poster that is nothing but type yields no regions; a flat card beats a crash."""
    plain = tmp_path / "plain.png"
    Image.new("RGB", (1024, 1536), (252, 250, 250)).save(plain)
    timeline = build(_brief(), (plain,), tmp_path / "work", store_asset=keeper)
    assert timeline.scenes


def test_the_timeline_roundtrips(poster: Path, keeper, tmp_path: Path) -> None:
    timeline = build(_brief(), (poster,), tmp_path / "work", store_asset=keeper)
    assert Timeline.model_validate_json(timeline.model_dump_json()) == timeline


def test_building_is_deterministic(poster: Path, keeper, tmp_path: Path) -> None:
    first = build(_brief(), (poster,), tmp_path / "a", store_asset=keeper, seed=7)
    second = build(_brief(), (poster,), tmp_path / "b", store_asset=keeper, seed=7)
    assert [s.model_dump() for s in first.scenes] == [s.model_dump() for s in second.scenes]
