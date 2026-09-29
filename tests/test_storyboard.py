"""The storyboard layer: finding a poster's pictures, drafting a story, the storybook,
the copy filter, and drawing frames.

Synthetic posters with pictures in known places, so "did it find the makeup photo" has
an answer rather than an opinion.
"""

from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from genvai.adapters.compositor import Compositor
from genvai.cast import Picture, find_cast
from genvai.media import Face
from genvai.ocr import Box
from genvai.palette import type_palette
from genvai.pipeline.story import _clean_line, _stems
from genvai.promo import Brief, Offer
from genvai.stories import ARC_NAMES, Copy, arc_for, draft
from genvai.storyboard import Caption, Shot, Storyboard
from genvai.storybook import page


def _poster() -> np.ndarray:
    """Paper, two photo tiles on the right with labels beside them, a hero at top."""
    rng = np.random.default_rng(5)
    page_ = np.full((1536, 1024, 3), 250, dtype=np.uint8)
    page_[500:640, 820:960] = rng.integers(30, 220, (140, 140, 3), dtype=np.uint8)
    page_[800:940, 820:960] = rng.integers(30, 220, (140, 140, 3), dtype=np.uint8)
    page_[40:460, 680:1000] = rng.integers(60, 240, (420, 320, 3), dtype=np.uint8)
    return page_


def _label(text: str, x: float, y: float) -> Box:
    return Box(text=text, confidence=0.99, x=x, y=y, width=120, height=26)


LABELS = (_label("MAKEUP", 680, 560), _label("NAILS", 680, 860))


# --------------------------------------------------------------------------- cast


def test_each_picture_is_found_as_its_own_object() -> None:
    cast = find_cast(_poster(), boxes=LABELS)
    assert len([p for p in cast if p.kind != "ornament"]) == 3


def test_a_tile_is_named_by_the_word_beside_it() -> None:
    names = {p.name for p in find_cast(_poster(), boxes=LABELS)}
    assert {"makeup", "nails"} <= names


def test_a_sentence_beside_a_tile_is_not_its_name() -> None:
    boxes = (_label("with beautiful self care", 680, 560),)
    assert all(p.name == "" for p in find_cast(_poster(), boxes=boxes))


def test_a_face_inside_a_tile_makes_it_a_portrait() -> None:
    face = Face(rect=(0.83, 0.34, 0.05, 0.04), confidence=0.9)
    cast = find_cast(_poster(), boxes=LABELS, faces=(face,))
    makeup = next(p for p in cast if p.name == "makeup")
    assert makeup.kind == "portrait" and makeup.face is not None
    assert cast[0].id == "p1" and cast[0].kind == "portrait", "people rank first"


def test_a_face_on_bare_paper_gets_a_portrait_of_its_own() -> None:
    """Line art on paper - a logo - has no colour for the grid to see."""
    face = Face(rect=(0.1, 0.1, 0.08, 0.05), confidence=0.9)
    cast = find_cast(_poster(), faces=(face,))
    assert any(p.face == face.rect for p in cast)


def test_a_banner_along_a_pictures_edge_is_trimmed_off() -> None:
    banner = (680 / 1024, 440 / 1536, 320 / 1024, 18 / 1536)
    untrimmed = find_cast(_poster())
    trimmed = find_cast(_poster(), stubborn=(banner,))
    hero = max(untrimmed, key=lambda p: p.pixels[0] * p.pixels[1])
    cut = max(trimmed, key=lambda p: p.pixels[0] * p.pixels[1])
    assert cut.rect[1] + cut.rect[3] < hero.rect[1] + hero.rect[3]


# ------------------------------------------------------------------------ palette


def test_the_palette_is_read_from_the_type() -> None:
    """Photographs of hair and skin must not outvote the colour the headings are set in."""
    image = np.full((400, 400, 3), 245, dtype=np.uint8)
    image[:, :200] = (150, 90, 50)  # a big brown photograph
    image[250:300, 220:380] = (200, 10, 75)  # magenta type
    colours = type_palette(image, ((220 / 400, 250 / 400, 160 / 400, 50 / 400),))
    assert colours.accent[0] > 150 and colours.accent[1] < 60


# ------------------------------------------------------------------------ stories


def _cast() -> tuple[Picture, ...]:
    face = Face(rect=(0.83, 0.34, 0.05, 0.04), confidence=0.9)
    logo = Face(rect=(0.1, 0.1, 0.08, 0.05), confidence=0.9)
    return find_cast(_poster(), boxes=LABELS, faces=(face, logo))


def _brief() -> Brief:
    return Brief(
        business="Glow Glam",
        occasion="Navratri offer",
        phone="7043641428",
        offers=(
            Offer(service="EYEBROW", price="₹40", note="Uparlips Free"),
            Offer(service="MAKEUP", price="₹900"),
            Offer(service="NAILS", price="₹300"),
            Offer(service="BLEACH", price="₹250"),
        ),
    )


def _board(arc: str, **kw) -> Storyboard:
    return draft(
        _brief(),
        _cast(),
        ("assets/images/poster.png",),
        arc=arc_for(arc, 0),
        palette=("#781327", "#FDF7F4", "#C2084E"),
        seed=1,
        **kw,
    )


@pytest.mark.parametrize("arc", ARC_NAMES)
def test_every_arc_opens_on_a_hook_and_closes_on_the_number(arc: str) -> None:
    board = _board(arc)
    assert board.shots[0].role == "hook"
    assert board.shots[-1].role == "cta"
    assert any("70436 41428" in c.text for c in board.shots[-1].captions)


@pytest.mark.parametrize("arc", ARC_NAMES)
def test_every_arc_names_the_cheapest_price(arc: str) -> None:
    text = " ".join(c.text for s in _board(arc).shots for c in s.captions)
    assert "₹40" in text


@pytest.mark.parametrize("arc", ARC_NAMES)
def test_every_shot_points_at_a_real_picture(arc: str) -> None:
    board = _board(arc)
    for shot in board.shots:
        if shot.picture is not None:
            assert board.picture(shot.picture) is not None


def test_arcs_differ_in_shape_and_tempo() -> None:
    shapes = {tuple(s.role for s in _board(a).shots) for a in ARC_NAMES}
    tempos = {_board(a).bpm for a in ARC_NAMES}
    assert len(shapes) == len(ARC_NAMES)
    assert len(tempos) == len(ARC_NAMES)


def test_a_service_is_shown_with_the_picture_named_like_it() -> None:
    board = _board("glow-up")
    makeup = next(s for s in board.shots if any(c.text == "MAKEUP" for c in s.captions))
    assert board.picture(makeup.picture).name == "makeup"


def test_the_logo_signs_off_and_does_not_open() -> None:
    board = _board("glow-up")
    logo = next(p for p in board.cast if p.kind == "portrait" and not p.name)
    assert board.shots[-1].picture == logo.id
    assert board.shots[0].picture != logo.id


def test_the_countdown_ends_on_the_cheapest() -> None:
    board = _board("countdown")
    reveal = next(s for s in board.shots if s.role == "reveal")
    assert reveal.captions[0].text.startswith("#1")
    assert any(c.text == "₹40" for c in reveal.captions)


def test_copy_overrides_only_what_it_fills() -> None:
    board = _board("glow-up", copy=Copy(hook="SHINE ON"))
    texts = [c.text for s in board.shots for c in s.captions]
    assert "SHINE ON" in texts
    assert "BOOK YOUR SLOT" in texts, "the arc's own cta stays"


def test_a_storyboard_is_measured_in_beats() -> None:
    board = _board("glow-up")
    beats = sum(s.beats for s in board.shots)
    assert board.duration == pytest.approx(beats * 60 / board.bpm)
    assert board.model_copy(update={"bpm": board.bpm * 2}).duration == pytest.approx(
        board.duration / 2
    )


def test_a_storyboard_roundtrips() -> None:
    board = _board("reveal")
    assert Storyboard.model_validate_json(board.model_dump_json()) == board


def test_drafting_is_deterministic() -> None:
    assert _board("treat") == _board("treat")


def test_an_unknown_arc_is_refused() -> None:
    with pytest.raises(ValueError, match="glow-up"):
        arc_for("bogus", 0)


# --------------------------------------------------------------------------- copy


BANNED = _stems("Navratri offer Glow Glam with Gracy")


@pytest.mark.parametrize(
    "line",
    [
        "NAVRAATRI IS HERE!",  # repeats the festival, misspelt
        "BOOK NOW BEFORE SOLD OUT",  # invents scarcity
        "BOOK NOW BEFORE IT'S GON",  # cut off at the cap
        "SAVE 50 TODAY",  # a number is an invented price
        "HURRY IN",
    ],
)
def test_bad_copy_is_refused(line: str) -> None:
    assert _clean_line(line, BANNED) == ""


def test_good_copy_is_kept() -> None:
    assert _clean_line("ready to shine?", BANNED) == "READY TO SHINE?"


# ---------------------------------------------------------------------- storybook


def test_the_storybook_has_a_card_per_shot() -> None:
    board = _board("glow-up")
    html = page(board, {s.id: f"frames/{s.id}.jpg" for s in board.shots})
    assert html.count('class="shot"') == len(board.shots)
    assert "frames/s1.jpg" in html
    assert board.logline.split()[0] in html


def test_the_storybook_escapes_what_it_shows() -> None:
    board = _board("glow-up")
    shot = board.shots[0].model_copy(update={"captions": (Caption(text="<script>x</script>"),)})
    html = page(board.model_copy(update={"shots": (shot,)}), {})
    assert "<script>" not in html and "&lt;script&gt;" in html


# ------------------------------------------------------------------------ drawing


@pytest.fixture
def small() -> Compositor:
    return Compositor(Path("ffmpeg-not-needed"), width=180, height=320)


@pytest.mark.parametrize("layout", ["full", "card", "poster", "title"])
def test_every_layout_draws_a_frame_of_the_right_size(small: Compositor, layout: str) -> None:
    board = _board("glow-up")
    picture = next(p for p in board.cast if p.name == "makeup")
    shot = Shot(
        id="s1",
        role="offer",
        beats=2,
        layout=layout,  # type: ignore[arg-type]
        picture=picture.id,
        move="pull_out" if layout == "poster" else "push_in",
        captions=(
            Caption(text="EYEBROW", role="kicker", entrance="words"),
            Caption(text="₹40", role="price", entrance="pop", at=0.5),
            Caption(text="types on", role="footer", entrance="type", at=1.0),
        ),
        cut="whip",
        effects=("petals", "bokeh", "light_leak", "grain", "vignette"),
    )
    poster = Image.fromarray(_poster())
    stage = small.stage(board, shot, (poster,), (poster,))
    for t in (0.0, 0.1, 0.5, board.seconds(shot) - 0.05):
        frame = small.frame(stage, t, cut_in="whip", cut_out="whip")
        assert frame.size == (180, 320)


def test_a_flash_starts_white(small: Compositor) -> None:
    board = _board("glow-up")
    shot = next(s for s in board.shots if s.cut == "flash")
    stage = small.stage(board, shot, (Image.fromarray(_poster()),), ())
    first = np.asarray(small.frame(stage, 0.0, cut_in="flash", cut_out=None))
    later = np.asarray(small.frame(stage, 0.5, cut_in="flash", cut_out=None))
    assert first.mean() > 200 > later.mean()


def test_a_shot_encodes(tmp_path: Path, ffmpeg_binary: Path) -> None:
    compositor = Compositor(ffmpeg_binary, width=180, height=320)
    board = _board("glow-up")
    stage = compositor.stage(board, board.shots[0], (Image.fromarray(_poster()),), ())
    out = compositor.render_shot(stage, tmp_path / "s1.mp4", frames=6, cut_in="cut", cut_out=None)
    assert out.stat().st_size > 0


# ---------------------------------------------------------------------- legibility


def _one_caption_board(
    background: tuple[int, int, int], role: str = "kicker"
) -> tuple[Storyboard, Image.Image]:
    """A board whose only picture is a flat colour, filling the frame."""
    board = _board("glow-up")
    picture = next(p for p in board.cast if p.name == "makeup")
    shot = Shot(
        id="s1",
        role="hook",
        beats=4,
        layout="full",
        picture=picture.id,
        move="static",
        captions=(Caption(text="READY TO GLOW", role=role, entrance="pop"),),
    )
    return board.model_copy(update={"shots": (shot,)}), Image.new("RGB", (1024, 1536), background)


@pytest.mark.parametrize(
    "background", [(255, 255, 255), (250, 230, 235), (120, 20, 40), (20, 20, 20)]
)
@pytest.mark.parametrize("role", ["kicker", "headline", "footer"])
def test_every_caption_is_readable_whatever_is_behind_it(
    small: Compositor, background: tuple[int, int, int], role: str
) -> None:
    board, plate = _one_caption_board(background, role)
    stage = small.stage(board, board.shots[0], (plate,), (plate,))
    ((_, _, ratio, _),) = small.legibility(stage)
    assert ratio >= 4.5


def test_light_text_over_a_dark_picture_is_left_alone(small: Compositor) -> None:
    board, plate = _one_caption_board((40, 10, 20))
    stage = small.stage(board, board.shots[0], (plate,), (plate,))
    assert small.legibility(stage)[0][3] == "as designed"


def test_over_pale_paper_the_text_turns_dark(small: Compositor) -> None:
    """On a pale scene dark ink reads better than any box."""
    board, plate = _one_caption_board((252, 248, 246))
    stage = small.stage(board, board.shots[0], (plate,), (plate,))
    assert small.legibility(stage)[0][3] == "dark ink"


def test_over_a_busy_picture_the_text_gets_a_plate(small: Compositor) -> None:
    """Neither light nor dark text reads over something both light and dark."""
    board, _ = _one_caption_board((0, 0, 0))
    stripes = np.zeros((1536, 1024, 3), dtype=np.uint8)
    stripes[:, ::8] = 255
    stripes[:, 1::8] = 255
    stripes[:, 2::8] = 255
    plate = Image.fromarray(stripes)
    stage = small.stage(board, board.shots[0], (plate,), (plate,))
    assert small.legibility(stage)[0][3] == "plate"


def test_the_storybook_shows_each_captions_contrast() -> None:
    board = _board("glow-up")
    first = board.shots[0]
    checks = {first.id: tuple((c.text, c.role, 6.2, "plate") for c in first.captions)}
    html = page(board, {}, checks)
    assert "6.2:1" in html and "plate" in html


# ------------------------------------------------------------------- reading time


@pytest.mark.parametrize("arc", ARC_NAMES)
def test_every_caption_stays_long_enough_to_read(arc: str) -> None:
    from genvai.storyboard import reading_seconds

    board = _board(arc)
    for index, shot in enumerate(board.shots):
        for caption in shot.captions:
            assert board.dwell(index, caption) >= reading_seconds(caption) - 1e-9, (
                shot.id,
                caption.text,
            )


def test_pacing_lengthens_in_half_beats() -> None:
    board = _board("glow-up")
    assert all((shot.beats * 2) == int(shot.beats * 2) for shot in board.shots)


def test_text_never_arrives_during_the_cut_into_a_shot() -> None:
    from genvai.storyboard import TRANSITION, appears

    assert appears(Caption(text="X", at=0.0), "whip", 0.5) == TRANSITION
    assert appears(Caption(text="X", at=0.0), "cut", 0.5) == 0.0
    assert appears(Caption(text="X", at=2.0), "whip", 0.5) == 1.0


def test_a_whip_does_not_wrap_the_far_edge_round() -> None:
    from genvai.adapters.compositor import _shifted

    row = np.arange(10, dtype=np.float32).reshape(1, 10, 1)
    assert list(_shifted(row, 3)[0, :, 0]) == [0, 0, 0, 0, 1, 2, 3, 4, 5, 6]
    assert list(_shifted(row, -3)[0, :, 0]) == [3, 4, 5, 6, 7, 8, 9, 9, 9, 9]


def test_text_has_left_before_a_whip_takes_the_picture(small: Compositor) -> None:
    board, plate = _one_caption_board((40, 10, 20))
    stage = small.stage(board, board.shots[0], (plate,), (plate,))
    end = board.seconds(board.shots[0]) - 0.001
    bare = replace(stage, sprites=())
    with_text = np.asarray(small.frame(stage, end, cut_in="cut", cut_out="whip"), dtype=float)
    without = np.asarray(small.frame(bare, end, cut_in="cut", cut_out="whip"), dtype=float)
    assert np.abs(with_text - without).mean() < 1.0


# ------------------------------------------------------------------ service icons


def _price_list_with_icons() -> tuple[np.ndarray, tuple[Box, ...]]:
    """Paper; a ringed icon, a bullet, then 'WAX' and its price; a tagline with a
    drawing beside it that is not a service."""
    page_ = np.full((600, 800, 3), 248, dtype=np.uint8)
    from PIL import ImageDraw

    image = Image.fromarray(page_)
    draw = ImageDraw.Draw(image)
    draw.ellipse([60, 100, 150, 190], outline=(200, 150, 40), width=3)  # gold ring
    draw.rectangle([90, 135, 120, 160], outline=(120, 20, 40), width=3)  # the bowl
    draw.ellipse([180, 140, 186, 146], fill=(120, 20, 40))  # the list's bullet
    draw.ellipse([60, 380, 150, 470], fill=(220, 30, 90))  # beside a tagline
    boxes = (
        Box(text="WAX", confidence=0.99, x=200, y=130, width=80, height=30),
        Box(text="250", confidence=0.99, x=600, y=130, width=60, height=30),
        Box(text="BEAUTY CARE", confidence=0.99, x=200, y=410, width=200, height=30),
    )
    return np.asarray(image), boxes


def test_the_icon_beside_a_service_is_found_and_named_after_it() -> None:
    from genvai.cast import find_icons

    image, boxes = _price_list_with_icons()
    icons = find_icons(image, boxes, ("WAX · Cream",))
    assert [i.name for i in icons] == ["wax"]
    x, y, w, h = icons[0].rect
    assert x * 800 < 70 and (x + w) * 800 > 140, "the whole ring, not the bullet"


def test_only_services_have_icons() -> None:
    from genvai.cast import find_icons

    image, boxes = _price_list_with_icons()
    assert find_icons(image, boxes, ()) == ()


def test_a_neighbouring_card_with_words_in_it_is_not_an_icon() -> None:
    """In an add-on grid the drawing left of a label is the previous option's card."""
    from genvai.cast import find_icons

    image, boxes = _price_list_with_icons()
    inside = Box(text="1 FINGER", confidence=0.99, x=80, y=140, width=50, height=12)
    assert find_icons(image, (*boxes, inside), ("WAX",)) == ()


def _offer_cast() -> tuple[Picture, ...]:
    def pic(id_: str, kind: str, name: str, source: str = "poster") -> Picture:
        return Picture(
            id=id_,
            rect=(0.1, 0.1, 0.2, 0.2),
            kind=kind,  # type: ignore[arg-type]
            name=name,
            source=source,  # type: ignore[arg-type]
            pixels=(900, 1600) if source == "client" else (130, 130),
        )

    return (
        pic("hero", "illustration", ""),
        pic("i1", "icon", "gel polish hand"),
        pic("i2", "icon", "gel polish hand toe"),
        pic("i3", "icon", "wax"),
        pic("photo1", "photo", "gel polish", "client"),
    )


def _casting():
    import random

    from genvai.stories import _Casting

    return _Casting(_offer_cast(), random.Random(0))


def test_a_clients_photo_beats_the_posters_icon() -> None:
    assert _casting().for_offer(Offer(service="GEL POLISH HAND", price="₹200")).id == "photo1"


def test_the_closest_named_icon_wins() -> None:
    cast = tuple(p for p in _offer_cast() if p.source == "poster")
    import random

    from genvai.stories import _Casting

    picked = _Casting(cast, random.Random(0)).for_offer(Offer(service="GEL POLISH HAND", price="1"))
    assert picked.id == "i1", "the bottle, not the foot"


def test_a_list_item_takes_its_sections_icon() -> None:
    assert _casting().for_offer(Offer(service="WAX · Cream", price="₹350")).id == "i3"


def test_a_service_with_nothing_of_its_own_gets_no_picture() -> None:
    """A pot and a diya under a service told the viewer nothing."""
    assert _casting().for_offer(Offer(service="BLEACH", price="₹250")) is None


def test_a_clients_photo_never_opens_the_reel() -> None:
    assert _casting().hero().id == "hero"
