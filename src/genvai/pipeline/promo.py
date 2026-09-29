"""Designed artwork -> a promo reel.

A third way in, alongside a camera roll and a bare idea: the content is already on a
poster, and what is missing is only motion, pacing and something legible on a phone.

Panning across a price list produces a reel nobody can read - the type is too small and
there is too much of it. So the poster is mined instead. Its photographs become
backdrops, its prices become the text, and the whole poster appears once as the moment
to screenshot.

Nothing here invents a price. The offers come from OCR and are confirmed by the user
before this is called.
"""

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

from genvai.errors import GenvaiError
from genvai.palette import Brand, brand
from genvai.ports import FaceDetector
from genvai.promo import Brief, Offer
from genvai.regions import erase_text, photo_regions
from genvai.timeline import (
    Asset,
    AssetProvenance,
    AssetVisual,
    Canvas,
    Captions,
    Export,
    KenBurns,
    Rect,
    Scene,
    TextOverlay,
    TextStyle,
    Timeline,
    Transition,
)

FEATURED = 4
"""How many offers get a beat of their own.

Enough to make the reel worth watching, few enough that each price is on screen long
enough to read. The rest are on the poster card.
"""

REGIONS = 4
"""Backdrops mined per poster. One more than a single structure needs, so structures
that use more beats do not repeat a picture as soon."""

OFFER_SECONDS = 2.5
CARD_SECONDS = 2.9
HOOK_SECONDS = 2.8
QUESTION_SECONDS = 2.6
CTA_SECONDS = 3.4
DEADLINE_SECONDS = 2.4
DISSOLVE = 0.22

LIST_ROWS = 3
LIST_SCREENS = 3
LIST_SECONDS = 3.4
"""Three services a screen, each on two lines - the name, then its prices. One line each
does not fit: the Reels buttons take the right 18% of the frame, which leaves 26
characters, and a two-column nail menu runs to 31."""

SCRIM = 0.34
"""How far a backdrop is pulled toward the deep brand colour. White text has to sit on
it, and a poster's own photographs are bright."""

FOOT = 0.62
"""Fraction of the frame, from the bottom, over which the scrim deepens."""

INSET = 0.14
"""How far a detected region is pulled in before it is cropped.

A picture on a poster has its caption right beside it, and a window drawn around the
picture catches the edge of that caption. Shrinking toward the middle trims the
neighbouring type away at the cost of a little of the photograph, which is the right
trade - a stray half-word behind a price reads as a mistake.
"""


def build(
    brief: Brief,
    posters: tuple[Path, ...],
    work: Path,
    *,
    canvas: Canvas | None = None,
    seed: int = 0,
    featured: int = FEATURED,
    style: str | None = None,
    store_asset: Callable[[Path, AssetProvenance], Asset],
    faces: FaceDetector | None = None,
    text: Mapping[Path, tuple[Rect, ...]] | None = None,
) -> Timeline:
    """Turn a confirmed brief and its artwork into a renderable timeline.

    `style` names one of `STRUCTURES`; left out, the seed picks one, so two reels for
    the same client come out as different shapes rather than the same reel twice.

    `text` is where OCR found words on each poster, kept out of the backdrops.

    `store_asset` is passed in rather than a store, so composition stays unaware of how
    projects are kept and can be exercised against a temporary directory.
    """
    frame = canvas or Canvas()
    work.mkdir(parents=True, exist_ok=True)
    palette = _palette_of(posters)
    structure = structure_for(style, seed)

    backdrops = _backdrops(posters, palette, frame, work, faces, text or {})
    cards = tuple(_poster_card(poster, palette, frame, work) for poster in posters)
    chosen = choose(brief.offers, featured)

    assets: dict[str, Asset] = {}

    def add(image: Path, note: str) -> str:
        asset = store_asset(image, AssetProvenance(provider="promo", prompt=note))
        assets[asset.sha256] = asset
        return asset.sha256

    beats: list[_Beat] = []
    for kind in structure.beats:
        beats.extend(_BEATS[kind](brief, chosen, structure, len(beats)))

    scenes: list[Scene] = []
    used = 0
    for beat in beats:
        if beat.picture == "card":
            images = [
                (card, f"menu {poster.stem}") for card, poster in zip(cards, posters, strict=True)
            ]
        elif beat.picture == "first":
            images = [(backdrops[0], beat.note)]
        else:
            images = [(backdrops[used % len(backdrops)], beat.note)]
            used += 1
        for image, note in images:
            index = len(scenes)
            scenes.append(
                _scene(
                    f"s{index + 1}",
                    beat.seconds,
                    add(image, note),
                    role=beat.role,
                    index=index,
                    label=beat.label,
                    headline=beat.headline,
                    footer=beat.footer,
                    headline_style=beat.headline_style,
                    transition=None if index == 0 else structure.transition(),
                    still=beat.picture == "card" or beat.still,
                )
            )

    return Timeline(
        intent=f"{brief.occasion or 'offer'} - {brief.business or 'promo'}".strip(" -"),
        canvas=frame.model_copy(update={"background": _hex(palette.deep)}),
        seed=seed,
        scenes=tuple(scenes),
        assets=assets,
        styles=_styles(palette),
        captions=Captions(enabled=False),
        # Silent only, until a track is attached: a "full" cut with no music in it is
        # the silent cut twice. The silent cut stays either way, because the reach comes
        # from attaching a trending sound in the app. Snapping is a no-op until then.
        export=Export(audio_variants=("silent",), snap_cuts_to_beat=True),
    )


def choose(offers: tuple[Offer, ...], count: int = FEATURED) -> tuple[Offer, ...]:
    """Which offers earn a beat.

    The cheapest leads, because a low number is what stops a thumb. After that the list
    is spread rather than sorted - four prices in ascending order reads as a list, while
    four spaced across the menu reads as a range of what the place does.
    """
    if not offers:
        return ()
    ranked = sorted(offers, key=lambda o: (_number(o.price), o.service))
    if len(ranked) <= count:
        return tuple(ranked)

    cheapest, rest = ranked[0], ranked[1:]
    step = max(1, len(rest) // max(1, count - 1))
    spread = [rest[i] for i in range(0, len(rest), step)][: count - 1]
    return (cheapest, *spread)


# -------------------------------------------------------------------- structures


@dataclass(frozen=True)
class Structure:
    """The shape of a reel: which beats, in what order, cut how.

    Data rather than code paths, so a new shape is a new entry and not a new branch.
    """

    name: str
    blurb: str
    beats: tuple[str, ...]
    offer_seconds: float
    cut: str = "dissolve"
    cut_seconds: float = DISSOLVE

    def transition(self) -> Transition:
        if self.cut == "cut":
            return Transition(kind="cut")
        return Transition(kind=self.cut, duration=self.cut_seconds)  # type: ignore[arg-type]


STRUCTURES: tuple[Structure, ...] = (
    Structure(
        "classic",
        "the occasion, a beat per price, the menu, how to book",
        ("hook", "offers", "menu", "cta"),
        offer_seconds=OFFER_SECONDS,
    ),
    Structure(
        "question",
        "opens on the cheapest price as a question, then fast cuts",
        ("question", "offers", "menu", "cta"),
        offer_seconds=1.8,
        cut="cut",
    ),
    Structure(
        "from",
        "opens on the lowest price, then the whole menu as stacked lists",
        ("from", "list", "deadline", "menu", "cta"),
        offer_seconds=OFFER_SECONDS,
        cut="push",
        cut_seconds=0.3,
    ),
    Structure(
        "menu-first",
        "opens on the full menu to screenshot, then the highlights",
        ("menu", "offers", "cta"),
        offer_seconds=2.3,
        cut="slide",
        cut_seconds=0.3,
    ),
    Structure(
        "countdown",
        "top picks counted down, the best price last",
        ("top", "countdown", "menu", "cta"),
        offer_seconds=2.2,
        cut="wipe",
        cut_seconds=0.25,
    ),
)
"""Trend advice checked in September 2026: the hook has to land in the first one or two
seconds, so three of these put a price in the very first frame. Question hooks and
countdowns are the formats most often cited for small-business offers."""

STYLES = tuple(s.name for s in STRUCTURES)


def structure_for(style: str | None, seed: int) -> Structure:
    """A named structure, or one picked by seed when none is named."""
    if style is None:
        return STRUCTURES[seed % len(STRUCTURES)]
    for structure in STRUCTURES:
        if structure.name == style:
            return structure
    raise GenvaiError(f"No promo style '{style}'. Choose from: {', '.join(STYLES)}")


@dataclass(frozen=True)
class _Beat:
    """One beat before it has a picture. `picture` says which one it gets."""

    role: str
    seconds: float
    label: str
    headline: str
    footer: str
    note: str
    picture: Literal["next", "first", "card"] = "next"
    headline_style: str = "headline"
    still: bool = False


_BeatMaker = Callable[[Brief, tuple[Offer, ...], Structure, int], tuple[_Beat, ...]]


def _hook(
    brief: Brief, chosen: tuple[Offer, ...], structure: Structure, at: int
) -> tuple[_Beat, ...]:
    return (
        _Beat(
            "hook",
            HOOK_SECONDS,
            label=_title(brief),
            headline=brief.tagline or "SPECIAL OFFER",
            footer=brief.business if brief.occasion else "",
            note="hook",
            picture="first",
        ),
    )


def _question(
    brief: Brief, chosen: tuple[Offer, ...], structure: Structure, at: int
) -> tuple[_Beat, ...]:
    if not chosen:
        return _hook(brief, chosen, structure, at)
    lead = chosen[0]
    return (
        _Beat(
            "hook",
            QUESTION_SECONDS,
            label=f"{lead.service} for just",
            headline=f"{lead.price}?",
            footer=brief.occasion or brief.business,
            note=f"question {lead.service}",
            picture="first",
        ),
    )


def _from(
    brief: Brief, chosen: tuple[Offer, ...], structure: Structure, at: int
) -> tuple[_Beat, ...]:
    if not chosen:
        return _hook(brief, chosen, structure, at)
    return (
        _Beat(
            "hook",
            HOOK_SECONDS,
            label=_title(brief),
            headline=f"FROM {chosen[0].price}",
            footer=brief.business if brief.occasion else "",
            note="from",
            picture="first",
        ),
    )


def _top(
    brief: Brief, chosen: tuple[Offer, ...], structure: Structure, at: int
) -> tuple[_Beat, ...]:
    picks = chosen
    if len(picks) < 2:
        return _hook(brief, chosen, structure, at)
    return (
        _Beat(
            "hook",
            HOOK_SECONDS,
            label=_title(brief),
            headline=f"TOP {len(picks)} PICKS",
            footer=brief.business if brief.occasion else "",
            note="top picks",
            picture="first",
        ),
    )


def _offers(
    brief: Brief, chosen: tuple[Offer, ...], structure: Structure, at: int
) -> tuple[_Beat, ...]:
    # A question hook has already shown the cheapest; do not show it twice.
    shown = chosen[1:] if "question" in structure.beats and chosen else chosen
    return tuple(
        _Beat(
            "body",
            structure.offer_seconds,
            label=offer.service,
            headline=offer.price,
            footer=offer.note,
            note=f"{offer.service} {offer.price}",
        )
        for offer in shown
    )


def _countdown(
    brief: Brief, chosen: tuple[Offer, ...], structure: Structure, at: int
) -> tuple[_Beat, ...]:
    picks = chosen
    if len(picks) < 2:
        return _offers(brief, chosen, structure, at)
    # choose() puts the cheapest first; counting down makes it the #1 reveal.
    ranked = sorted(enumerate(picks, start=1), key=lambda pair: -pair[0])
    return tuple(
        _Beat(
            "payoff" if rank == 1 else "body",
            structure.offer_seconds,
            label=f"#{rank}  {offer.service}",
            headline=offer.price,
            footer=offer.note,
            note=f"#{rank} {offer.service} {offer.price}",
        )
        for rank, offer in ranked
    )


def _list(
    brief: Brief, chosen: tuple[Offer, ...], structure: Structure, at: int
) -> tuple[_Beat, ...]:
    rows = _rows(brief.offers)[: LIST_ROWS * LIST_SCREENS]
    screens = [rows[i : i + LIST_ROWS] for i in range(0, len(rows), LIST_ROWS)]
    return tuple(
        _Beat(
            "body",
            LIST_SECONDS,
            label="THE MENU" if index == 0 else "AND MORE",
            headline="\n".join(screen),
            footer="",
            note=f"list {index}",
            headline_style="list",
            still=True,
        )
        for index, screen in enumerate(screens)
    )


def _rows(offers: tuple[Offer, ...]) -> list[str]:
    """One entry per service, its prices side by side beneath it.

    A two-column menu reads as the same name twice at two prices; grouped - "GEL POLISH
    HAND", then "200 · 350" - it reads as what it is, and takes half the screen.
    """
    prices: dict[str, list[str]] = {}
    for offer in offers:
        prices.setdefault(offer.service, []).append(offer.price)
    return [f"{service}\n{' · '.join(amounts)}" for service, amounts in prices.items()]


def _deadline(
    brief: Brief, chosen: tuple[Offer, ...], structure: Structure, at: int
) -> tuple[_Beat, ...]:
    # Without an occasion there is no deadline to state, and inventing one is out.
    if not brief.occasion:
        return ()
    return (
        _Beat(
            "body",
            DEADLINE_SECONDS,
            label=brief.business,
            headline=brief.occasion.upper(),
            footer="book before it ends",
            note="deadline",
            headline_style="statement",
        ),
    )


def _menu(
    brief: Brief, chosen: tuple[Offer, ...], structure: Structure, at: int
) -> tuple[_Beat, ...]:
    opening = at == 0
    return (
        _Beat(
            "hook" if opening else "body",
            CARD_SECONDS + (0.4 if opening else 0.0),
            label=_title(brief) if opening else "FULL MENU",
            headline="",
            footer="SAVE THIS",
            note="menu",
            picture="card",
        ),
    )


def _cta(
    brief: Brief, chosen: tuple[Offer, ...], structure: Structure, at: int
) -> tuple[_Beat, ...]:
    if not brief.phone:
        return ()
    return (
        _Beat(
            "cta",
            CTA_SECONDS,
            label="",
            headline=_spaced(brief.phone),
            footer=f"{brief.business} - call or DM to book".strip(" -"),
            note="cta",
            picture="first",
        ),
    )


_BEATS: dict[str, _BeatMaker] = {
    "hook": _hook,
    "question": _question,
    "from": _from,
    "top": _top,
    "offers": _offers,
    "countdown": _countdown,
    "list": _list,
    "deadline": _deadline,
    "menu": _menu,
    "cta": _cta,
}


def _title(brief: Brief) -> str:
    return (brief.occasion or brief.business).upper()


# ------------------------------------------------------------------- composition


def _palette_of(posters: tuple[Path, ...]) -> Brand:
    with Image.open(posters[0]) as opened:
        return brand(np.asarray(opened.convert("RGB")))


def _backdrops(
    posters: tuple[Path, ...],
    palette: Brand,
    canvas: Canvas,
    work: Path,
    faces: FaceDetector | None = None,
    text: Mapping[Path, tuple[Rect, ...]] | None = None,
) -> list[Path]:
    """One scrimmed backdrop per picture-like region found across the artwork.

    Printed words OCR found are painted out first, so a backdrop does not carry half a
    heading behind the reel's own type. With a face detector, regions holding a face
    rank first - the backdrops the client
    liked in the hand-built reel were the ones with a person in them.

    Always returns something. A poster that is nothing but type yields no regions, and a
    flat brand-coloured card is a better answer than a crash.
    """
    detector = faces if faces is not None and faces.is_available() else None
    made: list[Path] = []
    for poster in posters:
        with Image.open(poster) as opened:
            picture = opened.convert("RGB")
            array = np.asarray(picture)
        found = detector.detect(array) if detector else ()
        clean, stubborn = erase_text(array, (text or {}).get(poster, ()))
        picture = Image.fromarray(clean, "RGB")
        regions = photo_regions(
            clean, count=REGIONS, aspect=canvas.aspect, faces=found, text=stubborn
        )
        for index, rect in enumerate(regions):
            out = work / f"bd-{poster.stem[:16]}-{index}.png"
            _scrimmed(picture, rect, palette, canvas).save(out)
            made.append(out)

    if not made:
        out = work / "bd-plain.png"
        _plain(palette, canvas).save(out)
        made.append(out)
    return made


def _scrimmed(picture: Image.Image, rect: Rect, palette: Brand, canvas: Canvas) -> Image.Image:
    """A region of the artwork, filled to frame and darkened so text reads over it."""
    width, height = picture.size
    x, y, w, h = _inset(rect, INSET)
    crop = picture.crop(
        (int(x * width), int(y * height), int((x + w) * width), int((y + h) * height))
    )
    filled = _cover(crop, canvas.width, canvas.height)

    base = np.asarray(filled, dtype=np.float64)
    ramp = (
        np.clip((np.linspace(0, 1, canvas.height) - (1 - FOOT)) / FOOT, 0, 1)[:, None, None] ** 1.4
    )
    tint = np.array(palette.deep, dtype=np.float64)[None, None, :]
    mixed = base * (1 - SCRIM) + tint * SCRIM
    mixed = mixed * (1 - 0.55 * ramp) + tint * (0.55 * ramp)
    return _ruled(Image.fromarray(np.clip(mixed, 0, 255).astype(np.uint8), "RGB"), palette)


def _poster_card(poster: Path, palette: Brand, canvas: Canvas, work: Path) -> Image.Image | Path:
    """The whole poster on a pale card, with room above and below for text.

    Sized to leave those bands clear: the safe areas would otherwise put the title and
    the "save this" line on top of the artwork.
    """
    out = work / f"card-{poster.stem[:16]}.png"
    ramp = np.linspace(0, 1, canvas.height)[:, None, None]
    light = np.array(palette.light, dtype=np.float64)[None, None, :]
    column = light * (1 - ramp * 0.12)
    card = Image.fromarray(np.tile(column, (1, canvas.width, 1)).astype(np.uint8), "RGB")

    with Image.open(poster) as opened:
        art = opened.convert("RGB")
    scale = min((canvas.width - 250) / art.width, (canvas.height - 780) / art.height)
    art = art.resize((round(art.width * scale), round(art.height * scale)), Image.LANCZOS)

    left = (canvas.width - art.width) // 2
    top = 285
    shadow = Image.new(
        "RGB", (art.width + 26, art.height + 26), _blend(palette.light, palette.deep)
    )
    card.paste(shadow.filter(ImageFilter.GaussianBlur(9)), (left - 13, top - 5))
    card.paste(art, (left, top))
    ImageDraw.Draw(card).rectangle(
        [left - 2, top - 2, left + art.width + 1, top + art.height + 1],
        outline=palette.accent,
        width=3,
    )
    card.save(out)
    return out


def _plain(palette: Brand, canvas: Canvas) -> Image.Image:
    card = Image.new("RGB", (canvas.width, canvas.height), palette.deep)
    return _ruled(card, palette)


def _ruled(image: Image.Image, palette: Brand) -> Image.Image:
    """A hairline inset from the edge. Printed promos have one; a reel from one should."""
    margin = 34
    ImageDraw.Draw(image).rectangle(
        [margin, margin, image.width - margin, image.height - margin],
        outline=palette.accent,
        width=3,
    )
    return image


def _inset(rect: Rect, amount: float) -> Rect:
    """Shrink a rect toward its own centre, staying inside the frame."""
    x, y, w, h = rect
    dx, dy = w * amount / 2, h * amount / 2
    return (
        min(max(0.0, x + dx), 1.0),
        min(max(0.0, y + dy), 1.0),
        max(0.02, w - dx * 2),
        max(0.02, h - dy * 2),
    )


def _cover(image: Image.Image, width: int, height: int) -> Image.Image:
    scale = max(width / image.width, height / image.height)
    resized = image.resize(
        (max(1, round(image.width * scale)), max(1, round(image.height * scale))), Image.LANCZOS
    )
    left = (resized.width - width) // 2
    top = (resized.height - height) // 2
    return resized.crop((left, top, left + width, top + height))


# ------------------------------------------------------------------------- scenes


_MOVES: tuple[tuple[Rect, Rect], ...] = (
    ((0.00, 0.00, 1.00, 1.00), (0.06, 0.05, 0.88, 0.88)),
    ((0.06, 0.05, 0.88, 0.88), (0.00, 0.00, 1.00, 1.00)),
    ((0.00, 0.00, 0.94, 0.94), (0.06, 0.06, 0.94, 0.94)),
)
"""Push, pull, drift. Rotated so consecutive beats do not move identically, which is what
makes a run of stills read as a slideshow."""

_STILL = ((0.02, 0.02, 0.96, 0.96), (0.00, 0.00, 1.00, 1.00))
"""Barely anything, for a poster card - the viewer is trying to read it."""


def _scene(
    scene_id: str,
    seconds: float,
    asset_id: str,
    *,
    role: str,
    index: int,
    label: str,
    headline: str,
    footer: str,
    headline_style: str = "headline",
    transition: Transition | None = None,
    still: bool = False,
) -> Scene:
    start, end = _STILL if still else _MOVES[index % len(_MOVES)]
    overlays = []
    if label:
        overlays.append(TextOverlay(content=label, position="top_center", style_ref="label"))
    if headline:
        overlays.append(TextOverlay(content=headline, position="center", style_ref=headline_style))
    if footer:
        overlays.append(TextOverlay(content=footer, position="bottom_center", style_ref="footer"))

    return Scene(
        id=scene_id,
        duration=seconds,
        role=role,  # type: ignore[arg-type]
        visual=AssetVisual(asset_id=asset_id, fit="cover"),
        motion=KenBurns(start_rect=start, end_rect=end),
        overlays=tuple(overlays),
        transition_in=transition or Transition(kind="cut"),
    )


def _styles(palette: Brand) -> dict[str, TextStyle]:
    """Text in the brand's own colours, outlined in its deep one.

    A black outline would look borrowed; the poster's own dark colour belongs.
    """
    edge = _hex(palette.deep)
    return {
        "label": TextStyle(
            font="seguibl",
            size_pct=3.4,
            color="#FFE9D6",
            stroke_color=edge,
            stroke_px=7,
            max_chars_per_line=22,
            uppercase=True,
        ),
        "headline": TextStyle(
            font="seguibl",
            size_pct=10.5,
            color="#FFFFFF",
            stroke_color=edge,
            stroke_px=12,
            max_chars_per_line=14,
        ),
        "list": TextStyle(
            font="seguibl",
            size_pct=3.6,
            color="#FFFFFF",
            stroke_color=edge,
            stroke_px=7,
            max_chars_per_line=32,
            line_spacing=1.3,
            uppercase=True,
        ),
        "statement": TextStyle(
            font="seguibl",
            size_pct=7.0,
            color="#FFFFFF",
            stroke_color=edge,
            stroke_px=10,
            max_chars_per_line=16,
        ),
        "footer": TextStyle(
            font="segoeuib",
            size_pct=3.6,
            color="#FFF4EF",
            stroke_color=edge,
            stroke_px=7,
            max_chars_per_line=26,
        ),
    }


def _number(price: str) -> float:
    digits = "".join(c for c in price if c.isdigit())
    return float(digits) if digits else 1e9


def _spaced(phone: str) -> str:
    """Group a long number so it fits and can be read aloud."""
    digits = "".join(c for c in phone if c.isdigit())
    return f"{digits[:5]} {digits[5:]}" if len(digits) == 10 else phone


def _hex(colour: tuple[int, int, int]) -> str:
    return "#{:02X}{:02X}{:02X}".format(*colour)


def _blend(a: tuple[int, int, int], b: tuple[int, int, int]) -> tuple[int, int, int]:
    return tuple(int(x * 0.75 + y * 0.25) for x, y in zip(a, b, strict=True))  # type: ignore[return-value]
