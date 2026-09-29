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

from collections.abc import Callable
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

from genvai.palette import Brand, brand
from genvai.ports import FaceDetector
from genvai.promo import Brief, Offer
from genvai.regions import photo_regions
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
CTA_SECONDS = 3.4
DISSOLVE = 0.22

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
    store_asset: Callable[[Path, AssetProvenance], Asset],
    faces: FaceDetector | None = None,
) -> Timeline:
    """Turn a confirmed brief and its artwork into a renderable timeline.

    `store_asset` is passed in rather than a store, so composition stays unaware of how
    projects are kept and can be exercised against a temporary directory.
    """
    frame = canvas or Canvas()
    work.mkdir(parents=True, exist_ok=True)
    palette = _palette_of(posters)

    backdrops = _backdrops(posters, palette, frame, work, faces)
    chosen = choose(brief.offers, featured)

    scenes: list[Scene] = []
    assets: dict[str, Asset] = {}

    def add(image: Path, note: str) -> str:
        asset = store_asset(image, AssetProvenance(provider="promo", prompt=note))
        assets[asset.sha256] = asset
        return asset.sha256

    # 1. The occasion, over the most picture-like part of the artwork.
    scenes.append(
        _scene(
            "s1",
            HOOK_SECONDS,
            add(backdrops[0], "hook"),
            role="hook",
            index=0,
            label=brief.occasion.upper() or brief.business.upper(),
            headline=brief.tagline or "SPECIAL OFFER",
            footer=brief.business if brief.occasion else "",
            first=True,
        )
    )

    # 2. One beat per featured offer, cycling the remaining backdrops.
    for position, offer in enumerate(chosen):
        # Cycle from the second onward, wrapping. Modulo the whole list, not the list
        # minus one: artwork with a single usable region produced an index past the end.
        backdrop = backdrops[(1 + position) % len(backdrops)]
        scenes.append(
            _scene(
                f"s{len(scenes) + 1}",
                OFFER_SECONDS,
                add(backdrop, f"{offer.service} {offer.price}"),
                role="body",
                index=len(scenes),
                label=offer.service.upper(),
                headline=offer.price,
                footer=offer.note,
            )
        )

    # 3. Each poster whole, as the moment to screenshot.
    for poster in posters:
        card = _poster_card(poster, palette, frame, work)
        scenes.append(
            _scene(
                f"s{len(scenes) + 1}",
                CARD_SECONDS,
                add(card, f"menu {poster.stem}"),
                role="body",
                index=len(scenes),
                label="FULL MENU",
                headline="",
                footer="SAVE THIS",
                still=True,
            )
        )

    # 4. How to book.
    if brief.phone:
        scenes.append(
            _scene(
                f"s{len(scenes) + 1}",
                CTA_SECONDS,
                add(backdrops[-1], "cta"),
                role="cta",
                index=len(scenes),
                label="",
                headline=_spaced(brief.phone),
                footer=f"{brief.business} - call or DM to book".strip(" -"),
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
        # Silent as well as full: the reach comes from attaching a trending sound in the
        # app, and a baked-in track forfeits it.
        export=Export(audio_variants=("silent", "full"), snap_cuts_to_beat=False),
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
) -> list[Path]:
    """One scrimmed backdrop per picture-like region found across the artwork.

    With a face detector, regions holding a face rank first - the backdrops the client
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
        regions = photo_regions(array, count=REGIONS, aspect=canvas.aspect, faces=found)
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
    first: bool = False,
    still: bool = False,
) -> Scene:
    start, end = _STILL if still else _MOVES[index % len(_MOVES)]
    overlays = []
    if label:
        overlays.append(TextOverlay(content=label, position="top_center", style_ref="label"))
    if headline:
        overlays.append(TextOverlay(content=headline, position="center", style_ref="headline"))
    if footer:
        overlays.append(TextOverlay(content=footer, position="bottom_center", style_ref="footer"))

    return Scene(
        id=scene_id,
        duration=seconds,
        role=role,  # type: ignore[arg-type]
        visual=AssetVisual(asset_id=asset_id, fit="cover"),
        motion=KenBurns(start_rect=start, end_rect=end),
        overlays=tuple(overlays),
        transition_in=Transition(kind="cut")
        if first
        else Transition(kind="dissolve", duration=DISSOLVE),
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
