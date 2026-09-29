"""Story templates: proven arcs, filled from a brief and a poster's cast.

Each arc is a function from what is known - the offers, the pictures, the occasion - to a
storyboard. The arc carries the craft: where the hook lands, when the price is revealed,
how the pace rises and falls, which cuts sit where. The brief and the cast supply the
content. Copy comes from `Copy`, which has sensible defaults per arc and which a small
model may rewrite line by line - never the structure, which it is measured to be bad at.

Pure: no I/O, deterministic for a given seed.
"""

import random
import re
from collections.abc import Callable

from pydantic import Field

from genvai.cast import Picture
from genvai.promo import Brief, Offer, choose
from genvai.storyboard import Caption, Cut, Effect, Move, Shot, Storyboard
from genvai.timeline import Frozen

COPY_LIMIT = 24
"""Characters a line of copy may run to. Beyond this it wraps at headline size."""


class Copy(Frozen):
    """The words a story needs that are not on the poster. Short by construction."""

    hook: str = Field(default="", max_length=COPY_LIMIT)
    promise: str = Field(default="", max_length=COPY_LIMIT)
    cta: str = Field(default="", max_length=COPY_LIMIT)


class Arc(Frozen):
    name: str
    blurb: str
    bpm: float
    copy_defaults: Copy


ARCS: tuple[Arc, ...] = (
    Arc(
        name="glow-up",
        blurb="festive energy: the look, the glow, then the prices - fast and bright",
        bpm=112.0,
        copy_defaults=Copy(
            hook="READY TO GLOW?", promise="YOUR GLOW, SORTED", cta="BOOK YOUR SLOT"
        ),
    ),
    Arc(
        name="countdown",
        blurb="the top three picks counted down, the best price revealed last",
        bpm=120.0,
        copy_defaults=Copy(
            hook="TOP 3 PICKS", promise="SAVE THE BEST FOR LAST", cta="BOOK BEFORE IT ENDS"
        ),
    ),
    Arc(
        name="treat",
        blurb="slow and soft: self-care as a gift to yourself, then the offer",
        bpm=92.0,
        copy_defaults=Copy(hook="YOU DESERVE THIS", promise="TIME FOR YOU", cta="TREAT YOURSELF"),
    ),
    Arc(
        name="reveal",
        blurb="starts tight on one face and pulls out to reveal the whole menu",
        bpm=104.0,
        copy_defaults=Copy(hook="WAIT FOR IT", promise="EVERYTHING, ONE MENU", cta="CALL TO BOOK"),
    ),
)

ARC_NAMES = tuple(a.name for a in ARCS)


def arc_for(name: str | None, seed: int) -> Arc:
    if name is None:
        return ARCS[seed % len(ARCS)]
    for arc in ARCS:
        if arc.name == name:
            return arc
    raise ValueError(f"No story arc '{name}'. Choose from: {', '.join(ARC_NAMES)}")


def draft(
    brief: Brief,
    cast: tuple[Picture, ...],
    posters: tuple[str, ...],
    *,
    arc: Arc,
    palette: tuple[str, str, str],
    seed: int = 0,
    copy: Copy | None = None,
    bpm: float | None = None,
) -> Storyboard:
    """A storyboard for this brief, drawn from `arc`.

    `copy` overrides the arc's default lines where it has them; an empty field keeps the
    default. `bpm` overrides the arc's tempo - pass the music's when there is a track.
    """
    words = _merged(arc.copy_defaults, copy)
    cast_ = _Casting(cast, random.Random(seed))
    shots = _ARC_SHOTS[arc.name](brief, cast_, words)
    return Storyboard(
        title=f"{_festival(brief)} - {brief.business}".strip(" -") or "Offer reel",
        logline=_LOGLINES[arc.name].format(
            business=brief.business or "the salon", festival=_festival(brief).title()
        ),
        arc=arc.name,
        bpm=bpm or arc.bpm,
        seed=seed,
        posters=posters,
        cast=cast,
        palette=palette,
        shots=tuple(s.model_copy(update={"id": f"s{i + 1}"}) for i, s in enumerate(shots)),
    ).paced()


# ------------------------------------------------------------------------ casting


class _Casting:
    """Chooses pictures for shots, so the same one is not used twice running."""

    def __init__(self, cast: tuple[Picture, ...], rng: random.Random) -> None:
        self._cast = [p for p in cast if p.kind != "ornament"]
        self._rng = rng
        self._last: str | None = None
        self._shown: dict[str, int] = {}

    def hero(self) -> Picture | None:
        """The big picture a poster leads with - set near the top, and large."""
        # The logo portrait is the brand's sign-off, not the opening image.
        big = [p for p in self._cast if p.pixels[0] * p.pixels[1] >= 240 * 240 and not _is_logo(p)]
        return self._take(min(big, key=lambda p: p.rect[1]) if big else self._first())

    def people(self, count: int) -> list[Picture]:
        """Pictures of people, best first; photographs after them."""
        ranked = [p for p in self._cast if p.kind == "portrait" and p.name] + [
            p for p in self._cast if p.kind == "photo" and p.name
        ]
        ranked += [p for p in self._cast if p not in ranked and p.pixels[0] < 240]
        chosen = [p for p in ranked if p.id != self._last][:count]
        if chosen:
            self._last = chosen[-1].id
        return chosen

    def for_offer(self, offer: Offer) -> Picture | None:
        """The picture named like the service, else the next that was not just shown."""
        wanted = _tokens(offer.service)
        named = [p for p in self._cast if p.name and wanted & _tokens(p.name)]
        if named:
            return self._take(named[0])
        # Otherwise a person, the least recently shown - never the logo mid-story.
        people = [p for p in self._cast if p.kind in ("portrait", "photo") and not _is_logo(p)]
        pool = [p for p in people if p.id != self._last] or [
            p for p in self._cast if p.id != self._last and not _is_logo(p)
        ]
        if not pool:
            return self._take(None)
        pick = min(pool, key=lambda p: (self._shown.get(p.id, 0), self._rng.random()))
        return self._take(pick)

    def brand(self) -> Picture | None:
        """The logo portrait - a face on bare paper - or else the hero."""
        logos = [p for p in self._cast if _is_logo(p)]
        return self._take(logos[0]) if logos else self.hero()

    def _first(self) -> Picture | None:
        return self._cast[0] if self._cast else None

    def _take(self, picture: Picture | None) -> Picture | None:
        if picture is not None:
            self._last = picture.id
            self._shown[picture.id] = self._shown.get(picture.id, 0) + 1
        return picture


# -------------------------------------------------------------------------- arcs


_LOGLINES = {
    "glow-up": "{festival} is here and {business} has the look - glow to prices in one breath.",
    "countdown": "{business}'s three best {festival} picks, counted down to the one not to miss.",
    "treat": "A slow, warm invitation to take time for yourself this {festival}, at {business}.",
    "reveal": "One face, then the whole menu: {business}'s {festival} offer, revealed.",
}

_LIVELY: tuple[Effect, ...] = ("petals", "light_leak", "grain", "vignette")
_CLEAN: tuple[Effect, ...] = ("grain", "vignette")
_SOFT: tuple[Effect, ...] = ("bokeh", "light_leak", "grain", "vignette")


def _glow_up(brief: Brief, cast: _Casting, words: Copy) -> list[Shot]:
    festival = _festival(brief)
    offers = choose(brief.offers, 3)
    hero = cast.hero()
    shots = [
        _shot(
            "hook",
            4,
            hero,
            layout="full",
            framing="wide",
            move="push_in",
            captions=(
                Caption(text=f"{festival} IS HERE", role="kicker", entrance="words"),
                Caption(text=words.hook, role="headline", entrance="pop", at=1.0),
            ),
            effects=_LIVELY,
            note="Open on the poster's hero art, full frame, with the festival named at once.",
        )
    ]
    moves: tuple[Move, ...] = ("push_in", "pan_left", "pan_right")
    for i, picture in enumerate(cast.people(3)):
        shots.append(
            _shot(
                "build",
                2,
                picture,
                framing="close",
                move=moves[i % 3],
                cut="whip",
                captions=(Caption(text=picture.name.upper(), role="kicker", entrance="slide_up"),),
                effects=("shine", "grain") if i == 0 else _CLEAN,
                note="Quick close-ups on the beat: what the salon does, before any price.",
            )
        )
    if offers:
        shots.append(_from_price(offers[0], festival, cut="flash"))
    shots += _offer_shots(offers, cast, cut="zoom")
    return [*shots, _proof(), _cta(brief, cast, words)]


def _countdown(brief: Brief, cast: _Casting, words: Copy) -> list[Shot]:
    festival = _festival(brief)
    picks = choose(brief.offers, 3)
    shots = [
        _shot(
            "hook",
            4,
            cast.hero(),
            layout="full",
            framing="wide",
            move="push_in",
            captions=(
                Caption(text=f"{festival} OFFER", role="kicker", entrance="slide_up"),
                Caption(text=f"TOP {len(picks)} PICKS", role="headline", entrance="pop", at=1.0),
            ),
            effects=_LIVELY,
            note="Promise a countdown in the first second; the rest of the reel pays it off.",
        )
    ]
    ranked = sorted(enumerate(picks, start=1), key=lambda pair: -pair[0])
    for rank, offer in ranked:
        last = rank == 1
        shots.append(
            _shot(
                "reveal" if last else "offer",
                4 if not last else 6,
                cast.for_offer(offer),
                framing="close",
                move="push_in" if last else ("pan_left" if rank % 2 else "pan_right"),
                cut="flash" if last else "whip",
                captions=(
                    # Rank and name share the top line; below the card there is room for
                    # the price and its note, not for a headline as well.
                    Caption(text=f"#{rank} · {offer.service}", role="kicker", entrance="words"),
                    Caption(text=offer.price, role="price", entrance="pop", at=1.5),
                    *_note(offer, at=2.0),
                ),
                effects=("shine", "grain", "vignette") if last else _CLEAN,
                note="The best price comes last, with a flash." if last else "Counting down.",
            )
        )
    return [*shots, _proof(), _cta(brief, cast, words)]


def _treat(brief: Brief, cast: _Casting, words: Copy) -> list[Shot]:
    festival = _festival(brief)
    offers = choose(brief.offers, 3)
    people = cast.people(3)
    lead = people[0] if people else cast.hero()
    shots = [
        _shot(
            "hook",
            6,
            lead,
            framing="close",
            move="drift",
            captions=(
                Caption(text=f"{festival} SELF-CARE", role="kicker", entrance="fade"),
                Caption(text=words.hook, role="headline", entrance="type", at=1.5),
            ),
            effects=_SOFT,
            note="Slow open on a face. The pace says rest; the copy says it is for you.",
        )
    ]
    for picture in people[1:]:
        shots.append(
            _shot(
                "build",
                4,
                picture,
                framing="medium",
                move="drift",
                cut="fade",
                captions=(Caption(text=words.promise, role="kicker", entrance="fade"),),
                effects=_SOFT,
                note="Let each picture breathe; soft light, no hard cuts.",
            )
        )
    shots += _offer_shots(offers, cast, cut="fade", beats=5, effects=_SOFT)
    return [*shots, _proof(cut="fade"), _cta(brief, cast, words, cut="fade")]


def _reveal(brief: Brief, cast: _Casting, words: Copy) -> list[Shot]:
    festival = _festival(brief)
    offers = choose(brief.offers, 3)
    people = cast.people(1)
    lead = people[0] if people else cast.hero()
    shots = [
        _shot(
            "hook",
            4,
            lead,
            framing="detail",
            move="push_in",
            captions=(
                Caption(text=f"{festival} OFFER", role="kicker", entrance="fade"),
                Caption(text=words.hook, role="headline", entrance="type", at=1.0),
            ),
            effects=("light_leak", "grain", "vignette"),
            note="Tight on one face - nothing else in frame, nothing explained yet.",
        ),
        _shot(
            "reveal",
            6,
            lead,
            layout="poster",
            framing="wide",
            move="pull_out",
            cut="zoom",
            captions=(
                Caption(text=words.promise, role="kicker", entrance="slide_up", at=2.0),
                Caption(text="SAVE THIS", role="footer", entrance="pop", at=3.5),
            ),
            effects=_CLEAN,
            note="Pull out from that face to the whole poster - the reveal.",
        ),
    ]
    shots += _offer_shots(offers, cast, cut="whip")
    return [*shots, _cta(brief, cast, words)]


_ARC_SHOTS: dict[str, Callable[[Brief, _Casting, Copy], list[Shot]]] = {
    "glow-up": _glow_up,
    "countdown": _countdown,
    "treat": _treat,
    "reveal": _reveal,
}


# ------------------------------------------------------------------ shared shots


def _from_price(offer: Offer, festival: str, *, cut: Cut) -> Shot:
    return _shot(
        "reveal",
        4,
        None,
        layout="title",
        framing="wide",
        move="push_in",
        cut=cut,
        captions=(
            Caption(text=f"THIS {festival}", role="kicker", entrance="type"),
            Caption(text=f"FROM {offer.price}", role="price", entrance="pop", at=1.5),
        ),
        effects=("bokeh", "light_leak", "grain"),
        note="The payoff: the lowest price, alone, on brand colour.",
    )


def _offer_shots(
    offers: tuple[Offer, ...],
    cast: _Casting,
    *,
    cut: Cut,
    beats: float = 3,
    effects: tuple[Effect, ...] = _CLEAN,
) -> list[Shot]:
    moves: tuple[Move, ...] = ("push_in", "pan_right", "pan_left")
    return [
        _shot(
            "offer",
            beats,
            cast.for_offer(offer),
            framing="close",
            move=moves[i % 3],
            cut=cut,
            captions=(
                Caption(text=offer.service, role="kicker", entrance="slide_up"),
                Caption(text=offer.price, role="price", entrance="pop", at=0.5),
                *_note(offer, at=1.0),
            ),
            effects=effects,
            note="One offer, one picture, the price on the beat.",
        )
        for i, offer in enumerate(offers)
    ]


def _proof(*, cut: Cut = "fade") -> Shot:
    return _shot(
        "proof",
        5,
        None,
        layout="poster",
        framing="wide",
        move="drift",
        cut=cut,
        captions=(
            Caption(text="FULL MENU", role="kicker", entrance="slide_up"),
            Caption(text="SAVE THIS FOR LATER", role="footer", entrance="pop", at=1.0),
        ),
        # No vignette: on a pale board it reads as a grey halo, not as light falling off.
        effects=("grain",),
        note="Hold on the whole poster long enough to screenshot.",
    )


def _cta(brief: Brief, cast: _Casting, words: Copy, *, cut: Cut = "whip") -> Shot:
    phone = _spaced(brief.phone)
    return _shot(
        "cta",
        6,
        cast.brand(),
        framing="medium",
        move="push_in",
        cut=cut,
        captions=(
            Caption(text=words.cta, role="kicker", entrance="slide_up"),
            *((Caption(text=phone, role="price", entrance="pop", at=1.0),) if phone else ()),
            Caption(
                text=f"{brief.business}\ncall or DM to book".strip(),
                role="footer",
                entrance="fade",
                at=2.0,
            ),
        ),
        effects=("bokeh", "grain", "vignette"),
        note="End on the brand and the number, held long enough to dial.",
    )


def _shot(
    role: str,
    beats: float,
    picture: Picture | None,
    *,
    layout: str = "card",
    framing: str = "medium",
    move: Move = "push_in",
    cut: Cut = "cut",
    captions: tuple[Caption, ...] = (),
    effects: tuple[Effect, ...] = (),
    note: str = "",
) -> Shot:
    if picture is None and layout in ("card", "full"):
        layout = "title"
    return Shot(
        id="",
        role=role,  # type: ignore[arg-type]
        beats=beats,
        layout=layout,  # type: ignore[arg-type]
        picture=picture.id if picture else None,
        subject=_subject(picture) if layout != "poster" or picture else "the full poster",
        framing=framing,  # type: ignore[arg-type]
        angle=_angle(picture),
        move=move,
        cut=cut,
        captions=tuple(c for c in captions if c.text.strip()),
        effects=effects,
        note=note,
    )


def _is_logo(picture: Picture) -> bool:
    """A face on bare paper with no label beside it: the brand mark, not a photograph."""
    return picture.kind == "portrait" and not picture.name


def _note(offer: Offer, *, at: float) -> tuple[Caption, ...]:
    return (Caption(text=offer.note, role="footer", entrance="fade", at=at),) if offer.note else ()


def _subject(picture: Picture | None) -> str:
    if picture is None:
        return "brand colour and light, no picture"
    if _is_logo(picture):
        return "the brand's logo portrait"
    what = picture.name or {"photo": "a photograph"}.get(picture.kind, "the poster's illustration")
    if picture.kind == "portrait":
        pose = "in profile" if _angle(picture) == "profile" else "face to camera"
        return f"{what} - a person, {pose}"
    return what


def _angle(picture: Picture | None) -> str:
    if picture is None or picture.face is None:
        return "eye_level"
    # A face set well off its picture's centre is turned: a profile, not a portrait.
    fx = picture.face[0] + picture.face[2] / 2
    px = picture.rect[0] + picture.rect[2] / 2
    return "profile" if abs(fx - px) > picture.rect[2] * 0.18 else "eye_level"


def _festival(brief: Brief) -> str:
    """'Navratri offer' -> 'NAVRATRI'. The occasion's name without the sales word."""
    name = re.sub(r"\b(offer|offers|sale|special|deal)\b", "", brief.occasion, flags=re.I)
    return " ".join(name.split()).upper() or "SPECIAL"


def _tokens(text: str) -> set[str]:
    return {w for w in re.findall(r"[a-z0-9]+", text.lower()) if len(w) > 2}


def _merged(defaults: Copy, override: Copy | None) -> Copy:
    if override is None:
        return defaults
    return Copy(
        hook=override.hook or defaults.hook,
        promise=override.promise or defaults.promise,
        cta=override.cta or defaults.cta,
    )


def _spaced(phone: str) -> str:
    digits = "".join(c for c in phone if c.isdigit())
    return f"{digits[:5]} {digits[5:]}" if len(digits) == 10 else phone
