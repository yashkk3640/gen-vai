"""A storyboard: the reel designed as a short film before any of it is rendered.

Between what a poster says (the brief) and the frames that get encoded sits a decision
the other paths never make explicitly: what the reel is *about*. A storyboard is that
decision written down - a logline, an arc, and one entry per shot saying what is seen,
how it is framed, how the camera moves, what text appears and how it arrives, how the
shot is cut in, and how long it lasts in beats.

Kept as data for the same reason the timeline is: it can be reviewed before anything is
spent on rendering, changed one shot at a time, and versioned. For promo reels it is the
source of truth; the render is a function of it.

Durations are in beats, not seconds. The rhythm is designed first and the tempo applied
after, so the same storyboard cuts on the beat of whatever track is put under it.
"""

from typing import Literal

from pydantic import Field

from genvai.cast import Picture
from genvai.timeline import Frozen

STORYBOARD_SCHEMA_VERSION = 1

FPS = 30
"""Frames per second a storyboard is drawn at."""

TRANSITION = 0.16
"""Seconds a whip, flash or zoom takes; a fade takes twice this."""

READ_BASE = 0.45
READ_PER_WORD = 0.24
"""How long a caption must sit on screen, fully arrived, to be read: a glance to find it
plus a quarter second a word. A caption shown for less was on screen, and unread."""

Role = Literal["hook", "build", "reveal", "offer", "proof", "cta"]
"""What a shot does in the story. Hook stops the thumb; build creates the want; reveal
pays it off; offer names a price; proof is the thing to screenshot; cta says how."""

Layout = Literal["full", "card", "poster", "title"]
"""How the picture sits in frame.

full - the picture fills the frame. For large pictures only; a small tile blown up to
       full screen is visibly soft.
card - the picture floats as a card over a blurred, tinted copy of itself: depth from
       layers, and a small picture shown at a size it can hold.
poster - the whole poster, to be read and saved.
title - no picture: brand colour, light and text.
"""

Framing = Literal["wide", "medium", "close", "detail"]
Angle = Literal["eye_level", "top_down", "low", "high", "profile"]
"""Descriptive: poster imagery has the angle it was drawn with. Recorded so the story
reads as a shot list, and so a future filmed version knows what to shoot."""

Move = Literal["static", "push_in", "pull_out", "pan_left", "pan_right", "rise", "drift"]
Entrance = Literal["pop", "slide_up", "type", "words", "fade"]
Cut = Literal["cut", "whip", "flash", "zoom", "fade"]
Effect = Literal["grain", "vignette", "light_leak", "petals", "bokeh", "shine"]
CaptionRole = Literal["kicker", "headline", "price", "footer"]


class Caption(Frozen):
    """A line of text in a shot, and how it arrives."""

    text: str
    role: CaptionRole = "headline"
    entrance: Entrance = "pop"
    at: float = Field(default=0.0, ge=0.0, description="Beats into the shot it appears.")


class Shot(Frozen):
    id: str
    role: Role
    beats: float = Field(gt=0.0)
    layout: Layout = "card"
    picture: str | None = Field(default=None, description="A cast id, or None.")
    subject: str = Field(default="", description="What is seen: who, doing what, posed how.")
    framing: Framing = "medium"
    angle: Angle = "eye_level"
    move: Move = "push_in"
    captions: tuple[Caption, ...] = ()
    cut: Cut = "cut"
    effects: tuple[Effect, ...] = ()
    note: str = Field(default="", description="Why this shot is here. For the storybook.")


class Storyboard(Frozen):
    schema_version: int = STORYBOARD_SCHEMA_VERSION
    version: int = 1
    title: str
    logline: str = Field(description="The reel in one sentence.")
    arc: str = Field(description="Which story template it was drawn from.")
    bpm: float = Field(default=110.0, gt=30.0, lt=240.0)
    seed: int = 0
    posters: tuple[str, ...] = Field(description="Poster asset paths, project-relative.")
    cast: tuple[Picture, ...] = ()
    palette: tuple[str, str, str] = Field(
        default=("#3A0F24", "#FFE9D6", "#E5A93B"), description="Deep, light, accent."
    )
    shots: tuple[Shot, ...] = ()

    @property
    def beat(self) -> float:
        return 60.0 / self.bpm

    @property
    def duration(self) -> float:
        return sum(shot.beats for shot in self.shots) * self.beat

    def seconds(self, shot: Shot) -> float:
        return shot.beats * self.beat

    def picture(self, picture_id: str | None) -> Picture | None:
        return next((p for p in self.cast if p.id == picture_id), None)

    def dwell(self, index: int, caption: Caption) -> float:
        """Seconds a caption sits on screen fully arrived and not yet leaving."""
        shot = self.shots[index]
        following = self.shots[index + 1].cut if index + 1 < len(self.shots) else None
        leaving = transition_seconds(following) if following in ("whip", "fade") else 0.0
        return (
            self.seconds(shot)
            - appears(caption, shot.cut if index else "cut", self.beat)
            - entrance_seconds(caption)
            - leaving
        )

    def paced(self) -> "Storyboard":
        """Every shot lengthened, in half beats, until each of its captions can be read.

        Cutting on the beat is kept - half a beat is still on the grid - but a caption
        that is gone before it can be read is worse than a shot half a beat long.
        """
        shots = list(self.shots)
        for index, shot in enumerate(shots):
            board = self.model_copy(update={"shots": tuple(shots)})
            short = max(
                (reading_seconds(c) - board.dwell(index, c) for c in shot.captions),
                default=0.0,
            )
            if short > 0:
                extra = -(-short // (self.beat / 2)) * 0.5
                shots[index] = shot.model_copy(update={"beats": shot.beats + extra})
        return self.model_copy(update={"shots": tuple(shots)})


def transition_seconds(cut: str | None) -> float:
    if cut in (None, "cut"):
        return 0.0
    return TRANSITION * (2 if cut == "fade" else 1)


def appears(caption: Caption, cut_in: str, beat: float) -> float:
    """When a caption starts to arrive: on its beat, but never during the cut into the
    shot - text smeared by a whip or washed out by a flash is not read."""
    return max(caption.at * beat, transition_seconds(cut_in))


def entrance_seconds(caption: Caption) -> float:
    if caption.entrance == "type":
        return 0.045 * len(caption.text)
    if caption.entrance == "words":
        return 0.22 * len(caption.text.split())
    return {"pop": 0.3, "slide_up": 0.35, "fade": 0.5}.get(caption.entrance, 0.3)


def reading_seconds(caption: Caption) -> float:
    return READ_BASE + READ_PER_WORD * len(caption.text.split())
