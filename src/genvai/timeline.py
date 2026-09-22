"""The Timeline: the single source of truth for a video.

Every model here is frozen. Edits produce new values via the pure transforms in
`ops.py`; nothing is ever mutated in place.

The prose contract, including field-by-field notes and the music state machine,
lives in docs/03-timeline-schema.md.
"""

from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field

SCHEMA_VERSION = 1

Seconds = Annotated[float, Field(ge=0.0)]
Unit = Annotated[float, Field(ge=0.0, le=1.0)]
"""A fraction of the canvas, so geometry is resolution-independent."""


class Frozen(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


# --------------------------------------------------------------------------- canvas


class Canvas(Frozen):
    width: int = Field(default=1080, ge=16, le=4096, multiple_of=2)
    height: int = Field(default=1920, ge=16, le=4096, multiple_of=2)
    fps: int = Field(default=30, ge=1, le=120)
    background: str = "#000000"

    @property
    def aspect(self) -> float:
        return self.width / self.height


# --------------------------------------------------------------------------- visuals


class GeneratedVisual(Frozen):
    """An image the diffusion model will produce from an LLM-written prompt."""

    kind: Literal["generated"] = "generated"
    prompt: str
    negative_prompt: str | None = None
    seed: int | None = None
    asset_id: str | None = Field(
        default=None, description="None means unresolved; the resolve phase fills it."
    )
    approved: bool = Field(
        default=False,
        description="An approved visual is never regenerated, even if the prompt changes.",
    )


class AssetVisual(Frozen):
    """An image the user supplied."""

    kind: Literal["asset"] = "asset"
    asset_id: str
    fit: Literal["cover", "contain", "blur_pad"] = "cover"


class CardVisual(Frozen):
    """A procedurally drawn typographic card. The fallback that always works."""

    kind: Literal["card"] = "card"
    text: str
    palette: str = "default"


class ColorVisual(Frozen):
    kind: Literal["color"] = "color"
    color: str = "#000000"


Visual = Annotated[
    GeneratedVisual | AssetVisual | CardVisual | ColorVisual,
    Field(discriminator="kind"),
]


# --------------------------------------------------------------------------- motion

Rect = tuple[Unit, Unit, Unit, Unit]
"""(x, y, w, h) in canvas fractions."""

Easing = Literal["linear", "ease_in", "ease_out", "ease_in_out"]


class KenBurns(Frozen):
    """A slow pan/zoom between two crop rectangles. The primary source of motion."""

    kind: Literal["ken_burns"] = "ken_burns"
    start_rect: Rect = (0.0, 0.0, 1.0, 1.0)
    end_rect: Rect = (0.1, 0.1, 0.8, 0.8)
    easing: Easing = "ease_in_out"


class StillMotion(Frozen):
    kind: Literal["still"] = "still"


class ParallaxMotion(Frozen):
    """Depth-layered drift. Needs a depth estimate; falls back to ken_burns without one."""

    kind: Literal["parallax"] = "parallax"
    direction: Literal["left", "right", "up", "down"] = "left"
    strength: float = Field(default=0.15, ge=0.0, le=1.0)


Motion = Annotated[KenBurns | StillMotion | ParallaxMotion, Field(discriminator="kind")]


class Transition(Frozen):
    kind: Literal["cut", "fade", "dissolve", "slide", "push", "wipe"] = "cut"
    duration: Seconds = 0.0


# --------------------------------------------------------------------------- overlays

Position = Literal[
    "top_left",
    "top_center",
    "top_right",
    "middle_left",
    "center",
    "middle_right",
    "bottom_left",
    "bottom_center",
    "bottom_right",
]


class TextOverlay(Frozen):
    kind: Literal["text"] = "text"
    content: str
    position: Position = "bottom_center"
    style_ref: str = "caption"
    start_offset: Seconds = 0.0
    duration: Seconds | None = Field(default=None, description="None means the whole scene.")


class ImageOverlay(Frozen):
    """A watermark or logo."""

    kind: Literal["image"] = "image"
    asset_id: str
    position: Position = "top_right"
    scale: float = Field(default=0.12, gt=0.0, le=1.0)
    opacity: float = Field(default=1.0, ge=0.0, le=1.0)


Overlay = Annotated[TextOverlay | ImageOverlay, Field(discriminator="kind")]


class TextStyle(Frozen):
    font: str = "Inter-SemiBold"
    size_pct: float = Field(default=4.2, gt=0.0, le=30.0, description="Percent of canvas height.")
    color: str = "#FFFFFF"
    stroke_color: str = "#000000"
    stroke_px: int = Field(default=3, ge=0, le=32)
    max_chars_per_line: int = Field(default=24, ge=8, le=120)
    line_spacing: float = 1.15


# --------------------------------------------------------------------------- audio


class Narration(Frozen):
    text: str
    voice: str | None = None
    asset_id: str | None = None
    gain_db: float = 0.0


MusicState = Literal[
    "none",  # nothing requested
    "suggested",  # LLM described a track; not yet searched
    "candidates_ready",  # provider returned options; awaiting the user's pick
    "approved",  # user picked one; download permitted
    "resolved",  # downloaded and on disk
    "declined",  # user wants no music
]


class MusicQuery(Frozen):
    """What the LLM thinks the video should sound like."""

    mood: str
    genre: str | None = None
    bpm: tuple[int, int] | None = None
    energy: Literal["low", "medium", "high"] = "medium"
    instruments: tuple[str, ...] = ()


class MusicCandidate(Frozen):
    id: str
    title: str
    source_url: str
    licence: str
    duration: Seconds
    attribution: str | None = None


class Music(Frozen):
    """Music, as a state machine.

    Nothing is downloaded until `state` reaches `approved`, and that transition can
    only be made by an explicit user choice. See docs/06-decisions.md D4.
    """

    state: MusicState = "none"
    query: MusicQuery | None = None
    candidates: tuple[MusicCandidate, ...] = ()
    selected_candidate_id: str | None = None
    asset_id: str | None = None
    gain_db: float = -18.0
    duck_under_narration: bool = True

    @property
    def is_renderable(self) -> bool:
        """True when the renderer may proceed.

        Any other state means the user has not been asked yet, which is a pipeline
        bug rather than a user error.
        """
        return self.state in ("none", "declined", "resolved")


class Captions(Frozen):
    enabled: bool = True
    style_ref: str = "caption"
    source: Literal["narration", "transcript"] = "narration"


# --------------------------------------------------------------------------- assets


class AssetProvenance(Frozen):
    """Where an asset came from. Records licence for anything fetched from the network."""

    provider: str
    model: str | None = None
    seed: int | None = None
    source_url: str | None = None
    licence: str | None = None
    prompt: str | None = None


class Asset(Frozen):
    """A file on disk, addressed by content hash.

    `path` is always relative to the project directory, so a project stays portable.
    """

    kind: Literal["image", "audio", "video"]
    path: str
    sha256: str
    provenance: AssetProvenance


# --------------------------------------------------------------------------- scene


class Scene(Frozen):
    id: str
    duration: Seconds = Field(default=4.0, ge=0.3, le=120.0)
    visual: Visual
    motion: Motion = StillMotion()
    narration: Narration | None = None
    overlays: tuple[Overlay, ...] = ()
    transition_in: Transition = Transition()
    note: str | None = Field(default=None, description="LLM rationale. Never rendered.")


# --------------------------------------------------------------------------- timeline


class Timeline(Frozen):
    schema_version: int = SCHEMA_VERSION
    version: int = Field(default=1, ge=1, description="Increments on every applied edit.")
    intent: str
    canvas: Canvas = Canvas()
    seed: int = Field(default=0, description="Seeds anything not explicitly seeded.")

    scenes: tuple[Scene, ...] = ()
    music: Music = Music()
    captions: Captions = Captions()
    styles: dict[str, TextStyle] = Field(default_factory=lambda: {"caption": TextStyle()})
    assets: dict[str, Asset] = Field(default_factory=dict)

    @property
    def duration(self) -> float:
        """Total runtime. Transitions overlap the outgoing scene, so they subtract."""
        total = sum(s.duration for s in self.scenes)
        overlap = sum(s.transition_in.duration for s in self.scenes[1:])
        return max(0.0, total - overlap)

    def scene(self, scene_id: str) -> Scene | None:
        return next((s for s in self.scenes if s.id == scene_id), None)

    @property
    def unresolved_visuals(self) -> tuple[Scene, ...]:
        """Scenes whose visual still needs generating."""
        return tuple(
            s
            for s in self.scenes
            if isinstance(s.visual, GeneratedVisual) and s.visual.asset_id is None
        )

    @property
    def unresolved_narration(self) -> tuple[Scene, ...]:
        return tuple(s for s in self.scenes if s.narration and s.narration.asset_id is None)


class Project(Frozen):
    """Project metadata. The timelines themselves are separate files on disk."""

    id: str
    intent: str
    created_at: str
    updated_at: str
    current_version: int = 1
    schema_version: int = SCHEMA_VERSION
