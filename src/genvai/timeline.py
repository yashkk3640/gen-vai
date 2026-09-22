"""The Timeline: the single source of truth for a video.

Every model here is frozen. Edits produce new values via the pure transforms in
`ops.py`; nothing is ever mutated in place.

The prose contract, including field-by-field notes and the music state machine,
lives in docs/timeline.md.
"""

from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field

SCHEMA_VERSION = 1

Seconds = Annotated[float, Field(ge=0.0)]
Unit = Annotated[float, Field(ge=0.0, le=1.0)]
"""A fraction of the canvas, so geometry is resolution-independent."""

Rect = tuple[Unit, Unit, Unit, Unit]
"""(x, y, w, h) as fractions of the frame it applies to."""


class Frozen(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


# --------------------------------------------------------------------------- canvas


class SafeArea(Frozen):
    """Fractions of canvas height obscured by platform UI chrome.

    On a vertical feed the top carries the status bar and the bottom carries the
    caption, handle and action rail. Text placed outside these bounds is simply not
    read. Defaults are sized for a 9:16 short-form feed; widen them per platform.
    """

    top: Unit = 0.10
    bottom: Unit = 0.20
    left: Unit = 0.05
    right: Unit = 0.18


class Canvas(Frozen):
    width: int = Field(default=1080, ge=16, le=4096, multiple_of=2)
    height: int = Field(default=1920, ge=16, le=4096, multiple_of=2)
    fps: int = Field(default=30, ge=1, le=120)
    background: str = "#000000"
    safe_area: SafeArea = SafeArea()

    @property
    def aspect(self) -> float:
        return self.width / self.height

    @property
    def is_vertical(self) -> bool:
        return self.height > self.width


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


Fit = Literal["cover", "contain", "blur_pad"]
"""How source material fills a canvas of a different shape.

`cover` crops to fill, `contain` letterboxes, `blur_pad` fills the margins with a
blurred copy of the source - the usual choice when a landscape phone clip has to sit
in a vertical frame without losing its edges.
"""


class AssetVisual(Frozen):
    """A photo the user supplied."""

    kind: Literal["asset"] = "asset"
    asset_id: str
    fit: Fit = "cover"
    crop: Rect | None = Field(
        default=None, description="None auto-fits; a rect picks the region to keep."
    )


class ClipVisual(Frozen):
    """A trimmed span of one of the user's own video clips.

    The core of camera-roll editing. A phone clip is typically 10-20 seconds of which
    two are worth keeping, so the span is the edit: `source_start` and `source_end`
    say which part survives.

    `Scene.duration` must equal `source_duration / speed`. Keeping them as separate
    fields rather than deriving one lets a scene be retimed and the trim adjusted
    independently, which is what a user means by "hold that shot longer" versus
    "show more of that clip".
    """

    kind: Literal["clip"] = "clip"
    asset_id: str
    source_start: Seconds = 0.0
    source_end: Seconds
    crop: Rect | None = None
    fit: Fit = "cover"
    speed: float = Field(default=1.0, gt=0.1, le=10.0)
    mute: bool = Field(
        default=True,
        description=(
            "Camera-roll audio is usually wind and chatter. Muted by default; unmute "
            "deliberately when the clip's own sound is the point."
        ),
    )

    @property
    def source_duration(self) -> float:
        return max(0.0, self.source_end - self.source_start)

    @property
    def output_duration(self) -> float:
        """How long this occupies the timeline once speed is applied."""
        return self.source_duration / self.speed


class CardVisual(Frozen):
    """A procedurally drawn typographic card. The fallback that always works."""

    kind: Literal["card"] = "card"
    text: str
    palette: str = "default"


class ColorVisual(Frozen):
    kind: Literal["color"] = "color"
    color: str = "#000000"


Visual = Annotated[
    ClipVisual | AssetVisual | GeneratedVisual | CardVisual | ColorVisual,
    Field(discriminator="kind"),
]


# --------------------------------------------------------------------------- motion

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
    size_pct: float = Field(default=5.5, gt=0.0, le=30.0, description="Percent of canvas height.")
    color: str = "#FFFFFF"
    stroke_color: str = "#000000"
    stroke_px: int = Field(default=6, ge=0, le=32)
    max_chars_per_line: int = Field(default=20, ge=8, le=120)
    line_spacing: float = 1.15
    uppercase: bool = False

    # Kinetic caption styling. Applies when Captions.mode is "word" or "line".
    highlight_color: str = "#FFD400"
    highlight_mode: Literal["color", "box", "scale"] = "color"


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


class BeatMap(Frozen):
    """Onset times in the music bed.

    Cuts landing on the beat are most of what separates a reel that feels edited
    from one that feels assembled. Onset detection is signal processing, not a
    model - it runs on CPU in about a second.
    """

    bpm: float = Field(ge=20.0, le=300.0)
    beats: tuple[Seconds, ...] = ()
    downbeats: tuple[Seconds, ...] = Field(
        default=(), description="Bar starts. Stronger cut points than ordinary beats."
    )


class Music(Frozen):
    """Music, as a state machine.

    Nothing is downloaded until `state` reaches `approved`, and that transition can
    only be made by an explicit user choice. See 'Music consent' in docs/decisions.md.
    """

    state: MusicState = "none"
    query: MusicQuery | None = None
    candidates: tuple[MusicCandidate, ...] = ()
    selected_candidate_id: str | None = None
    asset_id: str | None = None
    gain_db: float = -18.0
    duck_under_narration: bool = True
    beat_map: BeatMap | None = Field(
        default=None, description="Populated on resolve, when a track exists to analyse."
    )

    @property
    def is_renderable(self) -> bool:
        """True when the renderer may proceed.

        Any other state means the user has not been asked yet, which is a pipeline
        bug rather than a user error.
        """
        return self.state in ("none", "declined", "resolved")


CaptionMode = Literal["static", "line", "word"]
"""How captions animate.

`static` shows the whole line for the scene. `line` swaps a line at a time.
`word` highlights each word as it is spoken - the current short-form convention, and
the cheapest large gain in perceived production value. All three are ASS subtitle
features, so none needs a model.
"""


class Captions(Frozen):
    enabled: bool = True
    mode: CaptionMode = "word"
    style_ref: str = "caption"
    source: Literal["narration", "transcript"] = "narration"


# --------------------------------------------------------------------------- export

AudioVariant = Literal["full", "narration_only", "silent"]
"""Which audio bed an output carries.

Not a convenience. Short-form feeds weight reach toward *attached* trending audio,
and a track baked into the file is not an attached sound - it forfeits that signal
and risks Content-ID muting besides. So the default is to emit a `narration_only`
cut alongside the full mix: upload that one and attach the trending sound in the app.

See docs/backlog.md for the reasoning and its limits.
"""


class Export(Frozen):
    """What gets written out, and in what shape."""

    audio_variants: tuple[AudioVariant, ...] = ("full", "narration_only")
    snap_cuts_to_beat: bool = Field(
        default=True,
        description=(
            "Nudge scene boundaries onto the nearest beat when a beat map exists. "
            "Scene durations shift by a few frames; the cut lands with the music."
        ),
    )
    seamless_loop: bool = Field(
        default=False,
        description=(
            "Match the last frame to the first so the clip loops invisibly. "
            "Re-watches count as watch time, so a clean loop is worth real reach."
        ),
    )


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


SceneRole = Literal["hook", "body", "payoff", "cta"]
"""What a scene is for.

Making this explicit forces the planner to write an opening deliberately rather than
starting with scene one of an essay. Retention in short form is decided in the first
couple of seconds, so `hook` is the single most consequential scene in the timeline.
"""


class Scene(Frozen):
    id: str
    duration: Seconds = Field(
        default=2.2,
        ge=0.3,
        le=120.0,
        description=(
            "Short-form pacing runs roughly 1.5-2.5s per beat. The default targets that; "
            "longer scenes are legitimate but should be chosen, not inherited."
        ),
    )
    role: SceneRole = "body"
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

    style_suffix: str = Field(
        default="",
        description=(
            "Appended to every generated image prompt. Holding one phrase constant "
            "across scenes is the cheapest defence against each shot looking like it "
            "came from a different video. A partial fix - see "
            "'Character and scene consistency' in docs/backlog.md."
        ),
    )

    scenes: tuple[Scene, ...] = ()
    music: Music = Music()
    captions: Captions = Captions()
    export: Export = Export()
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
    def hook(self) -> Scene | None:
        """The opening beat, if the planner marked one."""
        return next((s for s in self.scenes if s.role == "hook"), None)

    @property
    def mean_scene_duration(self) -> float:
        """Average beat length. A quick read on whether the cut is paced for short form."""
        return self.duration / len(self.scenes) if self.scenes else 0.0

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
