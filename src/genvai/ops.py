"""Edit operations: the closed vocabulary the LLM writes when the user asks for a change.

On an edit request the model returns `{"ops": [...]}`, never a replacement timeline.
Each op is small, individually validatable, and applied atomically with the rest -
see 'Typed edit operations' in docs/06-decisions.md for why.

`apply` is pure and total: `Timeline -> Timeline`. Validation happens first, over the
whole list, so a rejected op leaves the project untouched.
"""

from typing import Annotated, Literal

from pydantic import Field

from genvai.timeline import (
    Canvas,
    CaptionMode,
    Export,
    Frozen,
    Motion,
    MusicQuery,
    Scene,
    SceneRole,
    TextStyle,
    Timeline,
    Transition,
)

MIN_SCENE_SECONDS = 0.3
MAX_SCENE_SECONDS = 120.0


# --------------------------------------------------------------------------- timing


class SetSceneDuration(Frozen):
    op: Literal["set_scene_duration"] = "set_scene_duration"
    scene_id: str
    seconds: float


class ScaleAllDurations(Frozen):
    """ "Make the whole thing faster." Scales every scene by one factor."""

    op: Literal["scale_all_durations"] = "scale_all_durations"
    factor: float = Field(gt=0.0, le=10.0)


# --------------------------------------------------------------------------- content


class SetVisualPrompt(Frozen):
    """Rewrite a scene's image prompt.

    Clears `asset_id`, so the resolver regenerates. Refuses if the visual is approved
    unless `force` is set.
    """

    op: Literal["set_visual_prompt"] = "set_visual_prompt"
    scene_id: str
    prompt: str
    negative_prompt: str | None = None
    force: bool = False


class SetMotion(Frozen):
    op: Literal["set_motion"] = "set_motion"
    scene_id: str
    motion: Motion


class SetNarration(Frozen):
    """Rewrite a line. Clears the synthesised audio so it is re-spoken."""

    op: Literal["set_narration"] = "set_narration"
    scene_id: str
    text: str


class SetOverlayText(Frozen):
    op: Literal["set_overlay_text"] = "set_overlay_text"
    scene_id: str
    index: int = Field(ge=0)
    content: str


class SetTransition(Frozen):
    op: Literal["set_transition"] = "set_transition"
    scene_id: str
    transition: Transition


# --------------------------------------------------------------------------- structure


class InsertScene(Frozen):
    op: Literal["insert_scene"] = "insert_scene"
    scene: Scene
    after_scene_id: str | None = Field(default=None, description="None inserts at the start.")


class RemoveScene(Frozen):
    op: Literal["remove_scene"] = "remove_scene"
    scene_id: str


class ReorderScenes(Frozen):
    """Must be an exact permutation of the existing ids - no adds, drops or duplicates."""

    op: Literal["reorder_scenes"] = "reorder_scenes"
    scene_ids: tuple[str, ...]


# --------------------------------------------------------------------------- global


class SetCanvas(Frozen):
    op: Literal["set_canvas"] = "set_canvas"
    canvas: Canvas


class SetCaptions(Frozen):
    op: Literal["set_captions"] = "set_captions"
    enabled: bool
    mode: CaptionMode | None = Field(default=None, description="None leaves the mode unchanged.")
    style_ref: str | None = None


class SetSceneRole(Frozen):
    """Retarget a scene - "make the third clip the hook"."""

    op: Literal["set_scene_role"] = "set_scene_role"
    scene_id: str
    role: SceneRole


class SetStyleSuffix(Frozen):
    """Change the phrase appended to every image prompt.

    Clears every unapproved generated visual, since the whole point is that they
    re-render in a consistent look.
    """

    op: Literal["set_style_suffix"] = "set_style_suffix"
    style_suffix: str


class SetExport(Frozen):
    op: Literal["set_export"] = "set_export"
    export: Export


class SetStyle(Frozen):
    op: Literal["set_style"] = "set_style"
    style_ref: str
    style: TextStyle


# --------------------------------------------------------------------------- music


class SetMusicQuery(Frozen):
    """Ask for different music. Resets the state machine to `suggested`."""

    op: Literal["set_music_query"] = "set_music_query"
    query: MusicQuery


class SelectMusicCandidate(Frozen):
    """Pick a candidate. Moves to `approved`; the download still needs confirmation."""

    op: Literal["select_music_candidate"] = "select_music_candidate"
    candidate_id: str


class RemoveMusic(Frozen):
    op: Literal["remove_music"] = "remove_music"


class SetMusicGain(Frozen):
    op: Literal["set_music_gain"] = "set_music_gain"
    gain_db: float = Field(ge=-60.0, le=12.0)


EditOp = Annotated[
    SetSceneDuration
    | ScaleAllDurations
    | SetVisualPrompt
    | SetMotion
    | SetNarration
    | SetOverlayText
    | SetTransition
    | InsertScene
    | RemoveScene
    | ReorderScenes
    | SetCanvas
    | SetCaptions
    | SetSceneRole
    | SetStyleSuffix
    | SetExport
    | SetStyle
    | SetMusicQuery
    | SelectMusicCandidate
    | RemoveMusic
    | SetMusicGain,
    Field(discriminator="op"),
]


class EditPlan(Frozen):
    """What the LLM returns for a change request.

    `rationale` is shown to the user alongside the diff, so they can see why the model
    interpreted the request the way it did before agreeing to re-render.
    """

    ops: tuple[EditOp, ...]
    rationale: str = ""


# --------------------------------------------------------------------------- transforms


def validate(timeline: Timeline, ops: tuple[EditOp, ...]) -> None:
    """Check every op against the timeline. Raises `InvalidEditOp` on the first failure.

    Validation runs over the whole list before anything is applied, so application is
    all-or-nothing. Note that ops are checked against the *original* timeline; an op
    list whose validity depends on an earlier op in the same list (inserting a scene
    then retiming it) is rejected. That is deliberate - it keeps validation a pure
    function of one state rather than a simulation.
    """
    raise NotImplementedError


def apply(timeline: Timeline, ops: tuple[EditOp, ...]) -> Timeline:
    """Apply a validated op list, returning a new timeline with `version` incremented.

    Pure and total: given ops that pass `validate`, this cannot fail.
    """
    raise NotImplementedError


def describe(timeline: Timeline, ops: tuple[EditOp, ...]) -> tuple[str, ...]:
    """Human-readable lines describing what each op will change.

    Shown to the user before re-rendering, per requirement F6.4.
    """
    raise NotImplementedError
