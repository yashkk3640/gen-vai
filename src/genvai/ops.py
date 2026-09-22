"""Edit operations: the closed vocabulary the LLM writes when the user asks for a change.

On an edit request the model returns `{"ops": [...]}`, never a replacement timeline.
Each op is small, individually validatable, and applied atomically with the rest -
see 'Typed edit operations' in docs/decisions.md for why.

`apply` is pure and total: `Timeline -> Timeline`. Validation happens first, over the
whole list, so a rejected op leaves the project untouched.
"""

from collections.abc import Callable
from typing import Annotated, Literal

from pydantic import Field

from genvai.errors import InvalidEditOp
from genvai.timeline import (
    AssetVisual,
    Canvas,
    CaptionMode,
    ClipVisual,
    Export,
    Fit,
    Frozen,
    GeneratedVisual,
    Motion,
    MusicQuery,
    Narration,
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


class SetClipSpan(Frozen):
    """Retrim which part of a source clip is used - "start that one a bit later".

    Adjusts `Scene.duration` to match, since the two must stay consistent.
    """

    op: Literal["set_clip_span"] = "set_clip_span"
    scene_id: str
    source_start: float = Field(ge=0.0)
    source_end: float = Field(gt=0.0)


class NudgeClipSpan(Frozen):
    """Shift a trim without changing its length - "show a bit more before that".

    Separate from SetClipSpan because it is what users actually ask for, and it
    leaves scene duration and therefore the beat alignment untouched.
    """

    op: Literal["nudge_clip_span"] = "nudge_clip_span"
    scene_id: str
    seconds: float = Field(description="Negative moves earlier in the source.")


class SetClipSpeed(Frozen):
    """Slow-motion or speed-up. Rescales scene duration accordingly."""

    op: Literal["set_clip_speed"] = "set_clip_speed"
    scene_id: str
    speed: float = Field(gt=0.1, le=10.0)


class SetCrop(Frozen):
    """Reframe a photo or clip inside the canvas - the 16:9 to 9:16 problem."""

    op: Literal["set_crop"] = "set_crop"
    scene_id: str
    crop: tuple[float, float, float, float] | None = Field(
        default=None, description="None restores auto-fit."
    )
    fit: Fit | None = None


class SetClipMuted(Frozen):
    """Unmute a clip whose own audio is the point."""

    op: Literal["set_clip_muted"] = "set_clip_muted"
    scene_id: str
    mute: bool


class ReplaceSource(Frozen):
    """Swap which media a scene uses - "use the other take of that".

    Keeps timing and treatment, changes only the source.
    """

    op: Literal["replace_source"] = "replace_source"
    scene_id: str
    asset_id: str


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
    | SetClipSpan
    | NudgeClipSpan
    | SetClipSpeed
    | SetCrop
    | SetClipMuted
    | ReplaceSource
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

    Runs over the whole list before anything is applied, so application is
    all-or-nothing. A half-applied list would leave the project in a state the user
    never asked for and cannot reason about.

    Ops are checked against the *original* timeline, so a list whose validity depends on
    an earlier op in the same list - inserting a scene then retiming it - is rejected.
    That is deliberate: it keeps validation a pure function of one state rather than a
    simulation of the list.
    """
    for op in ops:
        problem = _check(op, timeline)
        if problem:
            raise InvalidEditOp(op.op, problem)


def apply(timeline: Timeline, ops: tuple[EditOp, ...]) -> Timeline:
    """Apply an op list, returning a new timeline with `version` incremented.

    Validates first, so given ops that pass `validate` this cannot fail.
    """
    validate(timeline, ops)
    result = timeline
    for op in ops:
        result = _apply(op, result)
    return result.model_copy(update={"version": timeline.version + 1})


def describe(timeline: Timeline, ops: tuple[EditOp, ...]) -> tuple[str, ...]:
    """Human-readable lines for what each op will change.

    Shown before re-rendering, so the user sees the model's interpretation rather than
    discovering it in the output.
    """
    return tuple(_describe(op, timeline) for op in ops)


# ----------------------------------------------------------------------- validation


def _check(op: EditOp, timeline: Timeline) -> str | None:
    """One op's objection, or None. Pydantic has already checked types and ranges."""
    scene_id = getattr(op, "scene_id", None)
    scene = timeline.scene(scene_id) if scene_id is not None else None
    if scene_id is not None and scene is None:
        return f"no scene '{scene_id}' in this timeline"

    match op:
        case SetSceneDuration():
            if not MIN_SCENE_SECONDS <= op.seconds <= MAX_SCENE_SECONDS:
                return f"{op.seconds}s is outside {MIN_SCENE_SECONDS}-{MAX_SCENE_SECONDS}s"
        case SetClipSpan():
            if op.source_end <= op.source_start:
                return f"the span ends ({op.source_end}s) at or before it starts"
            if not isinstance(scene.visual, ClipVisual):
                return f"scene '{scene_id}' is not a video clip"
        case NudgeClipSpan() | SetClipSpeed() | SetClipMuted():
            if not isinstance(scene.visual, ClipVisual):
                return f"scene '{scene_id}' is not a video clip"
        case SetCrop():
            if not isinstance(scene.visual, ClipVisual | AssetVisual):
                return f"scene '{scene_id}' has nothing to crop"
        case SetVisualPrompt():
            if not isinstance(scene.visual, GeneratedVisual):
                return f"scene '{scene_id}' uses real footage, not a generated image"
            if scene.visual.approved and not op.force:
                return "that image was approved; pass force to replace it anyway"
        case ReplaceSource():
            if op.asset_id not in timeline.assets:
                return f"no asset '{op.asset_id}' in this project"
        case SetOverlayText():
            if op.index >= len(scene.overlays):
                return f"scene '{scene_id}' has no overlay {op.index}"
        case InsertScene():
            if timeline.scene(op.scene.id) is not None:
                return f"scene id '{op.scene.id}' is already taken"
            if op.after_scene_id and timeline.scene(op.after_scene_id) is None:
                return f"no scene '{op.after_scene_id}' to insert after"
        case RemoveScene():
            if len(timeline.scenes) <= 1:
                return "that is the only scene left"
        case ReorderScenes():
            existing = sorted(s.id for s in timeline.scenes)
            if sorted(op.scene_ids) != existing:
                return "must list every scene exactly once: " + ", ".join(existing)
        case SelectMusicCandidate():
            if op.candidate_id not in {c.id for c in timeline.music.candidates}:
                return f"no candidate '{op.candidate_id}' was suggested"
    return None


# ------------------------------------------------------------------------ application


def _apply(op: EditOp, timeline: Timeline) -> Timeline:
    """Apply one validated op. Total: given a passing `_check`, this cannot fail."""
    match op:
        case SetSceneDuration():
            return _map_scene(
                timeline, op.scene_id, lambda s: s.model_copy(update={"duration": op.seconds})
            )

        case ScaleAllDurations():
            return _map_scenes(
                timeline,
                lambda s: s.model_copy(update={"duration": _clamp(s.duration * op.factor)}),
            )

        case SetClipSpan():
            return _map_scene(
                timeline,
                op.scene_id,
                lambda s: _retimed(
                    s,
                    s.visual.model_copy(
                        update={"source_start": op.source_start, "source_end": op.source_end}
                    ),
                ),
            )

        case NudgeClipSpan():
            return _map_scene(
                timeline, op.scene_id, lambda s: _retimed(s, _shifted(s.visual, op.seconds))
            )

        case SetClipSpeed():
            return _map_scene(
                timeline,
                op.scene_id,
                lambda s: _retimed(s, s.visual.model_copy(update={"speed": op.speed})),
            )

        case SetCrop():
            patch: dict[str, object] = {"crop": op.crop}
            if op.fit is not None:
                patch["fit"] = op.fit
            return _map_scene(
                timeline,
                op.scene_id,
                lambda s: s.model_copy(update={"visual": s.visual.model_copy(update=patch)}),
            )

        case SetClipMuted():
            return _map_scene(
                timeline,
                op.scene_id,
                lambda s: s.model_copy(
                    update={"visual": s.visual.model_copy(update={"mute": op.mute})}
                ),
            )

        case ReplaceSource():
            return _map_scene(
                timeline,
                op.scene_id,
                lambda s: s.model_copy(
                    update={"visual": s.visual.model_copy(update={"asset_id": op.asset_id})}
                ),
            )

        case SetVisualPrompt():
            # Clearing asset_id is what forces the resolver to regenerate.
            return _map_scene(
                timeline,
                op.scene_id,
                lambda s: s.model_copy(
                    update={
                        "visual": s.visual.model_copy(
                            update={
                                "prompt": op.prompt,
                                "negative_prompt": op.negative_prompt,
                                "asset_id": None,
                                "approved": False,
                            }
                        )
                    }
                ),
            )

        case SetMotion():
            return _map_scene(
                timeline, op.scene_id, lambda s: s.model_copy(update={"motion": op.motion})
            )

        case SetNarration():
            return _map_scene(
                timeline,
                op.scene_id,
                lambda s: s.model_copy(
                    update={
                        "narration": Narration(
                            text=op.text, voice=s.narration.voice if s.narration else None
                        )
                    }
                ),
            )

        case SetOverlayText():
            return _map_scene(timeline, op.scene_id, lambda s: _captioned(s, op.index, op.content))

        case SetTransition():
            return _map_scene(
                timeline,
                op.scene_id,
                lambda s: s.model_copy(update={"transition_in": op.transition}),
            )

        case SetSceneRole():
            return _map_scenes(timeline, lambda s: _reroled(s, op.scene_id, op.role))

        case InsertScene():
            scenes = list(timeline.scenes)
            at = (
                0
                if op.after_scene_id is None
                else next(i + 1 for i, s in enumerate(scenes) if s.id == op.after_scene_id)
            )
            scenes.insert(at, op.scene)
            return timeline.model_copy(update={"scenes": tuple(scenes)})

        case RemoveScene():
            return timeline.model_copy(
                update={"scenes": tuple(s for s in timeline.scenes if s.id != op.scene_id)}
            )

        case ReorderScenes():
            by_id = {s.id: s for s in timeline.scenes}
            return timeline.model_copy(update={"scenes": tuple(by_id[i] for i in op.scene_ids)})

        case SetCanvas():
            return timeline.model_copy(update={"canvas": op.canvas})

        case SetCaptions():
            patch = {"enabled": op.enabled}
            if op.mode is not None:
                patch["mode"] = op.mode
            if op.style_ref is not None:
                patch["style_ref"] = op.style_ref
            return timeline.model_copy(
                update={"captions": timeline.captions.model_copy(update=patch)}
            )

        case SetStyleSuffix():
            # Every unapproved generated visual re-renders, which is the whole point.
            return _map_scenes(
                timeline.model_copy(update={"style_suffix": op.style_suffix}), _unresolved
            )

        case SetExport():
            return timeline.model_copy(update={"export": op.export})

        case SetStyle():
            return timeline.model_copy(
                update={"styles": {**timeline.styles, op.style_ref: op.style}}
            )

        case SetMusicQuery():
            return timeline.model_copy(
                update={
                    "music": timeline.music.model_copy(
                        update={
                            "query": op.query,
                            "state": "suggested",
                            "candidates": (),
                            "selected_candidate_id": None,
                            "asset_id": None,
                        }
                    )
                }
            )

        case SelectMusicCandidate():
            # `approved` only permits the download; fetching still asks separately.
            return timeline.model_copy(
                update={
                    "music": timeline.music.model_copy(
                        update={"selected_candidate_id": op.candidate_id, "state": "approved"}
                    )
                }
            )

        case RemoveMusic():
            return timeline.model_copy(
                update={
                    "music": timeline.music.model_copy(
                        update={
                            "state": "declined",
                            "asset_id": None,
                            "selected_candidate_id": None,
                        }
                    )
                }
            )

        case SetMusicGain():
            return timeline.model_copy(
                update={"music": timeline.music.model_copy(update={"gain_db": op.gain_db})}
            )

    return timeline


# ------------------------------------------------------------------------ description


def _describe(op: EditOp, timeline: Timeline) -> str:
    """One readable line.

    Scenes are named by position, because "shot 3" means something to someone who just
    watched the reel while "s7" does not. Ids are stable across edits; positions are
    what the user actually saw.
    """
    where = _where(getattr(op, "scene_id", None), timeline)
    match op:
        case SetSceneDuration():
            return f"{where}: hold for {op.seconds:.1f}s"
        case ScaleAllDurations():
            return f"whole reel {op.factor:.2f}x {'faster' if op.factor < 1 else 'slower'}"
        case SetClipSpan():
            return f"{where}: use {op.source_start:.1f}-{op.source_end:.1f}s of the clip"
        case NudgeClipSpan():
            direction = "later" if op.seconds > 0 else "earlier"
            return f"{where}: shift {abs(op.seconds):.1f}s {direction}"
        case SetClipSpeed():
            return f"{where}: play at {op.speed:.2f}x"
        case SetCrop():
            return f"{where}: {'reframe' if op.crop else 'reset framing'}"
        case SetClipMuted():
            return f"{where}: {'mute' if op.mute else 'unmute'} its own audio"
        case ReplaceSource():
            return f"{where}: use a different take"
        case SetVisualPrompt():
            return f"{where}: regenerate as '{op.prompt[:50]}'"
        case SetMotion():
            return f"{where}: camera move becomes {op.motion.kind}"
        case SetNarration():
            return f"{where}: say '{op.text[:50]}'"
        case SetOverlayText():
            return f"{where}: caption becomes '{op.content[:50]}'"
        case SetTransition():
            return f"{where}: enter with {op.transition.kind}"
        case SetSceneRole():
            return f"{where}: becomes the {op.role}"
        case InsertScene():
            return f"add a new shot after {_where(op.after_scene_id, timeline)}"
        case RemoveScene():
            return f"{where}: remove"
        case ReorderScenes():
            return "reorder to " + " then ".join(_where(i, timeline) for i in op.scene_ids)
        case SetCanvas():
            return f"reframe to {op.canvas.width}x{op.canvas.height}"
        case SetCaptions():
            mode = f", {op.mode} mode" if op.mode else ""
            return f"captions {'on' if op.enabled else 'off'}{mode}"
        case SetStyleSuffix():
            return f"restyle every generated image: '{op.style_suffix[:50]}'"
        case SetExport():
            return "export " + ", ".join(op.export.audio_variants)
        case SetStyle():
            return f"restyle '{op.style_ref}' text"
        case SetMusicQuery():
            genre = f", {op.query.genre}" if op.query.genre else ""
            return f"find music: {op.query.mood}{genre}"
        case SelectMusicCandidate():
            return f"approve track '{op.candidate_id}' for download"
        case RemoveMusic():
            return "drop the music"
        case SetMusicGain():
            return f"music at {op.gain_db:.0f} dB"
    return op.op


# ---------------------------------------------------------------------------- helpers


def _where(scene_id: str | None, timeline: Timeline) -> str:
    if scene_id is None:
        return "the start"
    for index, scene in enumerate(timeline.scenes, start=1):
        if scene.id == scene_id:
            return f"shot {index}"
    return f"shot '{scene_id}'"


def _map_scenes(timeline: Timeline, fn: Callable[[Scene], Scene]) -> Timeline:
    return timeline.model_copy(update={"scenes": tuple(fn(s) for s in timeline.scenes)})


def _map_scene(timeline: Timeline, scene_id: str, fn: Callable[[Scene], Scene]) -> Timeline:
    return _map_scenes(timeline, lambda s: fn(s) if s.id == scene_id else s)


def _retimed(scene: Scene, clip: ClipVisual) -> Scene:
    """Attach a retrimmed clip and keep the scene's duration consistent with it.

    `Scene.duration` must equal the span divided by speed, so any op touching the trim
    or the speed has to move both together or the render desynchronises.
    """
    return scene.model_copy(update={"visual": clip, "duration": _clamp(clip.output_duration)})


def _shifted(clip: ClipVisual, seconds: float) -> ClipVisual:
    """Move a trim without changing its length, never before the clip's start.

    The far end cannot be checked here: the timeline knows the span but not the source
    file's full duration. Over-reading past the end is clamped by the renderer.
    """
    start = max(0.0, clip.source_start + seconds)
    return clip.model_copy(
        update={"source_start": start, "source_end": start + clip.source_duration}
    )


def _captioned(scene: Scene, index: int, content: str) -> Scene:
    overlays = list(scene.overlays)
    overlays[index] = overlays[index].model_copy(update={"content": content})
    return scene.model_copy(update={"overlays": tuple(overlays)})


def _reroled(scene: Scene, target_id: str, role: SceneRole) -> Scene:
    """Promote one scene, and demote whoever held the role if it can only be held once."""
    if scene.id == target_id:
        return scene.model_copy(update={"role": role})
    if role == "hook" and scene.role == "hook":
        return scene.model_copy(update={"role": "body"})
    return scene


def _unresolved(scene: Scene) -> Scene:
    visual = scene.visual
    if isinstance(visual, GeneratedVisual) and not visual.approved:
        return scene.model_copy(update={"visual": visual.model_copy(update={"asset_id": None})})
    return scene


def _clamp(seconds: float) -> float:
    return min(MAX_SCENE_SECONDS, max(MIN_SCENE_SECONDS, seconds))
