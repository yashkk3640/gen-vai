"""The conversational edit loop: a change request against an existing project.

The model returns operations, not a timeline. They are validated as a set, applied
atomically, shown as a diff, and only then re-rendered. Every version is kept, so any
edit is reversible.

What the model is actually asked for is deliberately *not* the full op vocabulary.
Twenty-four discriminated union members inline into an enormous decoding grammar, and a
3B model handles that about as well as you would expect. It gets a dozen plain verbs and
shot *positions* instead - the way the user phrased the request in the first place - and
`to_ops` translates that into the precise vocabulary here, where mistakes are cheap and
visible.
"""

from typing import Literal

from pydantic import Field

from genvai.errors import GenvaiError
from genvai.ops import (
    EditOp,
    EditPlan,
    NudgeClipSpan,
    RemoveMusic,
    RemoveScene,
    ReorderScenes,
    ScaleAllDurations,
    SetClipMuted,
    SetClipSpeed,
    SetMusicGain,
    SetSceneCaption,
    SetSceneDuration,
    SetSceneRole,
)
from genvai.ops import apply as apply_ops
from genvai.ops import describe as describe_ops
from genvai.ports import LLMPort
from genvai.timeline import ClipVisual, Frozen, Timeline

Action = Literal[
    "remove",
    "reorder",
    "retime",
    "nudge",
    "speed",
    "caption",
    "hook",
    "mute",
    "unmute",
    "pace",
    "music_off",
    "music_level",
]
"""What the model may ask for. A closed list of plain verbs, not the full op vocabulary.

Each maps to one typed operation. Anything a user asks for that is not here simply does
not happen, which is a better failure than a model improvising against a schema it only
half understands.
"""


class EditCommand(Frozen):
    """One requested change, in the user's terms rather than the schema's."""

    action: Action
    shot: int = Field(
        default=0, description="Which shot, counting from 1. 0 when it applies to all."
    )
    value: float = Field(default=0.0, description="Seconds, a speed or pace factor, or decibels.")
    text: str = Field(default="", description="Caption text, when the action needs one.")
    order: tuple[int, ...] = Field(
        default=(), description="New shot order, as positions, for a reorder."
    )


class EditRequest(Frozen):
    """What the model returns for a change request."""

    commands: tuple[EditCommand, ...]
    note: str = Field(default="", description="One line on how the request was read.")


SYSTEM_PROMPT = (
    "You edit short videos. The user describes a change; you translate it into "
    "commands. Refer to shots by their number. Reply with JSON only."
)

_ACTION_HELP = """\
remove        drop a shot                       shot
reorder       rearrange every shot              order (all shot numbers, once each)
retime        hold a shot for N seconds         shot, value=seconds
nudge         show more before/after a clip     shot, value=seconds (negative = earlier)
speed         slow down or speed up a clip      shot, value=factor (0.5 = half speed)
caption       set or clear on-screen text       shot, text ("" clears it)
hook          make a shot the opener            shot
mute          silence a clip's own audio        shot
unmute        keep a clip's own audio           shot
pace          make the whole reel faster/slower value=factor (0.8 = 20% faster)
music_off     drop the music                    -
music_level   change music loudness             value=decibels (negative = quieter)"""


def summarise(timeline: Timeline) -> str:
    """Describe the current cut for the model, one line per shot.

    Positions, not ids: the model is answering a request phrased as "the third clip",
    and giving it ids to map would be an extra step to get wrong.
    """
    lines = []
    for index, scene in enumerate(timeline.scenes, start=1):
        kind = "clip" if isinstance(scene.visual, ClipVisual) else "photo"
        caption = next((o.content for o in scene.overlays if hasattr(o, "content")), "")
        parts = [f"shot {index}", kind, f"{scene.duration:.1f}s", scene.role]
        if caption:
            parts.append(f'caption "{caption}"')
        if scene.note:
            parts.append(f"({scene.note})")
        lines.append("  " + "  ".join(parts))
    music = "music on" if timeline.music.state == "resolved" else "no music"
    return "\n".join(lines) + f"\n  {music}"


def interpret(request: str, timeline: Timeline, llm: LLMPort) -> EditRequest:
    """Turn "shot 2 is too fast" into commands.

    The model sees a summary of the cut rather than the timeline itself, so the context
    stays small and it cannot reference fields it has no business writing.
    """
    prompt = (
        f"Here is the current cut:\n{summarise(timeline)}\n\n"
        f"The user asks: {request}\n\n"
        f"Available commands:\n{_ACTION_HELP}\n\n"
        "Return only the commands needed. Leave unused fields at their defaults."
    )
    return llm.structured(prompt, EditRequest, system=SYSTEM_PROMPT)


def to_ops(edit: EditRequest, timeline: Timeline) -> tuple[EditOp, ...]:
    """Translate commands into typed operations, dropping what cannot apply.

    Pure, and forgiving: a command naming shot 9 of a 4-shot reel is skipped rather than
    raising, because one misread instruction should not cost the user the other three.
    Whatever survives is still validated before anything is applied.
    """
    ops: list[EditOp] = []
    for command in edit.commands:
        op = _translate(command, timeline)
        if op is not None:
            ops.append(op)
    return tuple(ops)


def _translate(command: EditCommand, timeline: Timeline) -> EditOp | None:
    scene = _shot(command.shot, timeline)
    is_clip = scene is not None and isinstance(scene.visual, ClipVisual)

    match command.action:
        case "remove" if scene:
            return RemoveScene(scene_id=scene.id)
        case "reorder":
            ids = [_shot(n, timeline) for n in command.order]
            if len(ids) != len(timeline.scenes) or any(s is None for s in ids):
                return None
            return ReorderScenes(scene_ids=tuple(s.id for s in ids))  # type: ignore[union-attr]
        case "retime" if scene and command.value > 0:
            return SetSceneDuration(scene_id=scene.id, seconds=command.value)
        case "nudge" if is_clip and command.value:
            return NudgeClipSpan(scene_id=scene.id, seconds=command.value)  # type: ignore[union-attr]
        case "speed" if is_clip and command.value > 0:
            return SetClipSpeed(scene_id=scene.id, speed=command.value)  # type: ignore[union-attr]
        case "caption" if scene:
            return SetSceneCaption(scene_id=scene.id, content=command.text)
        case "hook" if scene:
            return SetSceneRole(scene_id=scene.id, role="hook")
        case "mute" | "unmute" if is_clip:
            return SetClipMuted(scene_id=scene.id, mute=command.action == "mute")  # type: ignore[union-attr]
        case "pace" if command.value > 0:
            return ScaleAllDurations(factor=command.value)
        case "music_off":
            return RemoveMusic()
        case "music_level" if command.value:
            return SetMusicGain(gain_db=command.value)
    return None


def _shot(position: int, timeline: Timeline):
    """Resolve a 1-based shot number, or None when it does not exist."""
    if 1 <= position <= len(timeline.scenes):
        return timeline.scenes[position - 1]
    return None


def apply_plan(timeline: Timeline, plan: EditPlan) -> Timeline:
    """Validate the whole op list, then apply it, returning the next version.

    Raises `InvalidEditOp` before touching anything if any op is inapplicable.
    """
    return apply_ops(timeline, plan.ops)


def diff(before: Timeline, ops: tuple[EditOp, ...]) -> tuple[str, ...]:
    """Readable summary of what will change, shown before re-rendering."""
    return describe_ops(before, ops)


def edit(request: str, timeline: Timeline, llm: LLMPort) -> tuple[Timeline, tuple[str, ...]]:
    """One round of conversation: text in, a new timeline and its diff out.

    Raises `GenvaiError` when nothing in the request could be turned into a change, so
    the caller can say so rather than silently producing an identical version.
    """
    ops = to_ops(interpret(request, timeline, llm), timeline)
    if not ops:
        raise GenvaiError(
            "Nothing in that request mapped to a change I can make. "
            "Try naming a shot number, for example: 'make shot 2 shorter'."
        )
    return apply_ops(timeline, ops), describe_ops(timeline, ops)
