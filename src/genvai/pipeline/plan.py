"""Intent -> Timeline, with no footage at all.

Idea mode. The user describes a video and gets one; nothing is imported, every frame is
made. This is the weakest thing the project does and it is deliberately last - see
docs/backlog.md for why - but it earns its place for title cards, for the gap where a
shot is missing, and for the case where someone has an idea and no camera roll.

The model writes the script and describes each backdrop. It never sees or makes pixels:
the visual provider does that, and on a 4 GB card the honest provider is a designed
graphic rather than an imitation photograph.

The planner emits a complete storyboard in one pass. It cannot look at a generated image
and reconsider, because the LLM is unloaded before any image model loads - see
'Phase-ordered model loading' in docs/decisions.md.
"""

from pydantic import Field

from genvai.errors import PlanningError
from genvai.ports import LLMPort
from genvai.timeline import (
    Canvas,
    Frozen,
    GeneratedVisual,
    KenBurns,
    Scene,
    SceneRole,
    TextOverlay,
    Timeline,
)

BEAT_SECONDS = 2.6
"""Seconds per beat. Slightly longer than a camera-roll cut, because a generated backdrop
carries less information and the text on it has to be read."""

MIN_BEATS = 2
MAX_BEATS = 12
TEXT_LIMIT = 60


class Beat(Frozen):
    """One moment of the video, as the model describes it."""

    text: str = Field(description="What appears on screen. Short.")
    visual: str = Field(description="What the backdrop looks like. A few words.")


class Storyboard(Frozen):
    """What the model returns.

    Two fields per beat and nothing else. M3 established that schema size is what breaks
    small models, and durations are not asked for because the target length divided by
    the beat count is a better answer than a 3B model's guess.
    """

    beats: tuple[Beat, ...]


SYSTEM_PROMPT = (
    "You write short social videos. You are given an idea and return the beats of the "
    "video: what appears on screen, and what the backdrop behind it looks like. "
    "Reply with JSON only."
)


def plan(
    intent: str,
    llm: LLMPort | None,
    *,
    target_duration: float = 20.0,
    canvas: Canvas | None = None,
    seed: int = 0,
    style_suffix: str = "",
) -> Timeline:
    """Turn an idea into a renderable timeline.

    Without an `llm` - or when the model returns nothing usable - it falls back to
    splitting the intent into sentences, one per beat. Crude, but it produces a video
    rather than an error, which is the right failure for a tool whose whole promise is
    that it runs locally.
    """
    if not intent.strip():
        raise PlanningError("Nothing to make a video from. Describe what you want.")

    wanted = _beat_count(target_duration)
    beats = _from_model(intent, llm, wanted) if llm is not None else ()
    if not beats:
        beats = _from_sentences(intent, wanted)

    duration = max(1.0, target_duration) / len(beats)
    return Timeline(
        intent=intent,
        canvas=canvas or Canvas(),
        seed=seed,
        style_suffix=style_suffix,
        scenes=tuple(
            _scene(beat, index, len(beats), duration, style_suffix, seed)
            for index, beat in enumerate(beats)
        ),
    )


def _scene(
    beat: Beat, index: int, total: int, duration: float, style_suffix: str, seed: int
) -> Scene:
    role: SceneRole = "hook" if index == 0 else "payoff" if index == total - 1 else "body"
    prompt = f"{beat.visual}, {style_suffix}".strip(", ") if style_suffix else beat.visual

    # Alternating the push keeps consecutive beats from moving identically, which is what
    # makes a generated sequence read as a slideshow.
    zoom_in = index % 2 == 0
    close = (0.10, 0.10, 0.80, 0.80)
    wide = (0.0, 0.0, 1.0, 1.0)

    overlays = (
        (TextOverlay(content=beat.text[:TEXT_LIMIT], position="center"),) if beat.text else ()
    )
    return Scene(
        id=f"s{index + 1}",
        duration=duration,
        role=role,
        visual=GeneratedVisual(prompt=prompt or "abstract backdrop", seed=seed + index),
        motion=KenBurns(
            start_rect=wide if zoom_in else close,
            end_rect=close if zoom_in else wide,
        ),
        overlays=overlays,
    )


def _beat_count(target_duration: float) -> int:
    return max(MIN_BEATS, min(MAX_BEATS, round(max(1.0, target_duration) / BEAT_SECONDS)))


def _from_model(intent: str, llm: LLMPort, wanted: int) -> tuple[Beat, ...]:
    """Ask for a storyboard, and accept a partial answer.

    A model that returns six beats when asked for eight has still done the job; the
    durations simply stretch. Returning nothing at all is the only real failure, and the
    caller has a fallback for that.
    """
    prompt = (
        f"Idea: {intent}\n\n"
        f"Write {wanted} beats for a {target_seconds(wanted)} second vertical video.\n"
        "For each beat give the on-screen text - a short line, at most a dozen words, "
        "or an empty string where the picture speaks for itself - and a few words "
        "describing the backdrop behind it.\n"
        "The first beat has to earn the next two seconds of attention. "
        "The last should land."
    )
    try:
        board = llm.structured(prompt, Storyboard, system=SYSTEM_PROMPT)
    except Exception:  # noqa: BLE001 - the caller falls back rather than failing
        return ()
    return tuple(b for b in board.beats if b.text.strip() or b.visual.strip())[:MAX_BEATS]


def _from_sentences(intent: str, wanted: int) -> tuple[Beat, ...]:
    """The no-model fallback: the user's own words, one sentence per beat."""
    parts = [p.strip() for p in intent.replace("!", ".").replace("?", ".").split(".")]
    lines = [p for p in parts if p][:wanted] or [intent.strip()]
    return tuple(Beat(text=line[:TEXT_LIMIT], visual=line) for line in lines)


def target_seconds(beats: int) -> int:
    return round(beats * BEAT_SECONDS)
