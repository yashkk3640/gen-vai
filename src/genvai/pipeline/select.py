"""Analysed library + intent -> a timeline.

The heart of camera-roll editing, and a deliberate split of labour:

- **Measurement** already happened at ingest and is objective - sharp, steady, exposed.
- **Shortlisting** is a scored filter: drop the unusable, keep one per duplicate group,
  prefer strong spans. Pure, deterministic, and needs no model.
- **Judgement** is the LLM's: ordering into something with a shape, deciding the opening
  shot, writing on-screen text, matching what the user asked for.

Keeping those apart means the edit is reproducible and the model is only asked the
question it is actually good at.
"""

import re

from genvai.errors import GenvaiError
from genvai.media import MediaItem, MediaLibrary
from genvai.ports import LLMPort, ProjectStore
from genvai.timeline import (
    Asset,
    AssetVisual,
    Canvas,
    ClipVisual,
    KenBurns,
    Rect,
    Scene,
    SceneRole,
    StillMotion,
    Timeline,
)

PHOTO_BEAT = 2.2
"""Seconds a still occupies. Matches the default scene duration - short-form pacing."""

MAX_BEAT = 4.0
"""Longest a single shot holds the screen before it stops feeling like a reel."""

MIN_BEAT = 0.8

OVERSHOOT = 1.6
"""How much more material to shortlist than the target needs.

Selection wants alternatives - a shot the ordering rejects, a second take to fall back
on - but handing a model a hundred clips wastes context and produces worse ordering than
handing it the best twenty.
"""

_WORD = re.compile(r"[a-z0-9]+")


def item_score(item: MediaItem, *, intent: str = "") -> float:
    """How much this shot deserves a place, 0-1.

    For a clip the answer is mostly its best span, which already folds in sharpness,
    exposure, steadiness and position within the clip. A still has no span, so it is
    scored from its own quality.

    `intent` only ever adds. A shot that matches what the user asked for is promoted,
    but one that does not is still eligible - the alternative is a request for "the food
    ones" silently producing a three-shot reel.
    """
    quality = item.quality
    if quality is None:
        return 0.0

    if item.kind == "video":
        span = item.best_span
        base = span.score if span else quality.sharpness * 0.5
    else:
        exposure_fit = max(0.0, 1.0 - abs(quality.exposure - 0.5) * 1.8)
        base = 0.65 * quality.sharpness + 0.35 * exposure_fit

    if not quality.usable:
        base *= 0.35
    return min(1.0, base + _intent_bonus(item, intent))


def contribution(item: MediaItem) -> float:
    """How many seconds this shot would add to the cut."""
    if item.kind == "image":
        return PHOTO_BEAT
    span = item.best_span
    if span is None:
        return min(MAX_BEAT, item.duration or PHOTO_BEAT)
    return max(MIN_BEAT, min(MAX_BEAT, span.duration))


def shortlist(
    library: MediaLibrary,
    *,
    target_duration: float,
    intent: str = "",
    max_items: int | None = None,
) -> tuple[MediaItem, ...]:
    """Score and filter down to a workable set, before any model sees anything.

    Pure and deterministic: the same library and target always yield the same shortlist.

    Returns items in capture order rather than score order. The scores decided *who* is
    in; chronology is the sane default for *when*, and it gives the ordering step a
    sensible starting point rather than a ranking it has to undo.
    """
    candidates = [i for i in library.deduplicated() if i.quality is not None]
    if not candidates:
        return ()

    ranked = sorted(
        candidates,
        key=lambda i: (-item_score(i, intent=intent), i.source_name),
    )

    budget = max(0.0, target_duration) * OVERSHOOT
    chosen: list[MediaItem] = []
    filled = 0.0
    for item in ranked:
        if max_items is not None and len(chosen) >= max_items:
            break
        if filled >= budget and chosen:
            break
        chosen.append(item)
        filled += contribution(item)

    return tuple(sorted(chosen, key=_chronological))


def _chronological(item: MediaItem) -> tuple[str, str]:
    """Capture time when the camera recorded one, filename otherwise.

    Falling back to the filename is not arbitrary: phone cameras number sequentially, so
    IMG_0041 really does come before IMG_0042. Items with no timestamp sort after those
    that have one, so a dated holiday is not interleaved with undated screenshots.
    """
    return (item.captured_at or "￿", item.source_name)


def _intent_bonus(item: MediaItem, intent: str) -> float:
    """Up to +0.25 for matching what the user asked for.

    Matches against tags and caption, which are empty unless a ContentTagger is
    installed - so without one this is uniformly zero and selection falls back to
    quality and chronology.
    """
    wanted = set(_WORD.findall(intent.lower()))
    if not wanted:
        return 0.0
    haystack = " ".join([*item.tags, item.caption or ""]).lower()
    if not haystack.strip():
        return 0.0
    hits = sum(1 for word in wanted if word in haystack)
    return min(0.25, hits * 0.1)


# ------------------------------------------------------------------------ assembly

_KEN_BURNS_MOVES: tuple[tuple[Rect, Rect], ...] = (
    ((0.00, 0.00, 1.00, 1.00), (0.10, 0.10, 0.80, 0.80)),  # push in
    ((0.10, 0.10, 0.80, 0.80), (0.00, 0.00, 1.00, 1.00)),  # pull out
    ((0.00, 0.05, 0.82, 0.90), (0.18, 0.05, 0.82, 0.90)),  # pan right
    ((0.18, 0.05, 0.82, 0.90), (0.00, 0.05, 0.82, 0.90)),  # pan left
    ((0.05, 0.00, 0.85, 0.85), (0.10, 0.15, 0.80, 0.80)),  # drift down
)
"""A rotation of camera moves for stills.

Variety is the point. One uniform push applied to every photo is what makes a reel read
as a slideshow; alternating the move is nearly free and breaks the pattern.
"""


def fit_to_duration(
    items: tuple[MediaItem, ...], target: float, *, intent: str = ""
) -> tuple[MediaItem, ...]:
    """Trim a shortlist down to what actually fits, keeping capture order.

    Drops the weakest shots rather than speeding everything up. A reel that runs short
    is fine; one where every shot is clipped to make the numbers work is not.
    """
    if target <= 0.0:
        return ()
    ranked = sorted(items, key=lambda i: (-item_score(i, intent=intent), i.source_name))
    kept: list[MediaItem] = []
    filled = 0.0
    for item in ranked:
        if filled >= target and kept:
            break
        kept.append(item)
        filled += contribution(item)
    return tuple(sorted(kept, key=_chronological))


def build_timeline(
    items: tuple[MediaItem, ...],
    assets: dict[str, Asset],
    *,
    intent: str,
    canvas: Canvas | None = None,
    seed: int = 0,
) -> Timeline:
    """Turn an ordered set of shots into a renderable timeline.

    Deliberately mechanical - no model involved. This is the fallback that keeps the
    tool working with Ollama down, and the structure the LLM step later rearranges
    rather than replaces.
    """
    scenes = tuple(_scene(item, index, len(items), seed) for index, item in enumerate(items))
    return Timeline(
        intent=intent,
        canvas=canvas or Canvas(),
        seed=seed,
        scenes=scenes,
        assets=assets,
    )


def _scene(item: MediaItem, index: int, total: int, seed: int) -> Scene:
    role: SceneRole = "hook" if index == 0 else "payoff" if index == total - 1 else "body"
    duration = contribution(item)

    if item.kind == "video":
        span = item.best_span
        start = span.start if span else 0.0
        end = span.end if span else min(duration, item.duration or duration)
        return Scene(
            id=f"s{index + 1}",
            duration=max(MIN_BEAT, end - start),
            role=role,
            visual=ClipVisual(asset_id=item.asset_id, source_start=start, source_end=end),
            motion=StillMotion(),
            note=span.reason if span else None,
        )

    start_rect, end_rect = _KEN_BURNS_MOVES[(index + seed) % len(_KEN_BURNS_MOVES)]
    return Scene(
        id=f"s{index + 1}",
        duration=duration,
        role=role,
        visual=AssetVisual(asset_id=item.asset_id, fit="cover"),
        motion=KenBurns(start_rect=start_rect, end_rect=end_rect),
    )


def fit_to_beats(timeline: Timeline) -> Timeline:
    """Nudge scene boundaries onto the music's beat grid.

    A no-op until a track has been chosen and analysed, which happens in the music
    milestone. Returning the timeline unchanged rather than raising means the render
    path works the same way with or without music.
    """
    beats = timeline.music.beat_map
    if beats is None or not timeline.export.snap_cuts_to_beat or not beats.beats:
        return timeline
    return timeline


def make_reel(
    project_id: str,
    store: ProjectStore,
    *,
    intent: str = "",
    target_duration: float = 30.0,
    canvas: Canvas | None = None,
    seed: int = 0,
    llm: LLMPort | None = None,
) -> Timeline:
    """Library -> a saved, renderable timeline.

    Shortlists, trims to the target, and builds. With an `llm` the ordering and on-screen
    text come from the model; without one it falls back to chronological order with
    best-span trims, which is a decent edit in its own right and keeps the tool usable
    when Ollama is not running.

    Saves as the project's next version, so every reel is reversible.
    """
    library = store.load_media(project_id)
    if not library.items:
        raise GenvaiError(
            f"project '{project_id}' has no media. "
            f"Import some first: genvai add {project_id} <files...>"
        )

    picked = fit_to_duration(
        shortlist(library, target_duration=target_duration, intent=intent),
        target_duration,
        intent=intent,
    )
    if not picked:
        raise GenvaiError("nothing usable to build a reel from")

    assets = {item.asset_id: store.load_asset(project_id, item.asset_id) for item in picked}
    timeline = build_timeline(picked, assets, intent=intent, canvas=canvas, seed=seed)

    versions = store.versions(project_id)
    timeline = timeline.model_copy(update={"version": (max(versions) + 1) if versions else 1})
    store.save_timeline(project_id, timeline)
    return timeline
