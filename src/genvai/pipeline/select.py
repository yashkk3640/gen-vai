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
from collections.abc import Callable

from pydantic import Field

from genvai.errors import GenvaiError
from genvai.media import MediaItem, MediaLibrary
from genvai.ports import LLMPort, ProjectStore
from genvai.timeline import (
    Asset,
    AssetVisual,
    Canvas,
    ClipVisual,
    Frozen,
    KenBurns,
    Rect,
    Scene,
    SceneRole,
    StillMotion,
    TextOverlay,
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
    captions: dict[str, str] | None = None,
    hook_asset_id: str | None = None,
) -> Timeline:
    """Turn an ordered set of shots into a renderable timeline.

    Mechanical by design - no model involved. With `captions` and `hook_asset_id` it
    applies the model's judgement; without them it is the chronological fallback that
    keeps the tool working when Ollama is not running.
    """
    scenes = tuple(
        _scene(item, index, len(items), seed, captions or {}, hook_asset_id)
        for index, item in enumerate(items)
    )
    return Timeline(
        intent=intent,
        canvas=canvas or Canvas(),
        seed=seed,
        scenes=scenes,
        assets=assets,
    )


def _scene(
    item: MediaItem,
    index: int,
    total: int,
    seed: int,
    captions: dict[str, str],
    hook_asset_id: str | None,
) -> Scene:
    role = _role(item, index, total, hook_asset_id)
    duration = contribution(item)
    overlays: tuple[TextOverlay, ...] = ()
    caption = (captions.get(item.asset_id) or "").strip()
    if caption:
        overlays = (TextOverlay(content=caption),)

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
            overlays=overlays,
            note=span.reason if span else None,
        )

    start_rect, end_rect = _KEN_BURNS_MOVES[(index + seed) % len(_KEN_BURNS_MOVES)]
    return Scene(
        id=f"s{index + 1}",
        duration=duration,
        role=role,
        visual=AssetVisual(asset_id=item.asset_id, fit="cover"),
        motion=KenBurns(start_rect=start_rect, end_rect=end_rect),
        overlays=overlays,
    )


def _role(item: MediaItem, index: int, total: int, hook_asset_id: str | None) -> SceneRole:
    """Hook first, payoff last, body between - unless the model nominated a hook."""
    if hook_asset_id and item.asset_id == hook_asset_id:
        return "hook"
    if hook_asset_id:
        return "payoff" if index == total - 1 else "body"
    return "hook" if index == 0 else "payoff" if index == total - 1 else "body"


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
    on_note: Callable[[str], None] | None = None,
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

    captions: dict[str, str] = {}
    hook: str | None = None
    if llm is not None:
        try:
            picked, captions, hook = order_with_llm(
                picked, llm, intent=intent, target_duration=target_duration
            )
        except GenvaiError as exc:
            # A model that is down or unhelpful must not cost the user their reel.
            # Chronological order with measured trims is a decent edit on its own.
            if on_note:
                on_note(f"{exc} Falling back to chronological order.")

    assets = {item.asset_id: store.load_asset(project_id, item.asset_id) for item in picked}
    timeline = build_timeline(
        picked,
        assets,
        intent=intent,
        canvas=canvas,
        seed=seed,
        captions=captions,
        hook_asset_id=hook,
    )

    versions = store.versions(project_id)
    timeline = timeline.model_copy(update={"version": (max(versions) + 1) if versions else 1})
    store.save_timeline(project_id, timeline)
    return timeline


# ---------------------------------------------------------------- model ordering


class ShotChoice(Frozen):
    """One shot's place in the cut, as the model returns it."""

    asset_id: str
    caption: str = Field(default="", description="On-screen text, or empty for none.")


class ReelPlan(Frozen):
    """What the model is asked for. Deliberately small.

    Schema size is the dominant factor in small-model reliability: asking a 3B model for
    order, duration, role *and* caption produced two hooks and a nonsense rationale,
    while asking only for order, caption and one hook id was reliable. Durations are not
    here because they were already measured - a model should not be asked to invent a
    number that is known.
    """

    order: tuple[ShotChoice, ...]
    hook_asset_id: str = Field(default="", description="Which shot opens the reel.")


SYSTEM_PROMPT = (
    "You are a short-form video editor. You arrange clips and photos into a reel. "
    "Reply with JSON only."
)

CAPTION_LIMIT = 40


def describe(items: tuple[MediaItem, ...]) -> str:
    """A compact description of the shortlist, one line per shot.

    The model never sees pixels. It sees what each shot is, how long, when it was taken
    and what the measurement said about it - enough to order them, and small enough that
    twenty shots still fit comfortably in a 3B model's attention.
    """
    lines = []
    for item in items:
        parts = [item.asset_id[:12], "clip" if item.kind == "video" else "photo"]
        span = item.best_span
        if span is not None:
            parts.append(f"{span.duration:.1f}s ({span.reason})")
        if item.captured_at:
            parts.append(f"taken {item.captured_at[:16]}")
        if item.tags:
            parts.append("shows " + ", ".join(item.tags[:4]))
        lines.append("  " + "  ".join(parts))
    return "\n".join(lines)


def reconcile(
    plan: ReelPlan, items: tuple[MediaItem, ...]
) -> tuple[tuple[MediaItem, ...], dict[str, str], str | None]:
    """Turn a model's answer into something safe to build from.

    Pure, and deliberately forgiving in one direction only. Ids the model invented are
    dropped; ids it forgot are appended in capture order. A dropped shot is a smaller
    reel, which is recoverable, while an invented id would fail the render outright.

    Ids are matched on the prefix the model was shown, because it is given a truncated
    id to keep the prompt short and will echo it back that way.
    """
    by_id = {item.asset_id: item for item in items}
    ordered: list[MediaItem] = []
    captions: dict[str, str] = {}

    for choice in plan.order:
        item = _match(choice.asset_id, by_id)
        if item is None or item in ordered:
            continue
        ordered.append(item)
        caption = choice.caption.strip()
        if caption:
            captions[item.asset_id] = caption[:CAPTION_LIMIT]

    missing = [i for i in items if i not in ordered]
    ordered.extend(sorted(missing, key=_chronological))

    hook = _match(plan.hook_asset_id, by_id)
    return tuple(ordered), captions, hook.asset_id if hook else None


def _match(candidate: str, by_id: dict[str, MediaItem]) -> MediaItem | None:
    cleaned = candidate.strip()
    if not cleaned:
        return None
    if cleaned in by_id:
        return by_id[cleaned]
    hits = [item for key, item in by_id.items() if key.startswith(cleaned)]
    return hits[0] if len(hits) == 1 else None


def order_with_llm(
    items: tuple[MediaItem, ...], llm: LLMPort, *, intent: str, target_duration: float
) -> tuple[tuple[MediaItem, ...], dict[str, str], str | None]:
    """Ask the model to order the shortlist and caption it.

    Raises whatever the LLM port raises; callers decide whether to fall back.
    """
    prompt = (
        f"Arrange these {len(items)} shots into a {target_duration:.0f} second reel.\n"
        f"What the user asked for: {intent or 'a reel from these shots'}\n\n"
        f"{describe(items)}\n\n"
        "Use every id exactly once, in the order they should appear. "
        "Choose the single most striking shot as the hook, to open on. "
        f"Give each shot a caption of at most {CAPTION_LIMIT} characters, "
        "or an empty string where words would add nothing."
    )
    return reconcile(llm.structured(prompt, ReelPlan, system=SYSTEM_PROMPT), items)
