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

from genvai.media import MediaItem, MediaLibrary

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
