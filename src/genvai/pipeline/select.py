"""Analysed library + intent -> a timeline.

The heart of camera-roll editing, and a deliberate split of labour:

- **Measurement** already happened at ingest and is objective - sharp, steady, exposed.
- **Selection** is a scored shortlist: drop the unusable, keep one per duplicate group,
  prefer good spans. Deterministic, explainable, and needs no model.
- **Judgement** is the LLM's: ordering into something with a shape, deciding what the
  opening shot should be, writing on-screen text, matching the user's intent.

Keeping those apart means the edit is reproducible and the model is only asked the
question it is actually good at.
"""

from genvai.media import MediaItem, MediaLibrary
from genvai.ports import LLMPort
from genvai.timeline import Canvas, Timeline


def shortlist(
    library: MediaLibrary,
    *,
    target_duration: float,
    intent: str = "",
    max_items: int | None = None,
) -> tuple[MediaItem, ...]:
    """Score and filter down to a workable set, before the LLM sees anything.

    Drops unusable shots, keeps the best of each duplicate group, and prefers items
    with strong spans. Handing a model a hundred clips wastes context and produces
    worse ordering than handing it the best twenty.

    Pure and deterministic: the same library and target always yield the same shortlist.
    """
    raise NotImplementedError


def assemble(
    library: MediaLibrary,
    llm: LLMPort,
    *,
    intent: str = "",
    target_duration: float = 30.0,
    canvas: Canvas | None = None,
    seed: int = 0,
) -> Timeline:
    """Order the shortlist into a timeline with trims, text and pacing.

    The LLM receives the shortlist as a compact description - what each item is, how
    long, when it was taken, its best span, its tags - and returns ordering, per-scene
    spans, roles and on-screen text. It never sees pixels and never invents an asset id.

    Falls back to chronological order with best-span trims if the model is unavailable,
    which is a decent edit on its own and keeps the tool useful without Ollama running.
    """
    raise NotImplementedError


def fit_to_beats(timeline: Timeline) -> Timeline:
    """Nudge scene boundaries onto the music's beat grid.

    A no-op when the music has no beat map or `export.snap_cuts_to_beat` is off.
    Adjusts durations by a few frames each; total runtime is preserved within a beat.
    """
    raise NotImplementedError
