"""Intent -> Timeline.

The planner emits a complete storyboard in a single pass. It cannot look at a
generated image and reconsider, because the LLM is unloaded before the diffusion model
loads - see 'Phase-ordered model loading' in docs/06-decisions.md.
"""

from pathlib import Path

from genvai.ports import LLMPort
from genvai.timeline import Canvas, Timeline


def plan(
    intent: str,
    llm: LLMPort,
    *,
    canvas: Canvas | None = None,
    assets: tuple[Path, ...] = (),
    seed: int = 0,
) -> Timeline:
    """Turn a free-text intent into a schema-valid timeline.

    With `assets`, plans around the user's images (photo mode) instead of writing prompts
    for new ones (idea mode).

    Raises `PlanningError` if the model cannot produce valid output within its retries.
    """
    raise NotImplementedError


def storyboard_schema_for_llm() -> dict[str, object]:
    """The JSON schema handed to the model for constrained decoding.

    A reduced view of `Timeline`: fields the model must never write - asset_id, sha256,
    path, provenance, version, assets - are excluded outright rather than filtered after
    the fact. See docs/03-timeline-schema.md.
    """
    raise NotImplementedError
