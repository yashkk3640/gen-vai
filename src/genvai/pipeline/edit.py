"""The conversational edit loop: a change request against an existing project.

The LLM returns operations, not a timeline. They are validated as a set, applied
atomically, shown as a diff, and only then re-rendered. Every version is kept, so any
edit is reversible.
"""

from genvai.ops import EditPlan
from genvai.ports import LLMPort
from genvai.timeline import Timeline


def interpret(request: str, timeline: Timeline, llm: LLMPort) -> EditPlan:
    """Turn "scene 2 is too fast" into a validated op list.

    The model receives a summarised timeline - scene ids, durations, prompts, narration -
    rather than the full document, so context stays small and it cannot reference
    fields it is not allowed to write.

    Raises `PlanningError` if no valid plan is produced within the retry budget.
    """
    raise NotImplementedError


def apply_plan(timeline: Timeline, plan: EditPlan) -> Timeline:
    """Validate the whole op list, then apply it, returning the next version.

    Raises `InvalidEditOp` before touching anything if any op is inapplicable.
    """
    raise NotImplementedError


def diff(before: Timeline, after: Timeline) -> tuple[str, ...]:
    """Human-readable summary of what changed, shown before re-rendering."""
    raise NotImplementedError
