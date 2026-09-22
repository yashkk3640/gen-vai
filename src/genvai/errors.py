"""Typed error hierarchy.

Every failure the user can plausibly hit has a distinct type, so the CLI can render
an actionable message instead of a traceback.
"""


class GenvaiError(Exception):
    """Base for every error raised by this package."""


class ConfigError(GenvaiError):
    """Configuration is missing or contradictory."""


class ProviderUnavailable(GenvaiError):
    """A backend is not installed or not reachable.

    Carries the fallback that will be used, when one exists, so the caller can
    report the degradation rather than failing.
    """

    def __init__(self, provider: str, reason: str, fallback: str | None = None) -> None:
        self.provider = provider
        self.reason = reason
        self.fallback = fallback
        detail = f" Falling back to {fallback}." if fallback else ""
        super().__init__(f"{provider} unavailable: {reason}.{detail}")


class PlanningError(GenvaiError):
    """The LLM failed to produce a schema-valid plan within the retry budget."""


class InvalidEditOp(GenvaiError):
    """An edit operation is not applicable to the current timeline.

    Raised before anything is applied; op lists are all-or-nothing.
    """

    def __init__(self, op: str, reason: str) -> None:
        self.op = op
        self.reason = reason
        super().__init__(f"cannot apply '{op}': {reason}")


class RenderError(GenvaiError):
    """ffmpeg failed. Carries the command and stderr tail for diagnosis."""

    def __init__(self, message: str, command: list[str] | None = None, stderr: str = "") -> None:
        self.command = command or []
        self.stderr = stderr
        super().__init__(message)


class ProjectNotFound(GenvaiError):
    """No project exists with the given id."""


class ConfirmationRequired(GenvaiError):
    """An action needing explicit user consent was attempted without it.

    Currently only music downloads. This is a guard against a code path
    reaching the network implicitly; see 'Music consent' in docs/decisions.md.
    """
