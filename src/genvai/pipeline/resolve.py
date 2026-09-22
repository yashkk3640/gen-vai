"""Fill unresolved asset slots: generate images, synthesise narration, settle music.

Runs after the LLM is unloaded. Every provider may be unavailable; the resolver
degrades to a fallback and reports it rather than aborting the run.
"""

from genvai.ports import ImageProvider, MusicProvider, ProjectStore, SpeechProvider
from genvai.timeline import Timeline


def resolve_visuals(
    timeline: Timeline,
    project_id: str,
    images: ImageProvider,
    store: ProjectStore,
) -> Timeline:
    """Generate one image per unresolved scene and attach the asset.

    Approved visuals are skipped, even if their prompt changed (requirement F3.4).
    Falls back to a procedural card when `images.is_available()` is False.
    """
    raise NotImplementedError


def resolve_narration(
    timeline: Timeline,
    project_id: str,
    speech: SpeechProvider,
    store: ProjectStore,
) -> Timeline:
    """Synthesise each narration line and reconcile scene durations against it.

    Scenes stretch to fit their audio rather than truncating speech - the least
    destructive of the options, though see the open question in docs/04-roadmap.md.
    """
    raise NotImplementedError


def suggest_music(timeline: Timeline, music: MusicProvider) -> Timeline:
    """Search for candidates and move the state to `candidates_ready`.

    Read-only: this never downloads. Presenting the options and taking the user's
    choice is the caller's job.
    """
    raise NotImplementedError


def fetch_approved_music(
    timeline: Timeline,
    project_id: str,
    music: MusicProvider,
    store: ProjectStore,
    *,
    confirmed: bool,
) -> Timeline:
    """Download the selected track and record its licence.

    Requires `state == "approved"` and `confirmed is True`; raises
    `ConfirmationRequired` otherwise. See docs/06-decisions.md D4.
    """
    raise NotImplementedError
