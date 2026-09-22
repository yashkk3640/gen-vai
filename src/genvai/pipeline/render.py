"""Timeline -> MP4, incrementally.

Scenes are rendered to individually cached segments keyed by fingerprint, concatenated,
then given an audio pass. An edit that touches one scene re-encodes one segment; an
edit that touches only music skips video entirely.
"""

from pathlib import Path

from genvai.ports import ProjectStore, RendererPort
from genvai.timeline import AudioVariant, Timeline


def render(
    timeline: Timeline,
    project_id: str,
    renderer: RendererPort,
    store: ProjectStore,
    *,
    preview: bool = False,
) -> dict[AudioVariant, Path]:
    """Render the timeline, reusing cached segments where fingerprints match.

    Returns one file per variant in `timeline.export.audio_variants`. They share a
    single video pass, so the extra cuts cost only an audio mux each.

    `preview` produces a fast low-resolution proxy for iterating on an edit.

    Raises `RenderError` if ffmpeg fails, and a `GenvaiError` if the music state is not
    renderable - that means the pipeline reached the renderer before asking the user.
    """
    raise NotImplementedError


def plan_render(timeline: Timeline, project_id: str, store: ProjectStore) -> "RenderPlan":
    """Work out what actually needs re-encoding, without encoding anything.

    Lets the CLI tell the user "3 of 12 scenes will re-render" before committing.
    """
    raise NotImplementedError


class RenderPlan:
    """Which segments are cached and which must be rebuilt."""

    cached: tuple[str, ...]
    rebuild: tuple[str, ...]
    remix_audio: bool
