"""RendererPort over ffmpeg. The only module that builds a filtergraph.

The binary is resolved PATH-first, falling back to the copy bundled by
`imageio-ffmpeg`, so a clean clone renders with no system install.

Ken Burns motion is produced with `zoompan`, transitions with `xfade`, and scenes are
joined with the concat demuxer so unchanged segments are copied rather than re-encoded.
"""

from pathlib import Path

from genvai.config import RenderSettings
from genvai.ports import MediaInfo
from genvai.timeline import Timeline


def resolve_ffmpeg() -> Path:
    """Locate ffmpeg: a system binary on PATH if present, else the bundled one.

    PATH wins because a full system build supports codecs the bundled one does not,
    and anyone who needs those will have installed it deliberately.
    """
    raise NotImplementedError


class FFmpegRenderer:
    """Implements `RendererPort`."""

    def __init__(self, settings: RenderSettings, binary: Path | None = None) -> None:
        self._settings = settings
        self._binary = binary

    def render_scene(self, timeline: Timeline, scene_id: str, out: Path) -> Path:
        """Render one scene to a segment: visual, motion, overlays, captions."""
        raise NotImplementedError

    def concat(self, segments: tuple[Path, ...], out: Path) -> Path:
        """Join segments, applying each scene's incoming transition."""
        raise NotImplementedError

    def mix_audio(self, video: Path, timeline: Timeline, out: Path) -> Path:
        """Lay narration and music over the cut, sidechain-ducking music under speech."""
        raise NotImplementedError

    def probe(self, media: Path) -> MediaInfo:
        raise NotImplementedError
