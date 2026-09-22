"""Content fingerprints, for incremental rendering.

A scene's fingerprint hashes everything that affects its pixels and samples. If it
matches a cached segment on disk, that segment is reused; otherwise the scene is
re-encoded. Changing scene 2 of a twelve-scene video re-encodes one segment.

This is what makes the conversational edit loop feel interactive rather than like a
fresh render per request.
"""

import hashlib
from pathlib import Path

from genvai.timeline import Scene, Timeline


def hash_file(path: Path, *, chunk_size: int = 1 << 20) -> str:
    """SHA-256 of a file. Asset ids are derived from this, so identical content dedupes."""
    raise NotImplementedError


def scene_fingerprint(timeline: Timeline, scene: Scene) -> str:
    """Hash everything that determines how this scene renders.

    Includes the scene itself, the canvas, the referenced assets' hashes, and any
    styles the scene's overlays point at - a font size change must invalidate the
    segment even though the scene object did not change.

    Excludes music and the global narration mix: those are applied in a later pass,
    so an audio-only edit must not invalidate video segments.
    """
    raise NotImplementedError


def audio_fingerprint(timeline: Timeline) -> str:
    """Hash the audio mix - narration assets, music asset, gains, ducking.

    Lets the final mix pass be skipped when only the video changed.
    """
    raise NotImplementedError


def timeline_fingerprint(timeline: Timeline) -> str:
    """Hash the whole timeline, for naming a final render."""
    raise NotImplementedError


def _stable_digest(*parts: str) -> str:
    """Join parts with a separator that cannot occur in a hex digest, then hash.

    The separator matters: concatenating hashes without one lets different inputs
    collide into the same digest.
    """
    joined = "\x1f".join(parts)
    return hashlib.sha256(joined.encode("utf-8")).hexdigest()
