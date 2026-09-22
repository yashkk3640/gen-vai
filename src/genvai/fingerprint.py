"""Content fingerprints, for incremental rendering.

A scene's fingerprint hashes everything that affects its pixels and samples. If it
matches a cached segment on disk, that segment is reused; otherwise the scene is
re-encoded. Changing scene 2 of a twelve-scene video re-encodes one segment.

This is what makes the conversational edit loop feel interactive rather than like a
fresh render per request.
"""

import hashlib
from pathlib import Path

from genvai.timeline import ImageOverlay, Scene, Timeline

RENDERER_VERSION = 1
"""Bump to invalidate every cached segment at once, after a filtergraph change."""


def hash_file(path: Path, *, chunk_size: int = 1 << 20) -> str:
    """SHA-256 of a file. Asset ids are derived from this, so identical content dedupes."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def scene_fingerprint(timeline: Timeline, scene: Scene) -> str:
    """Hash everything that determines how this scene renders.

    Covers the scene itself, the canvas, the content hashes of the assets it
    references, and any styles its overlays point at - a font size change must
    invalidate the segment even though the scene object did not change.

    Excludes music and the global audio mix, which are applied in a later pass, so an
    audio-only edit does not invalidate video segments.

    Narration is included, because with captions enabled it is burned into the picture.
    That makes a re-recorded line invalidate the segment; conservative, but a wrong
    reuse is far worse than a redundant encode.
    """
    return _stable_digest(
        str(RENDERER_VERSION),
        scene.model_dump_json(),
        timeline.canvas.model_dump_json(),
        timeline.captions.model_dump_json(),
        *_asset_digests(timeline, scene),
        *_style_digests(timeline, scene),
    )


def audio_fingerprint(timeline: Timeline) -> str:
    """Hash the audio mix - narration assets, music asset, gains, ducking.

    Lets the final mix pass be skipped when only the video changed.
    """
    narration = [s.narration.model_dump_json() for s in timeline.scenes if s.narration is not None]
    music_asset = timeline.assets.get(timeline.music.asset_id or "")
    return _stable_digest(
        str(RENDERER_VERSION),
        timeline.music.model_dump_json(),
        music_asset.sha256 if music_asset else "no-music",
        *narration,
    )


def timeline_fingerprint(timeline: Timeline) -> str:
    """Hash the whole timeline, for naming a final render."""
    return _stable_digest(
        str(RENDERER_VERSION),
        *(scene_fingerprint(timeline, s) for s in timeline.scenes),
        audio_fingerprint(timeline),
        timeline.export.model_dump_json(),
    )


def _asset_digests(timeline: Timeline, scene: Scene) -> tuple[str, ...]:
    """Content hashes of every asset the scene draws from.

    An asset id is already a content hash, but resolving through `timeline.assets`
    means a re-pointed id invalidates the segment even if the id string is reused.
    """
    ids: list[str] = []
    # Clip, photo and resolved generated visuals all carry an asset_id; cards and
    # colours carry none.
    visual_asset = getattr(scene.visual, "asset_id", None)
    if visual_asset:
        ids.append(visual_asset)
    ids.extend(o.asset_id for o in scene.overlays if isinstance(o, ImageOverlay))

    resolved = []
    for asset_id in ids:
        asset = timeline.assets.get(asset_id)
        resolved.append(asset.sha256 if asset else f"unresolved:{asset_id}")
    return tuple(resolved)


def _style_digests(timeline: Timeline, scene: Scene) -> tuple[str, ...]:
    """Styles this scene's text depends on, so a font change invalidates the cache."""
    refs = {o.style_ref for o in scene.overlays if hasattr(o, "style_ref")}
    if timeline.captions.enabled:
        refs.add(timeline.captions.style_ref)
    return tuple(
        timeline.styles[ref].model_dump_json() if ref in timeline.styles else f"missing:{ref}"
        for ref in sorted(refs)
    )


def _stable_digest(*parts: str) -> str:
    """Join parts with a separator that cannot occur in a hex digest, then hash.

    The separator matters: concatenating hashes without one lets different inputs
    collide into the same digest.
    """
    return hashlib.sha256("\x1f".join(parts).encode("utf-8")).hexdigest()
