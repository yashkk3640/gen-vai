"""Fitting shots to the canvas by cropping where the subject is.

Runs after the cut is decided, because it needs to know both which stretch of each clip
survived and what shape the canvas is. Nothing here changes the edit - only how each
shot is framed within it.

Everything is measurement and arithmetic: ffmpeg decodes a few frames, numpy finds the
detail, and the result is a crop rectangle on the scene. No model.
"""

from collections.abc import Callable
from pathlib import Path

from genvai.adapters.analyzer import sample_span
from genvai.analysis import to_greyscale
from genvai.reframe import crop_for, focus, needs_reframing, solve_path
from genvai.timeline import AssetVisual, ClipVisual, Scene, Timeline

SAMPLES_PER_CLIP = 8
"""Frames examined per shot.

Enough to notice a subject crossing the frame, few enough that a twenty-shot reel costs
a couple of seconds.
"""


def reframe(
    timeline: Timeline,
    resolve_asset: Callable[[str], Path],
    ffmpeg: Path,
    *,
    aspects: dict[str, float] | None = None,
) -> Timeline:
    """Give every shot a crop that keeps its subject in frame.

    Shots already the right shape are left alone - `needs_reframing` is checked first, so
    a vertical clip in a vertical reel costs nothing at all.

    A shot whose frames cannot be decoded keeps whatever crop it had. Reframing improves
    a cut; it should never be the reason one fails to render.
    """
    target = timeline.canvas.aspect
    return timeline.model_copy(
        update={
            "scenes": tuple(
                _reframed(scene, timeline, resolve_asset, ffmpeg, target, aspects or {})
                for scene in timeline.scenes
            )
        }
    )


def _reframed(
    scene: Scene,
    timeline: Timeline,
    resolve_asset: Callable[[str], Path],
    ffmpeg: Path,
    target: float,
    aspects: dict[str, float],
) -> Scene:
    visual = scene.visual
    if not isinstance(visual, ClipVisual | AssetVisual) or visual.crop is not None:
        return scene

    source = aspects.get(visual.asset_id)
    if source is None or not needs_reframing(source, target):
        return scene

    try:
        path = resolve_asset(visual.asset_id)
    except Exception:  # noqa: BLE001 - a missing file is the renderer's problem to report
        return scene

    if isinstance(visual, AssetVisual):
        frames = [to_greyscale(f) for f in sample_span(path, ffmpeg, start=0.0, end=0.1, count=1)]
        if not frames:
            return scene
        return scene.model_copy(
            update={
                "visual": visual.model_copy(
                    update={"crop": crop_for(focus(frames[0]), source, target)}
                )
            }
        )

    frames = [
        to_greyscale(f)
        for f in sample_span(
            path,
            ffmpeg,
            start=visual.source_start,
            end=visual.source_end,
            count=SAMPLES_PER_CLIP,
        )
    ]
    if not frames:
        return scene

    start, end = solve_path(frames, source, target)
    return scene.model_copy(
        update={"visual": visual.model_copy(update={"crop": start, "crop_end": end})}
    )


def source_aspects(timeline: Timeline, library) -> dict[str, float]:
    """Each asset's shape, taken from what ingest measured.

    Read from the media library rather than probed again: the dimensions are already
    known, and a clip shot in portrait has already had its rotation accounted for.
    """
    aspects: dict[str, float] = {}
    for asset_id in timeline.assets:
        item = library.item(asset_id)
        if item and item.height:
            aspects[asset_id] = item.width / item.height
    return aspects
