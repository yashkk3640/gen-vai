"""Filling in what a plan only described: generating the images a storyboard asks for.

Runs after the LLM is unloaded, because at 4 GB a text model and an image model cannot
share the card. Every provider may be unavailable, and the resolver degrades rather than
aborting - a video made of designed backdrops is a video; an exception is not.
"""

from genvai.ports import ImageProvider, MusicProvider, ProjectStore, SpeechProvider
from genvai.timeline import AssetProvenance, GeneratedVisual, Scene, Timeline


def resolve_visuals(
    timeline: Timeline,
    project_id: str,
    images: ImageProvider,
    store: ProjectStore,
) -> Timeline:
    """Generate an image for every unresolved scene and attach it as an asset.

    Approved visuals are skipped even if their prompt changed - a visual the user
    blessed is never silently replaced.

    Each image is stored by content hash, so two scenes asking for the same thing at the
    same seed share one file, and re-resolving an unchanged timeline costs nothing.
    """
    if not timeline.unresolved_visuals:
        return timeline

    scenes: list[Scene] = []
    assets = dict(timeline.assets)

    for scene in timeline.scenes:
        visual = scene.visual
        if not isinstance(visual, GeneratedVisual) or visual.asset_id is not None:
            scenes.append(scene)
            continue

        seed = visual.seed if visual.seed is not None else timeline.seed
        path = images.generate(
            visual.prompt,
            width=timeline.canvas.width,
            height=timeline.canvas.height,
            seed=seed,
            negative_prompt=visual.negative_prompt,
        )
        asset = store.store_asset(
            project_id,
            path,
            "image",
            AssetProvenance(provider=images.name, seed=seed, prompt=visual.prompt),
        )
        assets[asset.sha256] = asset
        scenes.append(
            scene.model_copy(
                update={"visual": visual.model_copy(update={"asset_id": asset.sha256})}
            )
        )

    return timeline.model_copy(update={"scenes": tuple(scenes), "assets": assets})


def resolve_narration(
    timeline: Timeline,
    project_id: str,
    speech: SpeechProvider,
    store: ProjectStore,
) -> Timeline:
    """Synthesise each narration line and reconcile scene durations against it.

    Scenes stretch to fit their audio rather than truncating speech - the least
    destructive option, and the one that keeps a sentence from being cut off mid-word.
    """
    raise NotImplementedError


def suggest_music(timeline: Timeline, music: MusicProvider) -> Timeline:
    """Search for candidates and move the state to `candidates_ready`.

    Superseded by `genvai.pipeline.music.suggest`, which also persists. Kept as the port
    boundary it always was.
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
    """Superseded by `genvai.pipeline.music.approve`."""
    raise NotImplementedError
