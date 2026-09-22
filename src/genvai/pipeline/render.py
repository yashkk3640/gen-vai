"""Timeline -> MP4, incrementally.

Scenes are rendered to individually cached segments keyed by fingerprint, concatenated,
then given an audio pass. An edit that touches one scene re-encodes one segment; an
edit that touches only music skips video entirely.
"""

from pathlib import Path

from genvai.errors import GenvaiError
from genvai.fingerprint import audio_fingerprint, scene_fingerprint, timeline_fingerprint
from genvai.ports import ProjectStore, RendererPort
from genvai.timeline import AudioVariant, Canvas, Frozen, Timeline


class RenderPlan(Frozen):
    """Which segments are cached and which must be rebuilt.

    Produced without encoding anything, so the CLI can say "3 of 12 scenes will
    re-render" before committing to the work.
    """

    cached: tuple[str, ...] = ()
    rebuild: tuple[str, ...] = ()
    remix_audio: bool = True

    @property
    def total(self) -> int:
        return len(self.cached) + len(self.rebuild)

    def summary(self) -> str:
        if not self.rebuild:
            return f"all {self.total} scenes cached; audio only"
        return f"{len(self.rebuild)} of {self.total} scenes to render"


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

    `preview` renders at reduced height for fast iteration. Preview segments cache
    separately, because the canvas is part of a scene's fingerprint.

    Raises `RenderError` if ffmpeg fails, and a `GenvaiError` if the music state is not
    renderable - that means the pipeline reached the renderer before asking the user.
    """
    if not timeline.scenes:
        raise GenvaiError("nothing to render: the timeline has no scenes")
    if not timeline.music.is_renderable:
        raise GenvaiError(
            f"music is in state '{timeline.music.state}': the user has not chosen a "
            "track yet. Resolve or decline it before rendering."
        )

    effective = _preview_of(timeline) if preview else timeline
    segments = _segments(effective, project_id, renderer, store)
    transitions = tuple(
        (s.transition_in.kind, s.transition_in.duration) for s in effective.scenes[1:]
    )

    renders = store.renders_dir(project_id)
    stem = f"v{effective.version}-{'preview' if preview else timeline_fingerprint(effective)[:10]}"
    joined = renders / f".{stem}-silent-cut.mp4"
    renderer.concat_with(segments, transitions, joined)

    outputs: dict[AudioVariant, Path] = {}
    for variant in effective.export.audio_variants:
        suffix = "" if variant == "full" else f"-{variant}"
        outputs[variant] = renderer.mix_audio(
            joined, effective, renders / f"{stem}{suffix}.mp4", variant
        )
    joined.unlink(missing_ok=True)
    return outputs


def plan_render(timeline: Timeline, project_id: str, store: ProjectStore) -> RenderPlan:
    """Work out what actually needs re-encoding, without encoding anything."""
    cache = store.segment_cache_dir(project_id)
    cached, rebuild = [], []
    for scene in timeline.scenes:
        target = cached if _segment_path(cache, timeline, scene).exists() else rebuild
        target.append(scene.id)

    mix = store.renders_dir(project_id) / f".audio-{audio_fingerprint(timeline)[:16]}"
    return RenderPlan(cached=tuple(cached), rebuild=tuple(rebuild), remix_audio=not mix.exists())


def _segments(
    timeline: Timeline, project_id: str, renderer: RendererPort, store: ProjectStore
) -> tuple[Path, ...]:
    """Render what is missing, reuse what is not. The whole point of fingerprinting."""
    cache = store.segment_cache_dir(project_id)
    paths: list[Path] = []
    for scene in timeline.scenes:
        path = _segment_path(cache, timeline, scene)
        if not path.exists():
            renderer.render_scene(timeline, scene.id, path)
        paths.append(path)
    return tuple(paths)


def _segment_path(cache: Path, timeline: Timeline, scene: object) -> Path:
    return cache / f"{scene_fingerprint(timeline, scene)[:32]}.mp4"  # type: ignore[arg-type]


def _preview_of(timeline: Timeline, height: int = 480) -> Timeline:
    """A reduced-resolution copy, keeping aspect and even dimensions.

    Everything else is untouched, so a preview and a final render differ only in scale
    and the preview is an honest look at the real edit.
    """
    canvas = timeline.canvas
    if canvas.height <= height:
        return timeline
    scaled_width = max(16, round(canvas.width * height / canvas.height))
    return timeline.model_copy(
        update={
            "canvas": canvas.model_copy(
                update={"width": scaled_width - scaled_width % 2, "height": height - height % 2}
            )
        }
    )


def preview_canvas(canvas: Canvas, height: int = 480) -> Canvas:
    """Exposed for callers that want to show what a preview will look like."""
    return _preview_of(Timeline(intent="", canvas=canvas), height).canvas
