"""End-to-end render tests. These actually encode video with the bundled ffmpeg.

Slower than the rest of the suite by design: the filtergraph is the part most likely to
break silently, and a test that only checks a command string would not have caught any
of the bugs these did.
"""

from pathlib import Path

import pytest

from genvai.adapters.ffmpeg import FFmpegRenderer
from genvai.adapters.fs_store import FilesystemStore
from genvai.config import RenderSettings
from genvai.errors import GenvaiError
from genvai.pipeline.render import plan_render, render
from genvai.timeline import (
    AssetProvenance,
    AssetVisual,
    Canvas,
    ClipVisual,
    Export,
    KenBurns,
    Music,
    Scene,
    TextOverlay,
    Timeline,
    Transition,
)

FAST = RenderSettings(preset="ultrafast", crf=30)


@pytest.fixture
def project(tmp_path: Path, sample_media: dict[str, Path]) -> tuple[FilesystemStore, str, dict]:
    """A project with the sample media imported, plus a map of kind -> asset."""
    store = FilesystemStore(tmp_path / "projects")
    store.create("test reel", "t")
    assets = {}
    for key, kind in (("photo", "image"), ("clip_a", "video"), ("clip_b", "video")):
        asset = store.store_asset("t", sample_media[key], kind, AssetProvenance(provider="test"))
        assets[key] = asset
    return store, "t", assets


def _renderer(store: FilesystemStore, project_id: str) -> FFmpegRenderer:
    return FFmpegRenderer(FAST, lambda asset_id: store.asset_path(project_id, asset_id))


def _timeline(assets: dict, **overrides: object) -> Timeline:
    base = {
        "intent": "test reel",
        "canvas": Canvas(width=270, height=480, fps=24),
        "scenes": (
            Scene(
                id="s1",
                duration=1.2,
                role="hook",
                visual=ClipVisual(
                    asset_id=assets["clip_a"].sha256, source_start=2.0, source_end=3.2
                ),
                overlays=(TextOverlay(content="a trimmed clip"),),
            ),
            Scene(
                id="s2",
                duration=1.5,
                visual=AssetVisual(asset_id=assets["photo"].sha256, fit="cover"),
                motion=KenBurns(start_rect=(0.0, 0.0, 1.0, 1.0), end_rect=(0.1, 0.1, 0.7, 0.7)),
            ),
        ),
        "assets": {a.sha256: a for a in assets.values()},
        "export": Export(audio_variants=("full",)),
    }
    return Timeline(**{**base, **overrides})  # type: ignore[arg-type]


def test_renders_clips_and_photos_to_one_video(project: tuple) -> None:
    store, pid, assets = project
    timeline = _timeline(assets)
    outputs = render(timeline, pid, _renderer(store, pid), store)

    assert set(outputs) == {"full"}
    video = outputs["full"]
    assert video.exists() and video.stat().st_size > 0

    info = _renderer(store, pid).probe(video)
    assert (info.width, info.height) == (270, 480)
    assert info.duration == pytest.approx(timeline.duration, abs=0.3)


def test_landscape_source_is_fitted_to_a_vertical_canvas(project: tuple) -> None:
    """The camera-roll case: wide phone clips into a 9:16 frame."""
    store, pid, assets = project
    outputs = render(_timeline(assets), pid, _renderer(store, pid), store)
    info = _renderer(store, pid).probe(outputs["full"])
    assert info.height > info.width


def test_editing_one_scene_reencodes_only_that_scene(project: tuple) -> None:
    """The property the whole cache design exists for."""
    store, pid, assets = project
    renderer = _renderer(store, pid)
    first = _timeline(assets)
    render(first, pid, renderer, store)

    cache = store.segment_cache_dir(pid)
    before = {p.name for p in cache.glob("*.mp4")}
    assert len(before) == 2

    scenes = list(first.scenes)
    scenes[1] = scenes[1].model_copy(update={"duration": 2.4})
    second = first.model_copy(update={"scenes": tuple(scenes), "version": 2})

    plan = plan_render(second, pid, store)
    assert plan.rebuild == ("s2",)
    assert plan.cached == ("s1",)

    render(second, pid, renderer, store)
    after = {p.name for p in cache.glob("*.mp4")}
    assert len(after - before) == 1, "exactly one new segment"
    assert before <= after, "existing segments are reused, not replaced"


def test_unchanged_timeline_rebuilds_nothing(project: tuple) -> None:
    store, pid, assets = project
    timeline = _timeline(assets)
    render(timeline, pid, _renderer(store, pid), store)
    assert plan_render(timeline, pid, store).rebuild == ()


def test_audio_variants_share_one_video_pass(project: tuple) -> None:
    store, pid, assets = project
    timeline = _timeline(assets, export=Export(audio_variants=("full", "narration_only", "silent")))
    outputs = render(timeline, pid, _renderer(store, pid), store)

    assert set(outputs) == {"full", "narration_only", "silent"}
    renderer = _renderer(store, pid)
    assert renderer.probe(outputs["silent"]).has_audio is False
    assert renderer.probe(outputs["full"]).has_audio is True


def test_transitions_produce_a_watchable_cut(project: tuple) -> None:
    """A cross-fade re-encodes rather than stream-copying; it must still come out sane."""
    store, pid, assets = project
    timeline = _timeline(assets)
    scenes = list(timeline.scenes)
    scenes[1] = scenes[1].model_copy(
        update={"transition_in": Transition(kind="fade", duration=0.4)}
    )
    faded = timeline.model_copy(update={"scenes": tuple(scenes)})

    outputs = render(faded, pid, _renderer(store, pid), store)
    info = _renderer(store, pid).probe(outputs["full"])
    # The overlap shortens the result relative to the sum of the scenes.
    assert info.duration == pytest.approx(faded.duration, abs=0.35)
    assert info.duration < sum(s.duration for s in faded.scenes)


def test_preview_renders_smaller(project: tuple) -> None:
    store, pid, assets = project
    timeline = _timeline(assets, canvas=Canvas(width=1080, height=1920, fps=24))
    outputs = render(timeline, pid, _renderer(store, pid), store, preview=True)
    info = _renderer(store, pid).probe(outputs["full"])
    assert info.height == 480
    assert info.width == 270


def test_render_refuses_while_music_is_still_being_chosen(project: tuple) -> None:
    """Rendering mid-negotiation means the user was never asked. See the decision log."""
    store, pid, assets = project
    timeline = _timeline(assets, music=Music(state="candidates_ready"))
    with pytest.raises(GenvaiError, match="candidates_ready"):
        render(timeline, pid, _renderer(store, pid), store)


def test_empty_timeline_is_rejected(project: tuple) -> None:
    store, pid, assets = project
    with pytest.raises(GenvaiError, match="no scenes"):
        render(_timeline(assets, scenes=()), pid, _renderer(store, pid), store)


def test_probe_reads_real_media(project: tuple, sample_media: dict[str, Path]) -> None:
    store, pid, _ = project
    info = _renderer(store, pid).probe(sample_media["clip_a"])
    assert info.duration == pytest.approx(8.0, abs=0.2)
    assert (info.width, info.height) == (1280, 720)
    assert info.has_audio


def test_unmuted_clip_keeps_its_own_audio(project: tuple) -> None:
    store, pid, assets = project
    timeline = _timeline(
        assets,
        scenes=(
            Scene(
                id="s1",
                duration=1.2,
                visual=ClipVisual(
                    asset_id=assets["clip_a"].sha256,
                    source_start=1.0,
                    source_end=2.2,
                    mute=False,
                ),
            ),
        ),
    )
    outputs = render(timeline, pid, _renderer(store, pid), store)
    assert _renderer(store, pid).probe(outputs["full"]).has_audio


def test_sped_up_clip_retimes_audio_with_video(project: tuple) -> None:
    """setpts moves the picture but not the sound; without atempo they drift apart."""
    store, pid, assets = project
    timeline = _timeline(
        assets,
        scenes=(
            Scene(
                id="s1",
                duration=1.0,
                visual=ClipVisual(
                    asset_id=assets["clip_a"].sha256,
                    source_start=1.0,
                    source_end=3.0,
                    speed=2.0,
                    mute=False,
                ),
            ),
        ),
    )
    outputs = render(timeline, pid, _renderer(store, pid), store)
    info = _renderer(store, pid).probe(outputs["full"])
    assert info.has_audio
    assert info.duration == pytest.approx(1.0, abs=0.25)


def test_a_long_unbroken_word_is_broken_not_overflowed() -> None:
    """A phone number or URL has no spaces; left alone it runs off the side of the frame."""
    from genvai.adapters.ffmpeg import _wrap

    wrapped = _wrap("7043641428", 6)
    assert wrapped.split("\n") == ["704364", "1428"]


def test_wrapping_still_prefers_spaces() -> None:
    from genvai.adapters.ffmpeg import _wrap

    assert _wrap("call or DM to book", 8).split("\n") == ["call or", "DM to", "book"]


def test_a_long_word_among_short_ones_does_not_lose_the_others() -> None:
    from genvai.adapters.ffmpeg import _wrap

    assert "".join(_wrap("hi supercalifragilistic bye", 8).split("\n")).replace(" ", "") == (
        "hisupercalifragilisticbye"
    )


def test_line_breaks_in_the_text_are_kept() -> None:
    """A price list is written one offer per line; flattening it runs them together."""
    from genvai.adapters.ffmpeg import _wrap

    assert _wrap("WAX  250\nBLEACH  250", 30).split("\n") == ["WAX 250", "BLEACH 250"]


def test_centred_text_centres_each_line(tmp_path: Path) -> None:
    """Otherwise a short second line hangs off the left edge of a long first one."""
    from genvai.adapters.ffmpeg import _drawtext
    from genvai.timeline import TextStyle

    def graph(position: str) -> str:
        return _drawtext("a b", position, TextStyle(), Canvas(), 2.0, 0.0, None, tmp_path / "t.txt")

    assert "text_align=C" in graph("center")
    assert "text_align=C" in graph("bottom_center")
    assert "text_align" not in graph("bottom_left")
