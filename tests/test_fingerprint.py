"""Fingerprinting decides what gets re-encoded, so its job is to be *exactly* as
sensitive as rendering is: every input that changes the picture must change the hash,
and nothing else may.

A false match reuses a stale segment and ships the wrong video. A false miss only costs
time. The tests below check both directions, because only testing sensitivity would let
a fingerprint that hashes the clock pass.
"""

from pathlib import Path

from genvai.fingerprint import (
    audio_fingerprint,
    hash_file,
    scene_fingerprint,
    timeline_fingerprint,
)
from genvai.timeline import (
    Asset,
    AssetProvenance,
    Canvas,
    Captions,
    ClipVisual,
    Export,
    KenBurns,
    Music,
    Narration,
    Scene,
    TextOverlay,
    TextStyle,
    Timeline,
    Transition,
)


def _asset(digest: str) -> Asset:
    return Asset(
        kind="video",
        path=f"assets/clips/{digest}.mp4",
        sha256=digest,
        provenance=AssetProvenance(provider="test"),
    )


def _timeline(**overrides: object) -> Timeline:
    base = {
        "intent": "t",
        "scenes": (
            Scene(
                id="s1",
                duration=2.0,
                visual=ClipVisual(asset_id="aaa", source_start=1.0, source_end=3.0),
                overlays=(TextOverlay(content="hello"),),
            ),
            Scene(id="s2", duration=2.0, visual=ClipVisual(asset_id="bbb", source_end=2.0)),
        ),
        "assets": {"aaa": _asset("aaa"), "bbb": _asset("bbb")},
    }
    return Timeline(**{**base, **overrides})  # type: ignore[arg-type]


def _print(timeline: Timeline, index: int = 0) -> str:
    return scene_fingerprint(timeline, timeline.scenes[index])


# ---------------------------------------------------------------- stability


def test_identical_timelines_agree() -> None:
    assert _print(_timeline()) == _print(_timeline())


def test_repeated_calls_agree() -> None:
    """Nothing time- or identity-based may leak into the hash."""
    timeline = _timeline()
    assert _print(timeline) == _print(timeline)


def test_scenes_are_independent() -> None:
    timeline = _timeline()
    assert _print(timeline, 0) != _print(timeline, 1)


# -------------------------------------------------------------- sensitivity


def test_trim_change_invalidates() -> None:
    other = _timeline()
    scenes = list(other.scenes)
    scenes[0] = scenes[0].model_copy(
        update={"visual": ClipVisual(asset_id="aaa", source_start=4.0, source_end=6.0)}
    )
    assert _print(other) != _print(other.model_copy(update={"scenes": tuple(scenes)}))


def test_duration_change_invalidates() -> None:
    timeline = _timeline()
    scenes = list(timeline.scenes)
    scenes[0] = scenes[0].model_copy(update={"duration": 3.0})
    assert _print(timeline) != _print(timeline.model_copy(update={"scenes": tuple(scenes)}))


def test_motion_change_invalidates() -> None:
    timeline = _timeline()
    scenes = list(timeline.scenes)
    scenes[0] = scenes[0].model_copy(update={"motion": KenBurns()})
    assert _print(timeline) != _print(timeline.model_copy(update={"scenes": tuple(scenes)}))


def test_canvas_change_invalidates() -> None:
    timeline = _timeline()
    wider = timeline.model_copy(update={"canvas": Canvas(width=1920, height=1080)})
    assert _print(timeline) != _print(wider)


def test_style_change_invalidates_even_though_the_scene_is_identical() -> None:
    """The scene references a style by name; the style's contents are what render."""
    timeline = _timeline()
    restyled = timeline.model_copy(update={"styles": {"caption": TextStyle(size_pct=9.0)}})
    assert _print(timeline) != _print(restyled)


def test_overlay_text_change_invalidates() -> None:
    timeline = _timeline()
    scenes = list(timeline.scenes)
    scenes[0] = scenes[0].model_copy(update={"overlays": (TextOverlay(content="goodbye"),)})
    assert _print(timeline) != _print(timeline.model_copy(update={"scenes": tuple(scenes)}))


def test_repointed_asset_invalidates() -> None:
    """Same id, different content - the digest has to follow the bytes, not the name."""
    timeline = _timeline()
    swapped = timeline.model_copy(
        update={"assets": {**timeline.assets, "aaa": _asset("different-content")}}
    )
    assert _print(timeline) != _print(swapped)


# ------------------------------------------------------- audio/video isolation


def test_music_does_not_invalidate_video_segments() -> None:
    """An audio-only edit must not force a video re-encode. This is the whole point."""
    timeline = _timeline()
    with_music = timeline.model_copy(update={"music": Music(state="declined", gain_db=-6.0)})
    assert _print(timeline) == _print(with_music)


def test_music_change_does_invalidate_the_audio_mix() -> None:
    timeline = _timeline()
    with_music = timeline.model_copy(update={"music": Music(state="declined", gain_db=-6.0)})
    assert audio_fingerprint(timeline) != audio_fingerprint(with_music)


def test_narration_invalidates_video_because_captions_are_burned_in() -> None:
    timeline = _timeline()
    scenes = list(timeline.scenes)
    scenes[0] = scenes[0].model_copy(update={"narration": Narration(text="spoken line")})
    spoken = timeline.model_copy(update={"scenes": tuple(scenes)})
    assert _print(timeline) != _print(spoken)


def test_caption_toggle_invalidates_video() -> None:
    timeline = _timeline()
    off = timeline.model_copy(update={"captions": Captions(enabled=False)})
    assert _print(timeline) != _print(off)


# ------------------------------------------------------------------ whole timeline


def test_timeline_fingerprint_follows_any_scene() -> None:
    timeline = _timeline()
    scenes = list(timeline.scenes)
    scenes[1] = scenes[1].model_copy(update={"duration": 5.0})
    assert timeline_fingerprint(timeline) != timeline_fingerprint(
        timeline.model_copy(update={"scenes": tuple(scenes)})
    )


def test_timeline_fingerprint_follows_export_settings() -> None:
    timeline = _timeline()
    silent = timeline.model_copy(update={"export": Export(audio_variants=("silent",))})
    assert timeline_fingerprint(timeline) != timeline_fingerprint(silent)


def test_transition_change_is_visible_to_the_timeline_hash() -> None:
    timeline = _timeline()
    scenes = list(timeline.scenes)
    scenes[1] = scenes[1].model_copy(
        update={"transition_in": Transition(kind="fade", duration=0.5)}
    )
    assert timeline_fingerprint(timeline) != timeline_fingerprint(
        timeline.model_copy(update={"scenes": tuple(scenes)})
    )


# ------------------------------------------------------------------------ files


def test_hash_file_follows_content(tmp_path: Path) -> None:
    same_a, same_b, different = (tmp_path / n for n in ("a", "b", "c"))
    same_a.write_bytes(b"identical")
    same_b.write_bytes(b"identical")
    different.write_bytes(b"other")
    assert hash_file(same_a) == hash_file(same_b)
    assert hash_file(same_a) != hash_file(different)


def test_hash_file_handles_multi_chunk_input(tmp_path: Path) -> None:
    """Chunked reading must not depend on the file fitting in one buffer."""
    path = tmp_path / "big"
    path.write_bytes(b"x" * (3 << 20))
    assert hash_file(path) == hash_file(path, chunk_size=1024)
