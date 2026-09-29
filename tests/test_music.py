"""The music provider, and the consent gate around downloads.

The gate is the part that matters. `search` must never touch the network, and `fetch`
must refuse anything that was not explicitly approved - so those are tested by making a
download attempt fail loudly rather than by mocking it away.
"""

import json
from pathlib import Path

import pytest

from genvai.adapters.music_local import LocalMusicProvider
from genvai.config import MusicSettings
from genvai.errors import ConfirmationRequired
from genvai.timeline import MusicCandidate, MusicQuery


@pytest.fixture
def library(tmp_path: Path) -> Path:
    """A small tagged library: two described tracks and one bare file."""
    directory = tmp_path / "music"
    directory.mkdir()
    for name, meta in (
        ("quiet-hours", {"mood": "calm", "genre": "solo piano", "bpm": 72, "licence": "CC-BY-4.0"}),
        ("night-drive", {"mood": "energetic", "genre": "synthwave", "bpm": 118, "licence": "CC0"}),
    ):
        (directory / f"{name}.mp3").write_bytes(b"audio")
        (directory / f"{name}.json").write_text(json.dumps(meta), encoding="utf-8")
    (directory / "untitled-sketch.wav").write_bytes(b"audio")
    return directory


@pytest.fixture
def provider(library: Path) -> LocalMusicProvider:
    return LocalMusicProvider(MusicSettings(library_dir=library))


# -------------------------------------------------------------------------- search


def test_finds_tracks_in_the_library(provider: LocalMusicProvider) -> None:
    found = provider.search(MusicQuery(mood="calm"))
    assert len(found) == 3


def test_the_best_match_comes_first(provider: LocalMusicProvider) -> None:
    best = provider.search(MusicQuery(mood="calm", genre="piano"))[0]
    assert "quiet" in best.title.lower()


def test_genre_steers_the_result(provider: LocalMusicProvider) -> None:
    best = provider.search(MusicQuery(mood="energetic", genre="synthwave"))[0]
    assert "night" in best.title.lower()


def test_everything_is_returned_even_on_a_poor_match(provider: LocalMusicProvider) -> None:
    """A small library must not answer 'calm piano' with silence."""
    found = provider.search(MusicQuery(mood="industrial", genre="death metal"))
    assert found != ()


def test_a_bare_file_still_becomes_a_candidate(provider: LocalMusicProvider) -> None:
    titles = {c.title for c in provider.search(MusicQuery(mood="any"))}
    assert "untitled sketch" in titles, "dropping files in a folder is enough to start"


def test_licence_is_carried_through(provider: LocalMusicProvider) -> None:
    by_title = {c.title: c for c in provider.search(MusicQuery(mood="calm"))}
    assert by_title["quiet hours"].licence == "CC-BY-4.0"


def test_an_untagged_file_has_an_unknown_licence(provider: LocalMusicProvider) -> None:
    by_title = {c.title: c for c in provider.search(MusicQuery(mood="calm"))}
    assert by_title["untitled sketch"].licence == "unknown"


def test_the_limit_is_respected(provider: LocalMusicProvider) -> None:
    assert len(provider.search(MusicQuery(mood="calm"), limit=2)) == 2


def test_a_missing_library_searches_to_nothing(tmp_path: Path) -> None:
    absent = LocalMusicProvider(MusicSettings(library_dir=tmp_path / "nope"))
    assert absent.is_available() is False
    assert absent.search(MusicQuery(mood="calm")) == ()


def test_tempo_is_a_band_not_a_target(provider: LocalMusicProvider) -> None:
    """'Around 70' should not reject 72."""
    best = provider.search(MusicQuery(mood="calm", bpm=(60, 80)))[0]
    assert "quiet" in best.title.lower()


# ----------------------------------------------------------------------- the gate


def _remote(url: str = "https://example.invalid/track.mp3") -> MusicCandidate:
    return MusicCandidate(
        id="c1", title="Somewhere Else", source_url=url, licence="CC-BY-4.0", duration=90.0
    )


def test_a_local_track_needs_no_confirmation(provider: LocalMusicProvider, library: Path) -> None:
    """Nothing leaves the machine, so nothing needs consenting to."""
    candidate = provider.search(MusicQuery(mood="calm"))[0]
    assert provider.fetch(candidate, confirmed=False).exists()


def test_an_unapproved_download_is_refused(provider: LocalMusicProvider) -> None:
    with pytest.raises(ConfirmationRequired, match="not been approved"):
        provider.fetch(_remote(), confirmed=False)


def test_the_refusal_says_how_to_approve(provider: LocalMusicProvider) -> None:
    with pytest.raises(ConfirmationRequired, match="--approve c1"):
        provider.fetch(_remote(), confirmed=False)


def test_approval_alone_is_not_enough(library: Path) -> None:
    """The setting is standing policy; the flag is this particular consent. Both gate."""
    provider = LocalMusicProvider(MusicSettings(library_dir=library, allow_download=False))
    with pytest.raises(ConfirmationRequired, match="ALLOW_DOWNLOAD"):
        provider.fetch(_remote(), confirmed=True)


def test_nothing_is_written_when_a_download_is_refused(
    provider: LocalMusicProvider, library: Path
) -> None:
    before = set(library.iterdir())
    with pytest.raises(ConfirmationRequired):
        provider.fetch(_remote(), confirmed=False)
    assert set(library.iterdir()) == before


def test_searching_never_reaches_the_network(
    provider: LocalMusicProvider, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`search` is read-only by contract, so make any request an error and run it."""

    def forbidden(*args: object, **kwargs: object) -> None:
        raise AssertionError("search must not make a network request")

    monkeypatch.setattr("httpx.stream", forbidden)
    monkeypatch.setattr("httpx.get", forbidden)
    assert provider.search(MusicQuery(mood="calm", genre="piano")) != ()


# ---------------------------------------------------------------- beat detection


def test_beats_are_read_from_a_real_file(tmp_path: Path, ffmpeg_binary: Path) -> None:
    """The adapter path: an encoded file on disk, decoded and analysed."""
    import subprocess

    from genvai.adapters.beat import FFmpegBeatDetector

    track = tmp_path / "click.wav"
    # A 128 BPM metronome: a short tone every 60/128 seconds.
    subprocess.run(
        [
            str(ffmpeg_binary),
            "-y",
            "-hide_banner",
            "-loglevel",
            "error",
            "-f",
            "lavfi",
            "-i",
            "sine=frequency=880:duration=16",
            "-af",
            "atrim=0:16,asetrate=44100,volume='if(lt(mod(t,0.46875),0.03),1,0)':eval=frame",
            str(track),
        ],
        check=True,
    )

    result = FFmpegBeatDetector(ffmpeg_binary).detect(track)
    assert result.bpm == pytest.approx(128.0, rel=0.06)
    assert len(result.beats) > 20
    assert result.downbeats[1] == pytest.approx(result.beats[4])


def test_an_unreadable_track_does_not_break_the_render(tmp_path: Path, ffmpeg_binary: Path) -> None:
    """Music is a garnish; a track that will not decode costs beat alignment, not the cut."""
    from genvai.adapters.beat import FFmpegBeatDetector

    broken = tmp_path / "broken.mp3"
    broken.write_bytes(b"not audio")
    result = FFmpegBeatDetector(ffmpeg_binary).detect(broken)
    assert result.beats == ()
    assert result.bpm > 0


# ------------------------------------------------------------------- the two steps


@pytest.fixture
def project(tmp_path: Path, library: Path):
    """A project with a one-shot timeline, ready for a music bed."""
    from genvai.adapters.fs_store import FilesystemStore
    from genvai.timeline import AssetVisual, Scene, Timeline

    store = FilesystemStore(tmp_path / "projects")
    store.create("t", "t")
    store.save_timeline(
        "t",
        Timeline(
            intent="t",
            scenes=(Scene(id="s1", duration=2.2, visual=AssetVisual(asset_id="a")),),
        ),
    )
    return store


def test_suggesting_does_not_resolve_anything(project, library: Path) -> None:
    """candidates_ready is not renderable, precisely because nobody has chosen yet."""
    from genvai.pipeline.music import suggest

    timeline = suggest(
        "t", project, LocalMusicProvider(MusicSettings(library_dir=library)), mood="calm"
    )
    assert timeline.music.state == "candidates_ready"
    assert timeline.music.asset_id is None
    assert not timeline.music.is_renderable


def test_suggesting_records_what_was_offered(project, library: Path) -> None:
    from genvai.pipeline.music import suggest

    timeline = suggest(
        "t", project, LocalMusicProvider(MusicSettings(library_dir=library)), mood="calm"
    )
    assert len(timeline.music.candidates) == 3
    assert timeline.music.query is not None


def test_an_empty_library_says_what_to_do(project, tmp_path: Path) -> None:
    from genvai.errors import GenvaiError
    from genvai.pipeline.music import suggest

    absent = LocalMusicProvider(MusicSettings(library_dir=tmp_path / "nope"))
    with pytest.raises(GenvaiError, match="LIBRARY_DIR"):
        suggest("t", project, absent, mood="calm")


def test_approving_resolves_and_records_provenance(project, library: Path) -> None:
    """A track whose licence is lost cannot safely be published with."""
    from genvai.adapters.beat import FFmpegBeatDetector
    from genvai.pipeline.music import approve, suggest

    provider = LocalMusicProvider(MusicSettings(library_dir=library))
    suggest("t", project, provider, mood="calm")
    detector = FFmpegBeatDetector(Path("does-not-exist"))

    timeline = approve("t", project, provider, detector, "quiet-hours", confirmed=True)
    assert timeline.music.state == "resolved"
    assert timeline.music.is_renderable
    asset = timeline.assets[timeline.music.asset_id]
    assert asset.provenance.licence == "CC-BY-4.0"
    assert asset.kind == "audio"


def test_approving_an_unoffered_track_is_refused(project, library: Path) -> None:
    from genvai.adapters.beat import FFmpegBeatDetector
    from genvai.errors import GenvaiError
    from genvai.pipeline.music import approve, suggest

    provider = LocalMusicProvider(MusicSettings(library_dir=library))
    suggest("t", project, provider, mood="calm")
    with pytest.raises(GenvaiError, match="No candidate"):
        approve("t", project, provider, FFmpegBeatDetector(Path("x")), "made-up", confirmed=True)


def test_declining_is_renderable(project) -> None:
    """No music is an answer, not an absence of one."""
    from genvai.pipeline.music import decline

    timeline = decline("t", project)
    assert timeline.music.state == "declined"
    assert timeline.music.is_renderable


def test_each_music_step_is_its_own_version(project, library: Path) -> None:
    from genvai.pipeline.music import decline, suggest

    provider = LocalMusicProvider(MusicSettings(library_dir=library))
    first = suggest("t", project, provider, mood="calm")
    second = decline("t", project)
    assert second.version > first.version
    assert project.load_timeline("t", first.version).music.state == "candidates_ready"


# ------------------------------------------------------------------ a file of your own


def _silent_only(store) -> None:
    """Shape the project's timeline like a promo: silent cut only."""
    from genvai.timeline import Export

    timeline = store.load_timeline("t")
    store.save_timeline(
        "t",
        timeline.model_copy(
            update={"export": Export(audio_variants=("silent",)), "version": timeline.version + 1}
        ),
    )


def test_attaching_a_file_resolves_without_a_library(project, library: Path) -> None:
    from genvai.adapters.beat import FFmpegBeatDetector
    from genvai.pipeline.music import attach

    track = library / "night-drive.mp3"
    timeline = attach("t", project, FFmpegBeatDetector(Path("absent")), track)
    assert timeline.music.state == "resolved"
    asset = timeline.assets[timeline.music.asset_id]
    assert asset.kind == "audio"
    assert asset.provenance.provider == "user"


def test_attaching_adds_the_full_cut(project, library: Path) -> None:
    """A promo exports silent only until there is something to hear."""
    from genvai.adapters.beat import FFmpegBeatDetector
    from genvai.pipeline.music import attach

    _silent_only(project)
    timeline = attach("t", project, FFmpegBeatDetector(Path("absent")), library / "night-drive.mp3")
    assert timeline.export.audio_variants == ("full", "silent")


def test_a_sole_track_is_turned_up(project, library: Path) -> None:
    from genvai.adapters.beat import FFmpegBeatDetector
    from genvai.pipeline.music import SOLE_TRACK_DB, attach

    timeline = attach(
        "t",
        project,
        FFmpegBeatDetector(Path("absent")),
        library / "night-drive.mp3",
        gain_db=SOLE_TRACK_DB,
    )
    assert timeline.music.gain_db == SOLE_TRACK_DB


def test_attaching_a_missing_file_is_refused(project, tmp_path: Path) -> None:
    from genvai.adapters.beat import FFmpegBeatDetector
    from genvai.errors import GenvaiError
    from genvai.pipeline.music import attach

    with pytest.raises(GenvaiError, match="No audio file"):
        attach("t", project, FFmpegBeatDetector(Path("absent")), tmp_path / "nope.mp3")


def test_approving_adds_the_full_cut(project, library: Path) -> None:
    from genvai.adapters.beat import FFmpegBeatDetector
    from genvai.pipeline.music import approve, suggest

    _silent_only(project)
    provider = LocalMusicProvider(MusicSettings(library_dir=library))
    suggest("t", project, provider, mood="calm")
    timeline = approve(
        "t", project, provider, FFmpegBeatDetector(Path("x")), "quiet-hours", confirmed=True
    )
    assert "full" in timeline.export.audio_variants
