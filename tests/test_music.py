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
