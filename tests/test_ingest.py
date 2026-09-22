"""Ingest end to end: real files in, an analysed library out.

Uses the bundled ffmpeg and real decoding, because the parts most likely to break are
the ones that touch actual pixels.
"""

import shutil
from pathlib import Path

import pytest

from genvai.adapters.analyzer import FrameAnalyzer
from genvai.adapters.fs_store import FilesystemStore
from genvai.pipeline.ingest import ingest, probe_only, summarise


@pytest.fixture
def store(tmp_path: Path) -> FilesystemStore:
    store = FilesystemStore(tmp_path / "projects")
    store.create("test", "t")
    return store


@pytest.fixture
def roll(tmp_path: Path, sample_media: dict[str, Path]) -> Path:
    """A folder shaped like a camera roll: photos, clips, and junk to ignore."""
    directory = tmp_path / "roll"
    directory.mkdir()
    for key in ("photo", "clip_a", "clip_b"):
        shutil.copy2(sample_media[key], directory / sample_media[key].name)
    (directory / "notes.txt").write_text("not media")
    return directory


def _ingest(roll: Path, store: FilesystemStore, ffmpeg: Path):
    return ingest((roll,), "t", FrameAnalyzer(ffmpeg), store, ffmpeg)


def test_imports_photos_and_clips(roll: Path, store: FilesystemStore, ffmpeg_binary: Path) -> None:
    library = _ingest(roll, store, ffmpeg_binary)
    assert len(library.items) == 3
    assert len(library.clips) == 2
    assert len(library.photos) == 1


def test_ignores_non_media(roll: Path, store: FilesystemStore, ffmpeg_binary: Path) -> None:
    library = _ingest(roll, store, ffmpeg_binary)
    assert all(not i.source_name.endswith(".txt") for i in library.items)


def test_everything_gets_measured(roll: Path, store: FilesystemStore, ffmpeg_binary: Path) -> None:
    library = _ingest(roll, store, ffmpeg_binary)
    assert library.unanalysed == (), "no item should be left unmeasured"
    for item in library.items:
        assert item.quality is not None
        assert 0.0 <= item.quality.sharpness <= 1.0
        assert 0.0 <= item.quality.exposure <= 1.0


def test_clips_get_spans_photos_do_not(
    roll: Path, store: FilesystemStore, ffmpeg_binary: Path
) -> None:
    """A photo is a clip with no span to choose."""
    library = _ingest(roll, store, ffmpeg_binary)
    assert all(clip.spans for clip in library.clips)
    assert all(photo.spans == () for photo in library.photos)


def test_spans_fall_inside_their_clip(
    roll: Path, store: FilesystemStore, ffmpeg_binary: Path
) -> None:
    library = _ingest(roll, store, ffmpeg_binary)
    for clip in library.clips:
        assert clip.duration is not None
        for span in clip.spans:
            assert 0.0 <= span.start < span.end <= clip.duration + 0.5


def test_clips_get_an_audio_reading(
    roll: Path, store: FilesystemStore, ffmpeg_binary: Path
) -> None:
    """Peak level answers whether a clip's own sound is worth unmuting."""
    library = _ingest(roll, store, ffmpeg_binary)
    assert all(c.quality and c.quality.audio_peak_db is not None for c in library.clips)


def test_reimporting_is_free(roll: Path, store: FilesystemStore, ffmpeg_binary: Path) -> None:
    """The whole point of keying by content hash."""
    first = _ingest(roll, store, ffmpeg_binary)
    second = _ingest(roll, store, ffmpeg_binary)
    assert len(second.items) == len(first.items)
    assert {i.asset_id for i in second.items} == {i.asset_id for i in first.items}


def test_adding_more_files_keeps_the_old_ones(
    roll: Path, store: FilesystemStore, ffmpeg_binary: Path, sample_media: dict[str, Path]
) -> None:
    library = ingest(
        (sample_media["photo"],), "t", FrameAnalyzer(ffmpeg_binary), store, ffmpeg_binary
    )
    assert len(library.items) == 1
    library = _ingest(roll, store, ffmpeg_binary)
    assert len(library.items) == 3


def test_duplicate_takes_are_grouped(
    tmp_path: Path, store: FilesystemStore, ffmpeg_binary: Path, sample_media: dict[str, Path]
) -> None:
    """Two copies of one photo under different names are one shot, not two."""
    directory = tmp_path / "burst"
    directory.mkdir()
    from PIL import Image

    with Image.open(sample_media["photo"]) as image:
        image.save(directory / "take1.jpg", quality=95)
        image.save(directory / "take2.jpg", quality=70)

    library = ingest((directory,), "t", FrameAnalyzer(ffmpeg_binary), store, ffmpeg_binary)
    assert len(library.items) == 2, "different bytes, so two assets"
    assert all(i.dedup_group for i in library.items), "but one shot"
    assert len({i.dedup_group for i in library.items}) == 1
    assert len(library.deduplicated()) == 1, "only one survives the view"


def test_library_persists(roll: Path, store: FilesystemStore, ffmpeg_binary: Path) -> None:
    library = _ingest(roll, store, ffmpeg_binary)
    assert store.load_media("t") == library


def test_assets_are_copied_into_the_project(
    roll: Path, store: FilesystemStore, ffmpeg_binary: Path
) -> None:
    """A project has to keep working after the original camera roll is deleted."""
    library = _ingest(roll, store, ffmpeg_binary)
    for item in library.items:
        assert store.asset_path("t", item.asset_id).exists()


def test_probe_only_is_fast_and_shallow(roll: Path, ffmpeg_binary: Path) -> None:
    library = probe_only((roll,), ffmpeg_binary)
    assert len(library.items) == 3
    assert all(i.quality is None for i in library.items), "probing does not measure"


def test_summary_reads_sensibly(roll: Path, store: FilesystemStore, ffmpeg_binary: Path) -> None:
    library = _ingest(roll, store, ffmpeg_binary)
    text = summarise(library.items)
    assert "3 files" in text
    assert "2 clips" in text
    assert "1 photos" in text


def test_progress_is_reported(roll: Path, store: FilesystemStore, ffmpeg_binary: Path) -> None:
    seen: list[tuple[str, int, int]] = []
    ingest(
        (roll,),
        "t",
        FrameAnalyzer(ffmpeg_binary),
        store,
        ffmpeg_binary,
        on_progress=lambda name, i, total: seen.append((name, i, total)),
    )
    assert len(seen) == 3
    assert all(total == 3 for _, _, total in seen)
