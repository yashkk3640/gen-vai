"""Probing: what a file is, before anything judges how good it is."""

from pathlib import Path

import pytest
from PIL import Image

from genvai.adapters.probe import media_kind, probe, walk


def test_recognises_photos_and_clips() -> None:
    assert media_kind(Path("a.JPG")) == "image"
    assert media_kind(Path("a.heic")) == "image"
    assert media_kind(Path("a.MOV")) == "video"
    assert media_kind(Path("a.mp4")) == "video"


def test_ignores_everything_else() -> None:
    """A camera roll folder is full of files nobody wants imported."""
    assert media_kind(Path("notes.txt")) is None
    assert media_kind(Path(".DS_Store")) is None


def test_walk_expands_folders(tmp_path: Path) -> None:
    (tmp_path / "sub").mkdir()
    (tmp_path / "a.jpg").write_bytes(b"x")
    (tmp_path / "sub" / "b.mp4").write_bytes(b"x")
    (tmp_path / "notes.txt").write_bytes(b"x")
    (tmp_path / ".hidden.jpg").write_bytes(b"x")

    names = [p.name for p in walk((tmp_path,))]
    assert names == ["a.jpg", "b.mp4"]


def test_walk_deduplicates(tmp_path: Path) -> None:
    photo = tmp_path / "a.jpg"
    photo.write_bytes(b"x")
    assert len(walk((tmp_path, photo))) == 1


def test_walk_ignores_missing_paths(tmp_path: Path) -> None:
    assert walk((tmp_path / "nope",)) == ()


def test_probes_a_photo(tmp_path: Path, ffmpeg_binary: Path) -> None:
    path = tmp_path / "p.jpg"
    Image.new("RGB", (640, 480), (10, 20, 30)).save(path)

    item = probe(path, "abc", ffmpeg_binary)
    assert item.kind == "image"
    assert (item.width, item.height) == (640, 480)
    assert item.duration is None
    assert item.quality is None, "probing does not measure"
    assert item.source_name == "p.jpg"


def test_probes_a_clip(sample_media: dict[str, Path], ffmpeg_binary: Path) -> None:
    item = probe(sample_media["clip_a"], "def", ffmpeg_binary)
    assert item.kind == "video"
    assert (item.width, item.height) == (1280, 720)
    assert item.duration == pytest.approx(8.0, abs=0.2)
    assert item.fps == pytest.approx(30.0, abs=0.1)
    assert not item.is_vertical


def test_rejects_a_non_media_file(tmp_path: Path, ffmpeg_binary: Path) -> None:
    path = tmp_path / "notes.txt"
    path.write_text("hello")
    with pytest.raises(ValueError, match="not a media file"):
        probe(path, "x", ffmpeg_binary)


def test_reads_exif_capture_time_and_orientation(tmp_path: Path, ffmpeg_binary: Path) -> None:
    """Chronological order is the default narrative, so capture time has to survive."""
    path = tmp_path / "shot.jpg"
    image = Image.new("RGB", (400, 300), (90, 90, 90))
    exif = image.getexif()
    exif[0x0112] = 6  # Orientation: rotate 90 clockwise
    exif[0x0132] = "2026:09:22 14:03:11"  # DateTime
    image.save(path, exif=exif)

    item = probe(path, "x", ffmpeg_binary)
    assert item.captured_at == "2026-09-22T14:03:11"
    assert item.rotation == 90


def test_missing_exif_is_not_an_error(tmp_path: Path, ffmpeg_binary: Path) -> None:
    """Screenshots and downloads have no EXIF; they still have to import."""
    path = tmp_path / "plain.png"
    Image.new("RGB", (200, 200)).save(path)
    item = probe(path, "x", ffmpeg_binary)
    assert item.captured_at is None
    assert item.rotation == 0
