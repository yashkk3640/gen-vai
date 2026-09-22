"""Reading what a file *is*, before anything looks at how good it is.

Fast enough to run on drop - dimensions, duration, capture time, rotation - so the user
sees what was picked up before committing to the slow analysis pass.

Photos go through Pillow and fall back to ffmpeg, which covers formats Pillow was not
built for (HEIC from an iPhone being the one that matters).
"""

import re
import subprocess
from datetime import datetime
from pathlib import Path

from PIL import ExifTags, Image

from genvai.media import MediaItem

IMAGE_SUFFIXES = frozenset(
    {".jpg", ".jpeg", ".png", ".heic", ".heif", ".webp", ".bmp", ".tif", ".tiff", ".gif"}
)
VIDEO_SUFFIXES = frozenset(
    {".mp4", ".mov", ".m4v", ".avi", ".mkv", ".webm", ".3gp", ".mts", ".m2ts", ".mpg"}
)

_ORIENTATION_TO_DEGREES = {1: 0, 3: 180, 6: 90, 8: 270}
"""EXIF orientation values that are plain rotations. Mirrored variants are ignored."""

_CREATION = re.compile(r"creation_time\s*:\s*(\S+)")
_ROTATION = re.compile(r"rotation of (-?\d+(?:\.\d+)?) degrees")
_ROTATE_TAG = re.compile(r"\brotate\s*:\s*(-?\d+)")
_DIMENSIONS = re.compile(r"\b(\d{2,5})x(\d{2,5})\b")
_FPS = re.compile(r"(\d+(?:\.\d+)?) fps")
_DURATION = re.compile(r"Duration:\s*(\d+):(\d\d):(\d\d(?:\.\d+)?)")


def media_kind(path: Path) -> str | None:
    """`image`, `video`, or None when the extension is not media."""
    suffix = path.suffix.lower()
    if suffix in IMAGE_SUFFIXES:
        return "image"
    if suffix in VIDEO_SUFFIXES:
        return "video"
    return None


def walk(sources: tuple[Path, ...]) -> tuple[Path, ...]:
    """Expand folders into the media files inside them, sorted and deduplicated.

    Hidden files and anything that is not recognisable media are skipped silently - a
    camera roll folder is full of `.DS_Store` and sidecar files nobody wants imported.
    """
    found: set[Path] = set()
    for source in sources:
        if source.is_dir():
            found.update(
                p
                for p in source.rglob("*")
                if p.is_file() and media_kind(p) and not p.name.startswith(".")
            )
        elif source.is_file() and media_kind(source):
            found.add(source)
    return tuple(sorted(found))


def probe(path: Path, asset_id: str, ffmpeg: Path) -> MediaItem:
    """Read a file's shape without measuring its quality.

    Returns an item with `quality` and `spans` unset; the analyser fills those.
    """
    kind = media_kind(path)
    if kind is None:
        raise ValueError(f"not a media file: {path.name}")
    if kind == "image":
        return _probe_image(path, asset_id, ffmpeg)
    return _probe_video(path, asset_id, ffmpeg)


def _probe_image(path: Path, asset_id: str, ffmpeg: Path) -> MediaItem:
    try:
        with Image.open(path) as image:
            width, height = image.size
            captured, rotation = _exif(image)
    except Exception:  # noqa: BLE001 - Pillow cannot open every format; ffmpeg often can
        report = _ffmpeg_report(path, ffmpeg)
        width, height = _dimensions(report)
        captured, rotation = _creation_time(report), 0

    return MediaItem(
        asset_id=asset_id,
        kind="image",
        source_name=path.name,
        width=width,
        height=height,
        captured_at=captured,
        rotation=rotation,
    )


def _probe_video(path: Path, asset_id: str, ffmpeg: Path) -> MediaItem:
    report = _ffmpeg_report(path, ffmpeg)
    width, height = _dimensions(report)
    rotation = _rotation(report)
    # A clip shot in portrait reports landscape dimensions plus a rotation; swapping
    # them here means everything downstream sees the picture as it will be displayed.
    if rotation in (90, 270):
        width, height = height, width

    return MediaItem(
        asset_id=asset_id,
        kind="video",
        source_name=path.name,
        width=width,
        height=height,
        duration=_duration(report),
        fps=_fps(report),
        captured_at=_creation_time(report),
        rotation=rotation,
    )


def _ffmpeg_report(path: Path, ffmpeg: Path) -> str:
    """ffmpeg's own stream report. Exits non-zero by design - there is no output file."""
    result = subprocess.run(
        [str(ffmpeg), "-hide_banner", "-i", str(path)],
        capture_output=True,
        text=True,
        check=False,
    )
    return result.stderr


def _exif(image: Image.Image) -> tuple[str | None, int]:
    """Capture time and rotation from EXIF, if the camera wrote any."""
    try:
        raw = image.getexif()
    except Exception:  # noqa: BLE001
        return None, 0
    if not raw:
        return None, 0

    tags = {ExifTags.TAGS.get(key, key): value for key, value in raw.items()}
    rotation = _ORIENTATION_TO_DEGREES.get(tags.get("Orientation", 1), 0)

    stamp = tags.get("DateTimeOriginal") or tags.get("DateTime")
    return _parse_exif_time(stamp), rotation


def _parse_exif_time(stamp: object) -> str | None:
    """EXIF writes `2026:09:22 14:03:11`, which is ISO-8601 with the wrong separators."""
    if not isinstance(stamp, str):
        return None
    try:
        return datetime.strptime(stamp.strip(), "%Y:%m:%d %H:%M:%S").isoformat()
    except ValueError:
        return None


def _creation_time(report: str) -> str | None:
    match = _CREATION.search(report)
    return match.group(1) if match else None


def _rotation(report: str) -> int:
    """Normalise to a clockwise angle in [0, 360).

    Containers express this two ways - a `rotate` metadata tag or a display matrix - and
    the display matrix reports the inverse, hence the negation.
    """
    matrix = _ROTATION.search(report)
    if matrix:
        return int(-float(matrix.group(1))) % 360
    tag = _ROTATE_TAG.search(report)
    return int(tag.group(1)) % 360 if tag else 0


def _dimensions(report: str) -> tuple[int, int]:
    for line in report.splitlines():
        if "Video:" not in line:
            continue
        match = _DIMENSIONS.search(line)
        if match:
            return int(match.group(1)), int(match.group(2))
    return 0, 0


def _fps(report: str) -> float | None:
    match = _FPS.search(report)
    return float(match.group(1)) if match else None


def _duration(report: str) -> float | None:
    match = _DURATION.search(report)
    if not match:
        return None
    hours, minutes, seconds = match.groups()
    return int(hours) * 3600 + int(minutes) * 60 + float(seconds)
