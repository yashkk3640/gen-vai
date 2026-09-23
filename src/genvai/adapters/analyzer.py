"""MediaAnalyzer over ffmpeg frame extraction plus numpy arithmetic.

No OpenCV, no model, no GPU. ffmpeg pulls a handful of frames per clip, Pillow decodes
them and `genvai.analysis` does the measuring. That is what makes analysing a hundred
phone clips feasible on a laptop: the expensive part of camera-roll editing is the
judgement, not the measurement.

Results are keyed by content hash upstream, so a file is analysed exactly once, ever.
"""

import re
import subprocess
import tempfile
from pathlib import Path

import numpy as np
from PIL import Image

from genvai.analysis import (
    FrameScore,
    exposure,
    find_spans,
    group_duplicates,
    motion,
    perceptual_hash,
    shake,
    sharpness,
    to_greyscale,
)
from genvai.media import ClipQuality, MediaItem

SAMPLE_FPS = 4.0
"""Frames per second to measure.

Fast enough to catch a two-second good stretch, sparse enough that a minute of footage
is a few hundred small decodes rather than thousands.
"""

MAX_FRAMES = 240
"""Ceiling per clip, so one accidentally long video cannot stall an import."""

WORK_WIDTH = 256
"""Frames are measured at this width. Sharpness and motion are scale-relative, and
full-resolution decoding would dominate the runtime for no extra signal."""

_MAX_VOLUME = re.compile(r"max_volume:\s*(-?\d+(?:\.\d+)?) dB")


class FrameAnalyzer:
    """Implements `MediaAnalyzer`."""

    name = "frames"

    def __init__(self, ffmpeg: Path) -> None:
        self._ffmpeg = ffmpeg

    def is_available(self) -> bool:
        return self._ffmpeg.exists()

    def analyse(self, media: Path, item: MediaItem) -> MediaItem:
        """Return the item with quality, spans and fingerprint filled in.

        Pure with respect to the library - it observes one file and returns a new item -
        so analysis can be cached per content hash and parallelised later.
        """
        frames = self._sample(media, item)
        if not frames:
            return item

        greys = [to_greyscale(frame) for frame in frames]
        focus = [sharpness(g) for g in greys]
        light = [exposure(g) for g in greys]
        movement = [motion(a, b) for a, b in zip(greys, greys[1:], strict=False)]

        quality = ClipQuality(
            sharpness=float(np.median(focus)),
            exposure=float(np.mean(light)),
            motion=float(np.mean(movement)) if movement else 0.0,
            shake=shake(movement),
            audio_peak_db=self._audio_peak(media) if item.kind == "video" else None,
        )

        spans = ()
        if item.kind == "video" and item.duration:
            step = 1.0 / SAMPLE_FPS
            trace = [
                FrameScore(
                    at=index * step,
                    sharpness=focus[index],
                    exposure=light[index],
                    motion=movement[index - 1] if index else 0.0,
                )
                for index in range(len(greys))
            ]
            spans = find_spans(trace, duration=item.duration)

        return item.model_copy(
            update={
                "quality": quality,
                "spans": spans,
                "fingerprint": perceptual_hash(greys[len(greys) // 2]),
            }
        )

    def duplicate_groups(self, items: tuple[MediaItem, ...]) -> dict[str, str]:
        """Group near-identical shots by their stored fingerprints."""
        hashes = {i.asset_id: i.fingerprint for i in items if i.fingerprint is not None}
        return group_duplicates(hashes)

    # ------------------------------------------------------------------ frames

    def _sample(self, media: Path, item: MediaItem) -> list[np.ndarray]:
        """Decode a handful of evenly spaced frames, small and in memory."""
        if item.kind == "image":
            return _load_image(media)

        with tempfile.TemporaryDirectory(prefix="genvai-frames-") as scratch:
            directory = Path(scratch)
            command = [
                str(self._ffmpeg),
                "-hide_banner",
                "-loglevel",
                "error",
                "-i",
                str(media),
                "-vf",
                f"fps={SAMPLE_FPS},scale={WORK_WIDTH}:-2",
                "-frames:v",
                str(MAX_FRAMES),
                "-f",
                "image2",
                str(directory / "%05d.png"),
            ]
            result = subprocess.run(command, capture_output=True, text=True, check=False)
            if result.returncode != 0:
                return []
            return [_as_array(p) for p in sorted(directory.glob("*.png"))]

    def _audio_peak(self, media: Path) -> float | None:
        """Peak level in dB, or None when the clip has no audio.

        Cheap, and it answers whether a clip's own sound is worth unmuting - a track
        that peaks near silence is wind and room tone.
        """
        result = subprocess.run(
            [
                str(self._ffmpeg),
                "-hide_banner",
                "-i",
                str(media),
                "-af",
                "volumedetect",
                "-vn",
                "-f",
                "null",
                "-",
            ],
            capture_output=True,
            text=True,
            check=False,
        )
        match = _MAX_VOLUME.search(result.stderr)
        return float(match.group(1)) if match else None


def sample_span(
    media: Path, ffmpeg: Path, *, start: float, end: float, count: int = 8
) -> list[np.ndarray]:
    """Decode a handful of frames spread across one stretch of a clip.

    Used by reframing, which needs to watch the subject move through the span that was
    actually chosen rather than through the whole file.
    """
    duration = max(0.0, end - start)
    if duration <= 0 or count < 1:
        return []

    rate = max(1.0, count / duration)
    with tempfile.TemporaryDirectory(prefix="genvai-span-") as scratch:
        directory = Path(scratch)
        command = [
            str(ffmpeg),
            "-hide_banner",
            "-loglevel",
            "error",
            "-ss",
            f"{start:.4f}",
            "-t",
            f"{duration:.4f}",
            "-i",
            str(media),
            "-vf",
            f"fps={rate},scale={WORK_WIDTH}:-2",
            "-frames:v",
            str(count),
            "-f",
            "image2",
            str(directory / "%03d.png"),
        ]
        if subprocess.run(command, capture_output=True, check=False).returncode != 0:
            return []
        return [_as_array(p) for p in sorted(directory.glob("*.png"))]


def _load_image(path: Path) -> list[np.ndarray]:
    try:
        with Image.open(path) as image:
            thumbnail = image.convert("RGB")
            width = WORK_WIDTH
            height = max(1, round(thumbnail.height * width / thumbnail.width))
            return [np.asarray(thumbnail.resize((width, height)))]
    except Exception:  # noqa: BLE001 - one unreadable file must not fail the import
        return []


def _as_array(path: Path) -> np.ndarray:
    with Image.open(path) as image:
        return np.asarray(image.convert("RGB"))
