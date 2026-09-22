"""Shared fixtures.

Render tests need real media. Rather than committing binaries, fixtures are synthesised
once per session with ffmpeg and Pillow - a few seconds, and it keeps the repository
free of megabytes of sample footage.
"""

import subprocess
from pathlib import Path

import pytest
from PIL import Image, ImageDraw

from genvai.adapters.ffmpeg import resolve_ffmpeg


@pytest.fixture(scope="session")
def ffmpeg_binary() -> Path:
    return resolve_ffmpeg()


@pytest.fixture(scope="session")
def sample_media(tmp_path_factory: pytest.TempPathFactory, ffmpeg_binary: Path) -> dict[str, Path]:
    """A landscape photo and two distinct clips.

    Landscape on purpose: fitting wide source into a vertical canvas is the case that
    actually happens with a phone camera roll, and the one most likely to break.

    The clips use different patterns because identical content would hash to one asset -
    content addressing would dedupe them, which is correct behaviour and a useless
    fixture.
    """
    directory = tmp_path_factory.mktemp("media")

    image = Image.new("RGB", (1600, 1200), (24, 34, 58))
    draw = ImageDraw.Draw(image)
    for y in range(0, 1200, 40):
        draw.rectangle([0, y, 1600, y + 20], fill=(40, 60, 100))
    draw.ellipse([600, 400, 1000, 800], fill=(230, 180, 60))
    photo = directory / "photo.jpg"
    image.save(photo, quality=92)

    clips = {}
    for name, pattern, tone in (
        ("clip_a", "testsrc2=size=1280x720:rate=30:duration=8", 440),
        ("clip_b", "smptebars=size=1280x720:rate=30:duration=8", 660),
    ):
        path = directory / f"{name}.mp4"
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
                pattern,
                "-f",
                "lavfi",
                "-i",
                f"sine=frequency={tone}:duration=8",
                "-c:v",
                "libx264",
                "-pix_fmt",
                "yuv420p",
                "-c:a",
                "aac",
                str(path),
            ],
            check=True,
        )
        clips[name] = path

    return {"photo": photo, **clips}
