"""Locating a usable TrueType font.

Text has to render the same whether it is drawn by Pillow or burned in by ffmpeg, so
both go through here. Styles name a font family; this resolves it to a file, falling
back through a list of faces that ship with common systems.

A missing font is not an error. Text still renders, just in whatever was found.
"""

import sys
from functools import lru_cache
from pathlib import Path

_SEARCH_DIRS: tuple[Path, ...] = (
    Path("C:/Windows/Fonts"),
    Path("/usr/share/fonts"),
    Path("/usr/local/share/fonts"),
    Path.home() / ".fonts",
    Path("/System/Library/Fonts"),
    Path("/Library/Fonts"),
)

_FALLBACKS: tuple[str, ...] = (
    "Inter-SemiBold",
    "Inter-Bold",
    "SegoeUI-Bold",
    "seguibl",
    "segoeuib",
    "arialbd",
    "Arial-Bold",
    "DejaVuSans-Bold",
    "LiberationSans-Bold",
    "NotoSans-Bold",
    "Helvetica",
    "arial",
    "DejaVuSans",
    "segoeui",
)


@lru_cache(maxsize=64)
def resolve(family: str | None = None) -> Path | None:
    """Find a font file for `family`, else the best available fallback.

    Matching is loose - families are named in styles as "Inter-SemiBold" but land on
    disk as "Inter-SemiBold.ttf", "inter-semibold.otf" or not at all. Returns None only
    when no usable font exists anywhere, which callers treat as "use the default".
    """
    wanted = [family, *(_FALLBACKS)] if family else list(_FALLBACKS)
    for name in wanted:
        if not name:
            continue
        found = _find(name)
        if found is not None:
            return found
    return _any_font()


@lru_cache(maxsize=1)
def _font_files() -> tuple[Path, ...]:
    files: list[Path] = []
    for directory in _SEARCH_DIRS:
        if not directory.exists():
            continue
        try:
            files.extend(p for p in directory.rglob("*") if p.suffix.lower() in (".ttf", ".otf"))
        except OSError:
            continue
    return tuple(files)


def _find(name: str) -> Path | None:
    target = name.lower().replace("-", "").replace(" ", "")
    for path in _font_files():
        if path.stem.lower().replace("-", "").replace(" ", "") == target:
            return path
    return None


def _any_font() -> Path | None:
    files = _font_files()
    return files[0] if files else None


def for_ffmpeg(path: Path) -> str:
    """Escape a font path for use inside an ffmpeg filtergraph.

    Windows paths are the awkward case: a drive colon reads as an option separator and
    backslashes read as escapes, so `C:\\Windows\\Fonts\\arial.ttf` has to become
    `C\\:/Windows/Fonts/arial.ttf` before ffmpeg will accept it.
    """
    text = path.as_posix()
    if sys.platform == "win32":
        text = text.replace(":", r"\:")
    return text
