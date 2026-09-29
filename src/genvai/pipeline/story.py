"""Poster -> storyboard -> storybook -> reel.

Three steps, each one a command, because each ends at a point where a person should
look before more is spent:

1. `draft`  reads the artwork into material (clean plates, a cast, the brand colours),
            draws a storyboard from a story arc, and saves it as the project's next
            storyboard version.
2. `storybook` renders one key frame per shot and writes an HTML page to review.
3. `shoot`  renders every shot and joins them. Shots are cached by content, so after
            editing one shot in the storyboard only that shot is drawn again.

A project's storyboards live in `story/`, beside the timelines: `story/v1.json`, ...,
`story/plates/` for the posters with their words erased, and `story/frames/` for the
storybook's key frames.
"""

import hashlib
import json
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image

from genvai.cast import Picture, find_cast
from genvai.errors import GenvaiError
from genvai.ocr import Box, text_rects
from genvai.palette import type_palette
from genvai.ports import FaceDetector, LLMPort, ProjectStore, ShotRenderer
from genvai.promo import Brief
from genvai.regions import erase_text
from genvai.stories import Arc, Copy
from genvai.stories import draft as draft_board
from genvai.storyboard import FPS, Storyboard
from genvai.storybook import page
from genvai.timeline import AssetProvenance


@dataclass(frozen=True)
class Material:
    """What a storyboard is drawn from, read once off the artwork."""

    posters: tuple[str, ...]
    plates: tuple[Image.Image, ...]
    originals: tuple[Image.Image, ...]
    cast: tuple[Picture, ...]
    palette: tuple[str, str, str]


def read_material(
    project_id: str,
    store: ProjectStore,
    posters: tuple[Path, ...],
    boxes: tuple[tuple[Box, ...], ...],
    faces: FaceDetector | None,
) -> Material:
    """Store the posters, erase their words, find their pictures and their colours.

    `boxes` are the OCR boxes already read for the brief - one tuple per poster - so the
    artwork is read once, not twice.
    """
    directory = _story_dir(store, project_id) / "plates"
    directory.mkdir(parents=True, exist_ok=True)
    detector = faces if faces is not None and faces.is_available() else None

    stored: list[str] = []
    plates: list[Image.Image] = []
    originals: list[Image.Image] = []
    cast: list[Picture] = []
    palette: tuple[str, str, str] | None = None

    for index, (poster, poster_boxes) in enumerate(zip(posters, boxes, strict=True)):
        asset = store.store_asset(
            project_id, poster, "image", AssetProvenance(provider="user", prompt=poster.name)
        )
        stored.append(asset.path)
        with Image.open(poster) as opened:
            original = opened.convert("RGB")
        array = np.asarray(original)
        rects = text_rects(list(poster_boxes), *original.size)
        clean, stubborn = erase_text(array, rects)
        plate = Image.fromarray(clean, "RGB")
        plate.save(directory / f"{index}.png")
        found = detector.detect(array) if detector else ()
        letter = chr(ord("a") + index)
        cast += [
            p.model_copy(update={"id": f"{letter}{p.id[1:]}", "poster": index})
            for p in find_cast(clean, boxes=poster_boxes, faces=found, stubborn=stubborn)
        ]
        if palette is None:
            colours = type_palette(array, rects)
            palette = (_hex(colours.deep), _hex(colours.light), _hex(colours.accent))
        plates.append(plate)
        originals.append(original)

    return Material(
        posters=tuple(stored),
        plates=tuple(plates),
        originals=tuple(originals),
        cast=tuple(cast),
        palette=palette or ("#3A0F24", "#FFE9D6", "#C2185B"),
    )


def draft(
    project_id: str,
    store: ProjectStore,
    material: Material,
    brief: Brief,
    *,
    arc: Arc,
    seed: int,
    llm: LLMPort | None = None,
    on_note: Callable[[str], None] | None = None,
) -> Storyboard:
    """Draw a storyboard and save it as the next version."""
    copy = write_copy(llm, brief, arc, on_note=on_note) if llm is not None else None
    board = draft_board(
        brief,
        material.cast,
        material.posters,
        arc=arc,
        palette=material.palette,
        seed=seed,
        copy=copy,
    )
    return save(store, project_id, board)


def write_copy(
    llm: LLMPort,
    brief: Brief,
    arc: Arc,
    *,
    on_note: Callable[[str], None] | None = None,
) -> Copy | None:
    """Three short lines from the model; the arc's own lines if it cannot manage them.

    Only copy is asked for. Structure - which shot, how long, what cut - stays with the
    arc: a 3B model asked for a multi-field plan drifts, and that was measured in M3.
    """
    defaults = arc.copy_defaults
    prompt = (
        f"Write three short lines for a {arc.name} Instagram reel. "
        f"Business: {brief.business or 'a beauty salon'}. "
        f"Occasion: {brief.occasion or 'an offer'}. "
        f"Mood: {arc.blurb}. Each line at most 22 characters, in capitals, no emoji, no "
        "hashtags, no prices, and do not name the festival - it is already on screen. "
        f'Examples of the style: hook "{defaults.hook}", promise "{defaults.promise}", '
        f'cta "{defaults.cta}". Write new ones in that style.'
    )
    try:
        written = llm.structured(prompt, Copy, max_retries=2)
    except Exception as exc:  # noqa: BLE001 - copy is decoration; never lose the reel to it
        if on_note:
            on_note(f"The model could not write copy ({exc}); using the arc's own lines.")
        return None
    banned = _stems(f"{brief.occasion} {brief.business}")
    return Copy(
        hook=_clean_line(written.hook, banned),
        promise=_clean_line(written.promise, banned),
        cta=_clean_line(written.cta, banned),
    )


def storybook(project_id: str, store: ProjectStore, compositor: ShotRenderer) -> Path:
    """Render a key frame per shot and write the review page. Returns the page's path."""
    board = load(store, project_id)
    plates, originals = _pictures(store, project_id, board)
    frames_dir = _story_dir(store, project_id) / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)

    frames: dict[str, str] = {}
    for shot in board.shots:
        stage = compositor.stage(board, shot, plates, originals)
        moment = _key_moment(board, shot)
        image = compositor.frame(stage, moment, cut_in="cut", cut_out=None)
        name = f"v{board.version}-{shot.id}.jpg"
        image.save(frames_dir / name, quality=88)
        frames[shot.id] = f"frames/{name}"

    out = _story_dir(store, project_id) / "storybook.html"
    out.write_text(page(board, frames), encoding="utf-8", newline="\n")
    return out


def shoot(
    project_id: str,
    store: ProjectStore,
    compositor: ShotRenderer,
    *,
    track: Path | None = None,
    on_shot: Callable[[str, int, int], None] | None = None,
) -> dict[str, Path]:
    """Render the storyboard. Returns the silent cut, and the full one with a track."""
    board = load(store, project_id)
    plates, originals = _pictures(store, project_id, board)
    cache = store.project_dir(project_id) / "cache" / "story"
    cache.mkdir(parents=True, exist_ok=True)

    segments: list[Path] = []
    elapsed = 0.0
    for index, shot in enumerate(board.shots):
        start = round(elapsed * FPS)
        elapsed += board.seconds(shot)
        frames = round(elapsed * FPS) - start
        cut_out = board.shots[index + 1].cut if index + 1 < len(board.shots) else None
        key = _shot_key(board, index, compositor, plates)
        out = cache / f"{key}.mp4"
        if on_shot:
            on_shot(shot.id, index + 1, len(board.shots))
        if not out.exists():
            stage = compositor.stage(board, shot, plates, originals)
            compositor.render_shot(
                stage, out, frames=frames, cut_in=shot.cut if index else "cut", cut_out=cut_out
            )
        segments.append(out)

    renders = store.renders_dir(project_id)
    stem = f"story-v{board.version}-{board.arc}"
    silent = compositor.join(tuple(segments), renders / f"{stem}-silent.mp4")
    outputs = {"silent": silent}
    if track is not None:
        outputs["full"] = compositor.with_music(
            silent, track, renders / f"{stem}.mp4", seconds=board.duration
        )
    return outputs


# ------------------------------------------------------------------------ storage


def save(store: ProjectStore, project_id: str, board: Storyboard) -> Storyboard:
    directory = _story_dir(store, project_id)
    directory.mkdir(parents=True, exist_ok=True)
    versions = _versions(directory)
    numbered = board.model_copy(update={"version": (max(versions) + 1) if versions else 1})
    (directory / f"v{numbered.version}.json").write_text(
        numbered.model_dump_json(indent=2), encoding="utf-8", newline="\n"
    )
    return numbered


def load(store: ProjectStore, project_id: str, version: int | None = None) -> Storyboard:
    directory = _story_dir(store, project_id)
    versions = _versions(directory)
    if not versions:
        raise GenvaiError(f"'{project_id}' has no storyboard yet. Draft one with: genvai story")
    wanted = version or max(versions)
    path = directory / f"v{wanted}.json"
    if not path.exists():
        raise GenvaiError(f"No storyboard v{wanted}. There are: {sorted(versions)}")
    return Storyboard.model_validate_json(path.read_text(encoding="utf-8"))


def _versions(directory: Path) -> list[int]:
    return (
        [int(p.stem[1:]) for p in directory.glob("v*.json") if p.stem[1:].isdigit()]
        if directory.exists()
        else []
    )


def _story_dir(store: ProjectStore, project_id: str) -> Path:
    return store.project_dir(project_id) / "story"


def _pictures(
    store: ProjectStore, project_id: str, board: Storyboard
) -> tuple[tuple[Image.Image, ...], tuple[Image.Image, ...]]:
    root = store.project_dir(project_id)
    plates_dir = _story_dir(store, project_id) / "plates"
    plates, originals = [], []
    for index, poster in enumerate(board.posters):
        with Image.open(root / poster) as opened:
            originals.append(opened.convert("RGB"))
        plate = plates_dir / f"{index}.png"
        with Image.open(plate if plate.exists() else root / poster) as opened:
            plates.append(opened.convert("RGB"))
    return tuple(plates), tuple(originals)


def _shot_key(
    board: Storyboard, index: int, compositor: ShotRenderer, plates: tuple[Image.Image, ...]
) -> str:
    """Everything that changes a shot's pixels, hashed. Change any of it, redraw it."""
    shot = board.shots[index]
    following = board.shots[index + 1].cut if index + 1 < len(board.shots) else None
    picture = board.picture(shot.picture)
    plate = plates[picture.poster] if picture else None
    payload = {
        "shot": shot.model_dump(mode="json"),
        "picture": picture.model_dump(mode="json") if picture else None,
        "plate": hashlib.sha256(plate.tobytes()).hexdigest()[:16] if plate else None,
        "posters": board.posters if shot.layout == "poster" else None,
        "palette": board.palette,
        "bpm": board.bpm,
        "seed": board.seed,
        "first": index == 0,
        "next": following,
        "canvas": compositor.canvas,
        "engine": ENGINE_VERSION,
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()[:24]


ENGINE_VERSION = 2
"""Bumped when the compositor's drawing changes, so cached shots are not reused stale."""


def _key_moment(board: Storyboard, shot) -> float:  # noqa: ANN001
    """When to take a shot's key frame: once its last caption has fully arrived."""
    seconds = board.seconds(shot)
    last = max((c.at for c in shot.captions), default=0.0) * board.beat
    return min(seconds * 0.92, max(seconds * 0.55, last + 0.6))


def _clean_line(text: str, banned: set[str] = frozenset()) -> str:  # type: ignore[assignment]
    """A usable line, or empty to fall back to the arc's own.

    Rejected: too long, any digit (a model's number is an invented price), or naming the
    occasion or business - those are already on screen, and a 3B model repeating them
    measured as "NAVRAATRI IS HERE!" under "NAVRATRI IS HERE".
    """
    line = " ".join(text.replace("#", "").split()).upper()
    # A line exactly at the cap was cut there: "BOOK NOW BEFORE IT'S GON".
    if not 2 <= len(line) < 24 or any(c.isdigit() for c in line):
        return ""
    if any(claim in f" {line} " for claim in CLAIMS):
        return ""
    return "" if _stems(line) & banned else line


CLAIMS = (
    " SOLD OUT",
    " GONE",
    " LIMITED",
    " HURRY",
    " LAST ",
    " ONLY ",
    " FREE",
    " SALE",
    " DISCOUNT",
    " % ",
    " GUARANTEE",
    " BEST IN ",
)
"""Promises the poster does not make. A 3B model reaches for scarcity - it wrote "BOOK NOW
BEFORE SOLD OUT" for a salon with no stated limit - and a reel must not claim what the
client did not."""


def _stems(text: str) -> set[str]:
    """First five letters of each long word: loose enough to catch a misspelling."""
    return {
        w[:5]
        for w in "".join(c if c.isalpha() else " " for c in text.lower()).split()
        if len(w) > 4
    }


def _hex(colour: tuple[int, int, int]) -> str:
    return "#{:02X}{:02X}{:02X}".format(*colour)
