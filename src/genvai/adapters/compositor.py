"""Storyboard shots to frames, drawn one at a time and piped to ffmpeg.

The timeline renderer describes each scene as an ffmpeg filtergraph, which is right for
crops, trims and fades over footage. A storyboard asks for more: a picture floating as a
card over a blurred copy of itself, layers moving at different speeds, petals drifting in
front, a price that pops on the beat, a whip pan between shots. Expressed as filters that
is unreadable; expressed as a function from time to an image it is ordinary drawing.

So each shot is prepared once - its background, its card, its text sprites, its particles
- and then drawn frame by frame with Pillow and numpy, and the raw frames piped to x264.

Poster imagery is small: a photo tile is about 135 px across. Depth is what makes it read
as cinema rather than a blown-up thumbnail - a sharp card over a soft background, moving
against it - and a little grain hides the softness a 4x enlargement cannot avoid.
"""

import math
import random
import subprocess
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
from PIL import Image, ImageChops, ImageDraw, ImageFilter, ImageFont

from genvai.errors import RenderError
from genvai.fonts import resolve as resolve_font
from genvai.storyboard import (
    FPS,
    TRANSITION,
    Caption,
    Cut,
    Shot,
    Storyboard,
    appears,
    transition_seconds,
)

SAFE_TOP = 0.10
SAFE_BOTTOM = 0.80
SAFE_LEFT = 0.05
SAFE_RIGHT = 0.82
"""Where text and cards may sit. Reels covers the bottom fifth with the caption and the
right edge with its buttons; the centre line is therefore left of the frame's centre."""

CENTRE_X = (SAFE_LEFT + SAFE_RIGHT) / 2

MAX_ENLARGE = 5.0
"""A picture is never shown at more than this multiple of its size on the poster."""


MIN_CONTRAST = 4.5
"""The WCAG ratio for body text. A caption behind which the frame is lighter than this
allows gets a plate. Measured before plates existed: the white kicker over the pale
dancer backdrop scored 1.1:1 - there, but unreadable."""

PLATE_ALPHA = 0.72
"""How opaque a caption's plate is. Enough that white text clears 4.5:1 over pure white
paper behind it, measured, while the picture still shows through at the edges."""


@dataclass(frozen=True)
class Stage:
    """Everything a shot needs that does not change from frame to frame."""

    width: int
    height: int
    seconds: float
    move: str
    backdrop: Image.Image
    card: Image.Image | None
    card_centre: tuple[float, float]
    board: Image.Image | None
    zoom_from: tuple[float, float, float, float] | None
    sprites: tuple["_Sprite", ...]
    particles: tuple["_Particle", ...]
    petal_images: tuple[Image.Image, ...]
    bokeh_images: tuple[Image.Image, ...]
    vignette: Image.Image | None
    leak: Image.Image | None
    grain: tuple[np.ndarray, ...]


@dataclass(frozen=True)
class _Sprite:
    caption: Caption
    image: Image.Image
    centre: tuple[float, float]
    appears: float
    prefixes: tuple[Image.Image, ...]
    contrast: float = 21.0
    treatment: str = "as designed"


@dataclass(frozen=True)
class _Particle:
    kind: str
    x: float
    y: float
    speed: float
    sway: float
    phase: float
    size: int
    spin: int


class Compositor:
    """Implements `ShotRenderer`. Knows nothing about projects: callers pass pictures in."""

    name = "compositor"

    def __init__(self, ffmpeg: Path, *, width: int = 1080, height: int = 1920) -> None:
        self._ffmpeg = ffmpeg
        self._width = width
        self._height = height
        self._bold = resolve_font("seguibl")
        self._semibold = resolve_font("segoeuib")

    @property
    def canvas(self) -> tuple[int, int]:
        return self._width, self._height

    # ---------------------------------------------------------------------- public

    def stage(
        self,
        board: Storyboard,
        shot: Shot,
        plates: tuple[Image.Image, ...],
        posters: tuple[Image.Image, ...],
    ) -> Stage:
        """Prepare a shot. `plates` are the posters with their words erased; `posters`
        are the originals, shown whole in a poster shot."""
        return _prepare(
            board, shot, plates, posters, self._width, self._height, self._bold, self._semibold
        )

    def frame(self, stage: Stage, t: float, *, cut_in: Cut, cut_out: Cut | None) -> Image.Image:
        return _draw(stage, t, cut_in, cut_out)

    def legibility(self, stage: Stage) -> tuple[tuple[str, str, float, str], ...]:
        """Each caption's text, role, contrast ratio against what is behind it, and what
        was done to get it there: "as designed", "dark ink" or "plate"."""
        return tuple(
            (s.caption.text, s.caption.role, s.contrast, s.treatment) for s in stage.sprites
        )

    def render_shot(
        self, stage: Stage, out: Path, *, frames: int, cut_in: Cut, cut_out: Cut | None
    ) -> Path:
        """Encode one shot. `frames` is given by the caller so shots add up exactly."""
        out.parent.mkdir(parents=True, exist_ok=True)
        command = [
            str(self._ffmpeg),
            "-y",
            "-hide_banner",
            "-loglevel",
            "error",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "rgb24",
            "-s",
            f"{self._width}x{self._height}",
            "-r",
            str(FPS),
            "-i",
            "-",
            "-c:v",
            "libx264",
            "-preset",
            "medium",
            "-crf",
            "18",
            "-pix_fmt",
            "yuv420p",
            str(out),
        ]
        process = subprocess.Popen(command, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
        assert process.stdin is not None
        try:
            for index in range(frames):
                image = _draw(stage, index / FPS, cut_in, cut_out)
                process.stdin.write(image.tobytes())
            process.stdin.close()
        except BrokenPipeError:
            pass
        error = process.stderr.read().decode(errors="replace") if process.stderr else ""
        if process.wait() != 0:
            raise RenderError(f"encoding shot {out.stem}", stderr=error)
        return out

    def join(self, segments: tuple[Path, ...], out: Path) -> Path:
        """Concatenate shot segments. Same encoder settings, so a stream copy."""
        listing = out.with_suffix(".txt")
        listing.write_text(
            "".join(f"file '{p.resolve().as_posix()}'\n" for p in segments),
            encoding="utf-8",
            newline="\n",
        )
        self._run(
            [
                "-f",
                "concat",
                "-safe",
                "0",
                "-i",
                str(listing),
                "-c",
                "copy",
                "-movflags",
                "+faststart",
                str(out),
            ],
            "joining shots",
        )
        listing.unlink(missing_ok=True)
        return out

    def with_music(self, video: Path, track: Path, out: Path, *, seconds: float) -> Path:
        """Lay a track under the cut, faded out at the end. The track is the only sound."""
        fade = max(0.0, seconds - 1.5)
        self._run(
            [
                "-i",
                str(video),
                "-i",
                str(track),
                "-map",
                "0:v",
                "-map",
                "1:a",
                "-c:v",
                "copy",
                "-c:a",
                "aac",
                "-b:a",
                "192k",
                "-af",
                f"afade=t=in:d=0.3,afade=t=out:st={fade:.3f}:d=1.5",
                "-shortest",
                "-movflags",
                "+faststart",
                str(out),
            ],
            "adding music",
        )
        return out

    def _run(self, arguments: list[str], doing: str) -> None:
        result = subprocess.run(
            [str(self._ffmpeg), "-y", "-hide_banner", "-loglevel", "error", *arguments],
            capture_output=True,
            text=True,
            check=False,
        )
        if result.returncode != 0:
            raise RenderError(doing, stderr=result.stderr)


# ----------------------------------------------------------------------- preparing


def _prepare(
    board: Storyboard,
    shot: Shot,
    plates: tuple[Image.Image, ...],
    posters: tuple[Image.Image, ...],
    width: int,
    height: int,
    bold: Path | None,
    semibold: Path | None,
) -> Stage:
    deep, light, accent = (_rgb(c) for c in board.palette)
    rng = random.Random(f"{board.seed}-{shot.id}")
    picture = board.picture(shot.picture)
    plate = plates[picture.poster] if picture and picture.poster < len(plates) else None
    crop = _crop(plate, picture.rect, picture.face, shot.framing) if plate and picture else None

    card = None
    card_centre = (CENTRE_X, 0.42)
    board_image = None
    zoom_from = None

    if shot.layout == "full" and crop is not None:
        backdrop = _sharpened(_cover(crop, int(width * 1.2), int(height * 1.2)))
        backdrop = _tinted(backdrop, deep, 0.18)
    elif shot.layout == "card" and crop is not None:
        backdrop = _tinted(
            _cover(crop, int(width * 1.2), int(height * 1.2)).filter(
                ImageFilter.GaussianBlur(width // 28)
            ),
            deep,
            0.5,
        )
        card = _card(crop, width, height, accent)
    elif shot.layout == "poster" and posters:
        index = picture.poster if picture else 0
        poster = posters[min(index, len(posters) - 1)]
        board_image, placed = _poster_board(poster, width, height, deep, light, accent)
        backdrop = board_image
        if picture is not None and shot.move == "pull_out":
            zoom_from = _focus_window(picture.rect, placed, width, height)
    else:
        backdrop = _title_backdrop(width, height, deep, accent)

    def make(caption: Caption, treatment: str) -> _Sprite:
        return _sprite(
            caption, shot, board, width, height, deep, light, accent, bold, semibold, treatment
        )

    sprites = tuple(make(c, "as designed") for c in shot.captions)

    particles: list[_Particle] = []
    if "petals" in shot.effects:
        particles += [_particle("petal", rng) for _ in range(22)]
    if "bokeh" in shot.effects:
        particles += [_particle("bokeh", rng) for _ in range(12)]

    stage = Stage(
        width=width,
        height=height,
        seconds=board.seconds(shot),
        move=shot.move,
        backdrop=backdrop,
        card=card,
        card_centre=card_centre,
        board=board_image,
        zoom_from=zoom_from,
        sprites=sprites,
        particles=tuple(particles),
        petal_images=_petals(accent, width) if "petals" in shot.effects else (),
        bokeh_images=_bokeh(light, accent, width) if "bokeh" in shot.effects else (),
        vignette=_vignette(width, height) if "vignette" in shot.effects else None,
        leak=_leak(width, height, accent) if "light_leak" in shot.effects else None,
        grain=_grain(width, height, rng) if "grain" in shot.effects else (),
    )
    return _legible(stage, make, deep)


def _legible(stage: Stage, make, deep: tuple[int, int, int]) -> Stage:  # noqa: ANN001
    """Measure every caption against the frame behind it, and fix the ones that fail.

    The frame is drawn without text at two moments - as the caption arrives and at the
    end of the shot, since the camera keeps moving. Light text is judged against the
    brightest tenth of the ground under it, dark text against the darkest tenth.

    Fixes are tried in order of how little they change the design: the text as drawn;
    the same text in the brand's deep colour, which on a pale scene reads better than
    any box; and last a plate behind it.
    """
    bare = replace(stage, sprites=(), grain=())
    checked: list[_Sprite] = []
    for sprite in stage.sprites:
        if sprite.caption.role == "price":
            checked.append(sprite)  # set on its own pill: contrast is fixed by design
            continue
        moments = (min(stage.seconds * 0.95, sprite.appears + 0.3), stage.seconds * 0.95)
        grounds = [_ground(bare, sprite, t) for t in moments]
        brightest = max(g[1] for g in grounds)
        darkest = min(g[0] for g in grounds)

        contrast = _ratio(_text_luminance(sprite), brightest)
        if contrast >= MIN_CONTRAST:
            checked.append(replace(sprite, contrast=contrast))
            continue

        inked = make(sprite.caption, "dark ink")
        dark = _ratio(_text_luminance(inked, darkest_fill=True), darkest)
        if dark >= MIN_CONTRAST:
            checked.append(replace(inked, contrast=dark, treatment="dark ink"))
            continue

        plated_ground = _luminance_of(
            tuple(PLATE_ALPHA * d + (1 - PLATE_ALPHA) * 255 * _to_srgb(brightest) for d in deep)
        )
        checked.append(
            replace(
                make(sprite.caption, "plate"),
                contrast=_ratio(_text_luminance(sprite), plated_ground),
                treatment="plate",
            )
        )
    return replace(stage, sprites=tuple(checked))


def _ground(bare: Stage, sprite: _Sprite, t: float) -> tuple[float, float]:
    """Relative luminance of the darkest and brightest tenths of the frame under a
    caption."""
    frame = _draw(bare, t, "cut", None)
    w, h = sprite.image.size
    cx, cy = sprite.centre[0] * bare.width, sprite.centre[1] * bare.height
    box = (
        max(0, int(cx - w / 2)),
        max(0, int(cy - h / 2)),
        min(bare.width, int(cx + w / 2)),
        min(bare.height, int(cy + h / 2)),
    )
    pixels = np.asarray(frame.crop(box), dtype=np.float64).reshape(-1, 3) / 255.0
    if not pixels.size:
        return 0.0, 0.0
    linear = np.where(pixels <= 0.04045, pixels / 12.92, ((pixels + 0.055) / 1.055) ** 2.4)
    luminance = linear @ np.array([0.2126, 0.7152, 0.0722])
    return float(np.percentile(luminance, 10)), float(np.percentile(luminance, 90))


def _text_luminance(sprite: _Sprite, *, darkest_fill: bool = False) -> float:
    """The luminance of the caption's fill: its most opaque pixels, taking the brightest
    fifth for light text or the darkest fifth for dark - which leaves out the stroke."""
    array = np.asarray(sprite.image, dtype=np.float64)
    solid = array[array[:, :, 3] > 250][:, :3]
    if not solid.size:
        return 1.0
    sums = solid.sum(axis=1)
    fill = (
        solid[sums <= np.percentile(sums, 20)]
        if darkest_fill
        else solid[sums >= np.percentile(sums, 80)]
    )
    return _luminance_of(tuple(fill.mean(axis=0)))


def _luminance_of(colour: tuple[float, ...]) -> float:
    rgb = np.array(colour[:3], dtype=np.float64) / 255.0
    linear = np.where(rgb <= 0.04045, rgb / 12.92, ((rgb + 0.055) / 1.055) ** 2.4)
    return float(linear @ np.array([0.2126, 0.7152, 0.0722]))


def _to_srgb(luminance: float) -> float:
    """Luminance back to an sRGB fraction, near enough to blend a plate over it."""
    return 12.92 * luminance if luminance <= 0.0031308 else 1.055 * luminance ** (1 / 2.4) - 0.055


def _ratio(a: float, b: float) -> float:
    high, low = max(a, b), min(a, b)
    return (high + 0.05) / (low + 0.05)


def _crop(
    plate: Image.Image,
    rect: tuple[float, float, float, float],
    face: tuple[float, float, float, float] | None,
    framing: str,
) -> Image.Image:
    """The picture, framed. Detail goes in on the face when there is one."""
    width, height = plate.size
    x, y, w, h = rect
    if framing == "detail" and face is not None:
        fx, fy, fw, fh = face
        side = max(fw * width, fh * height) * 2.0
        cx, cy = (fx + fw / 2) * width, (fy + fh / 2) * height
        box = (cx - side / 2, cy - side * 0.45, cx + side / 2, cy + side * 0.55)
    else:
        # Never wider than the picture: its edges were trimmed of type for a reason.
        grow = {"wide": 0.0, "medium": 0.0, "close": -0.03, "detail": -0.12}.get(framing, 0.0)
        box = (
            (x - w * grow) * width,
            (y - h * grow) * height,
            (x + w * (1 + grow)) * width,
            (y + h * (1 + grow)) * height,
        )
    left, top = max(0, int(box[0])), max(0, int(box[1]))
    right, bottom = min(width, int(box[2])), min(height, int(box[3]))
    return plate.crop((left, top, max(left + 2, right), max(top + 2, bottom)))


def _card(crop: Image.Image, width: int, height: int, accent: tuple[int, int, int]) -> Image.Image:
    """The picture as a floating card: rounded, bordered, with a soft shadow."""
    box_w = width * (SAFE_RIGHT - SAFE_LEFT) * 0.92
    box_h = height * 0.40
    scale = min(box_w / crop.width, box_h / crop.height, MAX_ENLARGE)
    size = (max(2, round(crop.width * scale)), max(2, round(crop.height * scale)))
    picture = _sharpened(crop.resize(size, Image.LANCZOS))

    radius = max(8, width // 30)
    border = max(3, width // 200)
    pad = width // 14
    card = Image.new("RGBA", (size[0] + pad * 2, size[1] + pad * 2), (0, 0, 0, 0))

    shadow = Image.new("L", card.size, 0)
    ImageDraw.Draw(shadow).rounded_rectangle(
        [pad, pad + pad // 3, pad + size[0], pad + size[1] + pad // 3], radius, fill=150
    )
    shadow = shadow.filter(ImageFilter.GaussianBlur(pad // 2))
    card.paste((0, 0, 0, 255), (0, 0), shadow)

    mask = Image.new("L", size, 0)
    ImageDraw.Draw(mask).rounded_rectangle([0, 0, size[0] - 1, size[1] - 1], radius, fill=255)
    card.paste(picture.convert("RGBA"), (pad, pad), mask)
    ImageDraw.Draw(card).rounded_rectangle(
        [pad, pad, pad + size[0] - 1, pad + size[1] - 1], radius, outline=accent, width=border
    )
    return card


def _poster_board(
    poster: Image.Image,
    width: int,
    height: int,
    deep: tuple[int, int, int],
    light: tuple[int, int, int],
    accent: tuple[int, int, int],
) -> tuple[Image.Image, tuple[int, int, int, int]]:
    """The whole poster on a pale card, placed inside the safe area."""
    board = _gradient(width, height, light, _mix(light, deep, 0.18))
    box_w = width * (SAFE_RIGHT - SAFE_LEFT)
    box_h = height * (SAFE_BOTTOM - SAFE_TOP - 0.12)
    scale = min(box_w / poster.width, box_h / poster.height)
    size = (round(poster.width * scale), round(poster.height * scale))
    art = poster.resize(size, Image.LANCZOS)
    left = round(width * CENTRE_X - size[0] / 2)
    top = round(height * (SAFE_TOP + 0.07))
    shadow = Image.new("L", (width, height), 0)
    ImageDraw.Draw(shadow).rectangle(
        [left + 6, top + 14, left + size[0] + 6, top + size[1] + 14], fill=120
    )
    board.paste(deep, (0, 0), shadow.filter(ImageFilter.GaussianBlur(width // 60)))
    board.paste(art, (left, top))
    ImageDraw.Draw(board).rectangle(
        [left - 2, top - 2, left + size[0] + 1, top + size[1] + 1], outline=accent, width=3
    )
    return board, (left, top, size[0], size[1])


def _focus_window(
    rect: tuple[float, float, float, float],
    placed: tuple[int, int, int, int],
    width: int,
    height: int,
) -> tuple[float, float, float, float]:
    """Where on the board a picture sits, widened to the frame's shape, as fractions."""
    left, top, w, h = placed
    x = left + rect[0] * w
    y = top + rect[1] * h
    pw, ph = rect[2] * w, rect[3] * h
    aspect = width / height
    if pw / ph > aspect:
        ph = pw / aspect
    else:
        pw = ph * aspect
    pw, ph = pw * 1.3, ph * 1.3
    cx, cy = x + rect[2] * w / 2, y + rect[3] * h / 2
    fx = min(max(0.0, cx - pw / 2), width - pw) / width
    fy = min(max(0.0, cy - ph / 2), height - ph) / height
    return (fx, fy, pw / width, ph / height)


def _title_backdrop(
    width: int, height: int, deep: tuple[int, int, int], accent: tuple[int, int, int]
) -> Image.Image:
    base = _gradient(width, height, _mix(deep, (0, 0, 0), 0.1), _mix(deep, (0, 0, 0), 0.55))
    glow = Image.new("L", (width, height), 0)
    ImageDraw.Draw(glow).ellipse(
        [width * 0.05, height * 0.22, width * 0.85, height * 0.62], fill=110
    )
    glow = glow.filter(ImageFilter.GaussianBlur(width // 6))
    base.paste(accent, (0, 0), glow)
    return base


def _sprite(
    caption: Caption,
    shot: Shot,
    board: Storyboard,
    width: int,
    height: int,
    deep: tuple[int, int, int],
    light: tuple[int, int, int],
    accent: tuple[int, int, int],
    bold: Path | None,
    semibold: Path | None,
    treatment: str = "as designed",
) -> _Sprite:
    # Kicker and footer were 3.3% and 2.9% of the height: legible on a monitor, thin on a
    # phone held at arm's length. Raised to where a stroke still fits inside the letters.
    size_pct, font_path, fill, pill = {
        "kicker": (3.8, bold, light, False),
        "headline": (7.0, bold, (255, 255, 255), False),
        "price": (6.4, bold, (255, 255, 255), True),
        "footer": (3.3, bold, (255, 245, 240), False),
    }[caption.role]
    if shot.layout == "title" and caption.role == "kicker":
        size_pct = 4.2
    size = round(height * size_pct / 100)
    text = caption.text.upper() if caption.role in ("kicker", "price") else caption.text
    edge = deep
    if treatment == "dark ink" and not pill:
        fill, edge = deep, (255, 255, 255)
    render = _text_renderer(
        size,
        font_path,
        fill,
        edge,
        accent if pill else None,
        width,
        deep if treatment == "plate" else None,
        shadowed=treatment != "dark ink",
    )
    image = render(text)
    prefixes: tuple[Image.Image, ...] = ()
    if caption.entrance == "type":
        prefixes = tuple(render(text[:n]) for n in range(1, len(text) + 1))
    elif caption.entrance == "words":
        words = text.split()
        prefixes = tuple(render(" ".join(words[:n])) for n in range(1, len(words) + 1))
    return _Sprite(
        caption=caption,
        image=image,
        centre=(CENTRE_X, _row(caption.role, shot.layout)),
        appears=appears(caption, shot.cut, board.beat),
        prefixes=prefixes,
    )


def _row(role: str, layout: str) -> float:
    """Vertical centre, as a fraction of height, for each kind of line in each layout."""
    if layout == "title":
        return {"kicker": 0.35, "headline": 0.46, "price": 0.46, "footer": 0.60}[role]
    if layout == "poster":
        return {"kicker": 0.125, "headline": 0.74, "price": 0.74, "footer": 0.765}[role]
    if layout == "full":
        return {"kicker": 0.14, "headline": 0.60, "price": 0.66, "footer": 0.75}[role]
    return {"kicker": 0.155, "headline": 0.68, "price": 0.685, "footer": 0.775}[role]


def _text_renderer(
    size: int,
    font_path: Path | None,
    fill: tuple[int, int, int],
    edge: tuple[int, int, int],
    pill: tuple[int, int, int] | None,
    width: int,
    plate: tuple[int, int, int] | None = None,
    *,
    shadowed: bool = True,
):
    font = ImageFont.truetype(str(font_path), size) if font_path else ImageFont.load_default(size)
    stroke = 0 if pill else max(3, size // 9)
    limit = width * (SAFE_RIGHT - SAFE_LEFT) * 0.96

    def render(text: str) -> Image.Image:
        lines = _wrapped(text, font, limit, stroke)
        measure = ImageDraw.Draw(Image.new("L", (1, 1)))
        boxes = [measure.textbbox((0, 0), line, font=font, stroke_width=stroke) for line in lines]
        line_h = int(size * 1.18)
        text_w = max((b[2] - b[0]) for b in boxes) if boxes else 1
        pad_x, pad_y = (size // 2, size // 5) if pill else (stroke * 2, stroke * 2)
        shadow_pad = size // 5
        w = text_w + pad_x * 2 + shadow_pad * 2
        h = line_h * len(lines) + pad_y * 2 + shadow_pad * 2
        image = Image.new("RGBA", (w, h), (0, 0, 0, 0))
        draw = ImageDraw.Draw(image)
        if pill:
            draw.rounded_rectangle(
                [shadow_pad, shadow_pad, w - shadow_pad, h - shadow_pad],
                radius=size // 3,
                fill=(*pill, 255),
            )
        elif plate:
            # A soft plate in the brand's deep colour: the caption's own ground, carried
            # with it, so it reads over whatever the picture is doing behind.
            band = Image.new("L", (w, h), 0)
            ImageDraw.Draw(band).rounded_rectangle(
                [shadow_pad // 2, shadow_pad // 2, w - shadow_pad // 2, h - shadow_pad // 2],
                radius=size // 2,
                fill=int(255 * PLATE_ALPHA),
            )
            image.paste((*plate, 255), (0, 0), band.filter(ImageFilter.GaussianBlur(size // 10)))
        elif shadowed:
            shadow = Image.new("L", (w, h), 0)
            shadow_draw = ImageDraw.Draw(shadow)
            for i, line in enumerate(lines):
                lw = boxes[i][2] - boxes[i][0]
                shadow_draw.text(
                    ((w - lw) / 2 - boxes[i][0], shadow_pad + pad_y + i * line_h + size // 12),
                    line,
                    font=font,
                    fill=160,
                    stroke_width=stroke,
                )
            image.paste((0, 0, 0, 255), (0, 0), shadow.filter(ImageFilter.GaussianBlur(size // 8)))
        for i, line in enumerate(lines):
            lw = boxes[i][2] - boxes[i][0]
            draw.text(
                ((w - lw) / 2 - boxes[i][0], shadow_pad + pad_y + i * line_h - boxes[i][1] // 2),
                line,
                font=font,
                fill=(*fill, 255),
                stroke_width=stroke,
                stroke_fill=(*edge, 255),
            )
        return image

    return render


def _wrapped(text: str, font: ImageFont.FreeTypeFont, limit: float, stroke: int) -> list[str]:
    measure = ImageDraw.Draw(Image.new("L", (1, 1)))
    lines: list[str] = []
    for paragraph in text.split("\n"):
        current = ""
        for word in paragraph.split():
            candidate = f"{current} {word}".strip()
            if measure.textlength(candidate, font=font) + stroke * 2 <= limit or not current:
                current = candidate
            else:
                lines.append(current)
                current = word
        if current:
            lines.append(current)
    return lines or [""]


def _particle(kind: str, rng: random.Random) -> _Particle:
    return _Particle(
        kind=kind,
        x=rng.random(),
        y=rng.random(),
        speed=rng.uniform(0.05, 0.14) if kind == "petal" else rng.uniform(0.01, 0.035),
        sway=rng.uniform(0.01, 0.04),
        phase=rng.uniform(0, math.tau),
        size=rng.randrange(0, 3),
        spin=rng.randrange(0, 12),
    )


def _petals(accent: tuple[int, int, int], width: int) -> tuple[Image.Image, ...]:
    """A petal at twelve angles, in three sizes: 36 sprites, drawn once."""
    sprites: list[Image.Image] = []
    colours = ((244, 143, 177), (255, 196, 214), accent)
    for size_index, scale in enumerate((0.022, 0.03, 0.04)):
        w = max(6, int(width * scale))
        base = Image.new("RGBA", (w * 2, w * 2), (0, 0, 0, 0))
        ImageDraw.Draw(base).ellipse(
            [w * 0.5, w * 0.75, w * 1.5, w * 1.25], fill=(*colours[size_index], 215)
        )
        base = base.filter(ImageFilter.GaussianBlur(max(1, w // 10)))
        sprites += [base.rotate(angle * 30, resample=Image.BICUBIC) for angle in range(12)]
    return tuple(sprites)


def _bokeh(
    light: tuple[int, int, int], accent: tuple[int, int, int], width: int
) -> tuple[Image.Image, ...]:
    sprites = []
    for scale, colour in ((0.07, light), (0.11, accent), (0.16, light)):
        w = int(width * scale)
        image = Image.new("RGBA", (w * 2, w * 2), (0, 0, 0, 0))
        ImageDraw.Draw(image).ellipse([w * 0.4, w * 0.4, w * 1.6, w * 1.6], fill=(*colour, 70))
        sprites.append(image.filter(ImageFilter.GaussianBlur(w // 7)))
    return tuple(sprites)


def _vignette(width: int, height: int) -> Image.Image:
    ys, xs = np.mgrid[0:height, 0:width]
    d = np.sqrt(
        ((xs - width / 2) / (width * 0.62)) ** 2 + ((ys - height / 2) / (height * 0.6)) ** 2
    )
    alpha = (np.clip(d - 0.55, 0, 1) ** 1.6 * 200).astype(np.uint8)
    layer = Image.new("RGBA", (width, height), (0, 0, 0, 0))
    layer.putalpha(Image.fromarray(alpha, "L"))
    return layer


def _leak(width: int, height: int, accent: tuple[int, int, int]) -> Image.Image:
    """A warm glow, twice the frame, that drifts across during the shot."""
    leak = Image.new("RGB", (width * 2, height), (0, 0, 0))
    glow = Image.new("L", leak.size, 0)
    ImageDraw.Draw(glow).ellipse([width * 0.7, -height * 0.2, width * 1.4, height * 0.5], fill=150)
    ImageDraw.Draw(glow).ellipse([width * 0.2, height * 0.6, width * 0.8, height * 1.1], fill=70)
    leak.paste(
        _mix(accent, (255, 150, 60), 0.5), (0, 0), glow.filter(ImageFilter.GaussianBlur(width // 5))
    )
    return leak


def _grain(width: int, height: int, rng: random.Random) -> tuple[np.ndarray, ...]:
    seeded = np.random.default_rng(rng.randrange(1 << 30))
    small = (height // 2, width // 2)
    tiles = []
    for _ in range(6):
        noise = seeded.normal(0, 7.0, small).astype(np.float32)
        tiles.append(np.repeat(np.repeat(noise, 2, axis=0), 2, axis=1)[:height, :width])
    return tuple(tiles)


# ------------------------------------------------------------------------- drawing


def _draw(stage: Stage, t: float, cut_in: Cut, cut_out: Cut | None) -> Image.Image:
    width, height = stage.width, stage.height
    p = _ease(min(1.0, max(0.0, t / max(stage.seconds, 1e-6))))
    scale, dx, dy = _camera(stage, p)

    if stage.board is not None:
        frame = _zoomed_board(stage, p)
    else:
        # The backdrop moves a third as far as the card: that difference is the depth.
        depth = 0.35 if stage.card is not None else 1.0
        frame = _window(
            stage.backdrop, width, height, 1 + (scale - 1) * depth, dx * depth, dy * depth
        )

    frame = frame.convert("RGBA")
    for particle in stage.particles:
        if particle.kind == "bokeh":
            _place_particle(frame, stage, particle, t)

    if stage.card is not None:
        _place_card(frame, stage, scale, dx, dy, t)

    for particle in stage.particles:
        if particle.kind == "petal":
            _place_particle(frame, stage, particle, t)

    if stage.leak is not None:
        frame = _with_leak(frame, stage, t)
    if stage.vignette is not None:
        frame = Image.alpha_composite(frame, stage.vignette)

    # The transition acts on the picture; the text is drawn over it afterwards, so a
    # whip never smears it. Text leaves by fading out before a whip or fade takes the
    # picture away.
    image = _transitioned(frame.convert("RGB"), t, stage.seconds, cut_in, cut_out)
    leaving = transition_seconds(cut_out) if cut_out in ("whip", "fade") else 0.0
    remaining = stage.seconds - t
    presence = 1.0 if leaving == 0 or remaining >= leaving else max(0.0, remaining / leaving)
    if presence > 0 and stage.sprites:
        frame = image.convert("RGBA")
        for sprite in stage.sprites:
            _place_sprite(frame, sprite, t, width, height, presence)
        image = frame.convert("RGB")
    if stage.grain:
        tile = stage.grain[int(t * FPS) % len(stage.grain)]
        array = np.asarray(image, dtype=np.float32) + tile[:, :, None]
        image = Image.fromarray(np.clip(array, 0, 255).astype(np.uint8), "RGB")
    return image


def _camera(stage: Stage, p: float) -> tuple[float, float, float]:
    """Scale and offset (fractions of the frame) for the shot's move at progress p."""
    move = stage.move
    if move == "push_in":
        return 1.0 + 0.12 * p, 0.0, 0.0
    if move == "pull_out":
        return 1.12 - 0.12 * p, 0.0, 0.0
    if move == "pan_left":
        return 1.08, 0.05 - 0.10 * p, 0.0
    if move == "pan_right":
        return 1.08, -0.05 + 0.10 * p, 0.0
    if move == "rise":
        return 1.06, 0.0, 0.04 - 0.08 * p
    if move == "drift":
        return 1.03 + 0.04 * p, 0.015 - 0.03 * p, 0.0
    return 1.0, 0.0, 0.0


def _window(
    image: Image.Image, width: int, height: int, scale: float, dx: float, dy: float
) -> Image.Image:
    """A frame-sized view into a larger image, zoomed by `scale` and shifted by dx, dy."""
    iw, ih = image.size
    base = max(width / iw, height / ih)
    view_w = width / (base * scale)
    view_h = height / (base * scale)
    cx = iw / 2 - dx * iw * 0.5
    cy = ih / 2 - dy * ih * 0.5
    left = min(max(0.0, cx - view_w / 2), iw - view_w)
    top = min(max(0.0, cy - view_h / 2), ih - view_h)
    return image.resize(
        (width, height), Image.BILINEAR, box=(left, top, left + view_w, top + view_h)
    )


def _zoomed_board(stage: Stage, p: float) -> Image.Image:
    assert stage.board is not None
    width, height = stage.width, stage.height
    if stage.zoom_from is None:
        scale, dx, dy = _camera(stage, p)
        return _window(stage.board, width, height, scale, dx, dy)
    fx, fy, fw, fh = stage.zoom_from
    # Hold tight for the first fifth, then pull out: the viewer must see the face first.
    q = _ease(min(1.0, max(0.0, (p - 0.18) / 0.62)))
    left = fx * (1 - q)
    top = fy * (1 - q)
    w = fw + (1 - fw) * q
    h = fh + (1 - fh) * q
    return stage.board.resize(
        (width, height),
        Image.BILINEAR,
        box=(left * width, top * height, (left + w) * width, (top + h) * height),
    )


def _place_card(
    frame: Image.Image, stage: Stage, scale: float, dx: float, dy: float, t: float
) -> None:
    assert stage.card is not None
    card = stage.card
    # The card settles in over the first third of a second, rising slightly.
    settle = _ease(min(1.0, t / 0.35))
    size = (max(2, round(card.width * scale)), max(2, round(card.height * scale)))
    scaled = card.resize(size, Image.BILINEAR) if scale != 1.0 else card
    cx = stage.card_centre[0] * stage.width + dx * stage.width * 0.6
    cy = (
        stage.card_centre[1] * stage.height
        + dy * stage.height * 0.6
        + (1 - settle) * stage.height * 0.03
    )
    frame.alpha_composite(scaled, (round(cx - size[0] / 2), round(cy - size[1] / 2)))


def _place_particle(frame: Image.Image, stage: Stage, particle: _Particle, t: float) -> None:
    if particle.kind == "petal":
        sprite = stage.petal_images[particle.size * 12 + (particle.spin + int(t * 8)) % 12]
        y = (particle.y + particle.speed * t) % 1.1 - 0.05
    else:
        sprite = stage.bokeh_images[particle.size]
        y = (particle.y - particle.speed * t) % 1.1 - 0.05
    x = particle.x + math.sin(particle.phase + t * 1.3) * particle.sway
    frame.alpha_composite(
        sprite,
        (
            int(x * stage.width - sprite.width / 2),
            int(y * stage.height - sprite.height / 2),
        ),
    )


def _with_leak(frame: Image.Image, stage: Stage, t: float) -> Image.Image:
    assert stage.leak is not None
    progress = min(1.0, t / max(stage.seconds, 1e-6))
    offset = int(stage.width * progress * 0.8)
    band = stage.leak.crop((offset, 0, offset + stage.width, stage.height))
    strength = 0.55 + 0.25 * math.sin(t * 2.1)
    lit = ImageChops.screen(frame.convert("RGB"), band)
    return Image.blend(frame.convert("RGB"), lit, strength).convert("RGBA")


def _place_sprite(
    frame: Image.Image,
    sprite: _Sprite,
    t: float,
    width: int,
    height: int,
    presence: float = 1.0,
) -> None:
    u = t - sprite.appears
    if u < 0:
        return
    entrance = sprite.caption.entrance
    image = sprite.image
    alpha = 1.0
    scale = 1.0
    lift = 0.0
    if entrance == "pop":
        alpha = min(1.0, u / 0.08)
        scale = (
            0.5 + 0.62 * _ease(min(1.0, u / 0.16))
            if u < 0.16
            else 1.12 - 0.12 * _ease(min(1.0, (u - 0.16) / 0.14))
        )
    elif entrance == "slide_up":
        k = _ease(min(1.0, u / 0.35))
        alpha, lift = k, (1 - k) * height * 0.03
    elif entrance == "fade":
        alpha = _ease(min(1.0, u / 0.5))
    elif entrance in ("type", "words") and sprite.prefixes:
        step = 0.045 if entrance == "type" else 0.22
        count = min(len(sprite.prefixes), int(u / step) + 1)
        image = sprite.prefixes[count - 1]
    if scale != 1.0:
        image = image.resize(
            (max(1, round(image.width * scale)), max(1, round(image.height * scale))),
            Image.BILINEAR,
        )
    alpha *= presence
    if alpha < 1.0:
        faded = image.copy()
        faded.putalpha(image.getchannel("A").point(lambda a: int(a * alpha)))
        image = faded
    x = round(sprite.centre[0] * width - image.width / 2)
    y = round(sprite.centre[1] * height - image.height / 2 + lift)
    frame.alpha_composite(image, (x, y))


def _transitioned(
    image: Image.Image, t: float, seconds: float, cut_in: Cut, cut_out: Cut | None
) -> Image.Image:
    if cut_in != "cut" and t < TRANSITION * (2 if cut_in == "fade" else 1):
        image = _cut_effect(
            image, cut_in, t / (TRANSITION * (2 if cut_in == "fade" else 1)), entering=True
        )
    remaining = seconds - t
    if cut_out in ("whip", "fade") and remaining < TRANSITION:
        image = _cut_effect(image, cut_out, 1 - remaining / TRANSITION, entering=False)
    return image


def _cut_effect(image: Image.Image, cut: Cut, k: float, *, entering: bool) -> Image.Image:
    """k runs 0 to 1 through the transition."""
    width, height = image.size
    if cut == "flash" and entering:
        return Image.blend(Image.new("RGB", image.size, (255, 255, 255)), image, _ease(k))
    if cut == "fade":
        amount = _ease(k) if entering else 1 - _ease(k)
        return Image.blend(Image.new("RGB", image.size, (0, 0, 0)), image, amount)
    if cut == "zoom" and entering:
        scale = 1.25 - 0.25 * _ease(k)
        return _window(image, width, height, scale, 0.0, 0.0)
    if cut == "whip":
        travel = (1 - _ease(k)) if entering else _ease(k)
        shift = int(width * 0.35 * travel) * (1 if entering else -1)
        array = np.asarray(image, dtype=np.float32)
        # Shifted with the edge held, not rolled: a roll wraps the far edge round and
        # left ghost copies of the picture on the near one.
        blurred = sum(_shifted(array, shift + i * width // 60) for i in range(-3, 4)) / 7
        return Image.fromarray(blurred.astype(np.uint8), "RGB")
    return image


# ------------------------------------------------------------------------- helpers


def _shifted(array: np.ndarray, shift: int) -> np.ndarray:
    """The image moved sideways by `shift` pixels, its edge column filling the gap."""
    width = array.shape[1]
    shift = max(-width + 1, min(width - 1, shift))
    if shift > 0:
        return np.concatenate([np.repeat(array[:, :1], shift, axis=1), array[:, :-shift]], axis=1)
    if shift < 0:
        return np.concatenate([array[:, -shift:], np.repeat(array[:, -1:], -shift, axis=1)], axis=1)
    return array


def _ease(x: float) -> float:
    return x * x * (3 - 2 * x)


def _cover(image: Image.Image, width: int, height: int) -> Image.Image:
    scale = max(width / image.width, height / image.height)
    resized = image.resize(
        (max(1, round(image.width * scale)), max(1, round(image.height * scale))), Image.LANCZOS
    )
    left = (resized.width - width) // 2
    top = (resized.height - height) // 2
    return resized.crop((left, top, left + width, top + height))


def _sharpened(image: Image.Image) -> Image.Image:
    return image.filter(ImageFilter.UnsharpMask(radius=2, percent=90, threshold=2))


def _tinted(image: Image.Image, colour: tuple[int, int, int], amount: float) -> Image.Image:
    return Image.blend(image.convert("RGB"), Image.new("RGB", image.size, colour), amount)


def _gradient(
    width: int, height: int, top: tuple[int, int, int], bottom: tuple[int, int, int]
) -> Image.Image:
    ramp = np.linspace(0, 1, height)[:, None, None]
    column = np.array(top)[None, None, :] * (1 - ramp) + np.array(bottom)[None, None, :] * ramp
    return Image.fromarray(np.tile(column, (1, width, 1)).astype(np.uint8), "RGB")


def _mix(a: tuple[int, int, int], b: tuple[int, int, int], amount: float) -> tuple[int, int, int]:
    return tuple(int(x * (1 - amount) + y * amount) for x, y in zip(a, b, strict=True))  # type: ignore[return-value]


def _rgb(colour: str) -> tuple[int, int, int]:
    value = colour.lstrip("#")
    return int(value[0:2], 16), int(value[2:4], 16), int(value[4:6], 16)
