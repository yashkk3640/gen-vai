"""RendererPort over ffmpeg. The only module that builds a filtergraph.

The binary is resolved PATH-first, falling back to the copy bundled by
`imageio-ffmpeg`, so a clean clone renders with no system install.

Scenes are rendered to individually cached segments, encoded identically so they can be
joined. Photos get Ken Burns motion via `zoompan`; clips are trimmed with an input seek
and fitted to the canvas. Joining is a stream copy when every transition is a cut, and
an `xfade` chain when it is not.
"""

import shutil
import subprocess
import tempfile
from collections.abc import Callable
from pathlib import Path

from genvai.config import RenderSettings
from genvai.errors import RenderError
from genvai.fonts import for_ffmpeg
from genvai.fonts import resolve as resolve_font
from genvai.ports import MediaInfo
from genvai.timeline import (
    AssetVisual,
    AudioVariant,
    Canvas,
    CardVisual,
    ClipVisual,
    ColorVisual,
    GeneratedVisual,
    ImageOverlay,
    KenBurns,
    Rect,
    Scene,
    TextOverlay,
    TextStyle,
    Timeline,
)

SAMPLE_RATE = 48000
"""Every segment is encoded at this rate so concatenation never resamples."""


def resolve_ffmpeg() -> Path:
    """Locate ffmpeg: a system binary on PATH if present, else the bundled one.

    PATH wins because a full system build supports codecs the bundled one does not,
    and anyone who needs those will have installed it deliberately.
    """
    system = shutil.which("ffmpeg")
    if system:
        return Path(system)
    try:
        import imageio_ffmpeg

        return Path(imageio_ffmpeg.get_ffmpeg_exe())
    except Exception as exc:  # noqa: BLE001 - reported as a typed error below
        raise RenderError(
            "ffmpeg not found: no system binary on PATH and the bundled copy could not "
            f"be located ({exc}). Run 'uv sync' to reinstall imageio-ffmpeg."
        ) from exc


def resolve_ffprobe() -> Path | None:
    """ffprobe if the system has it. The bundled package ships ffmpeg only."""
    system = shutil.which("ffprobe")
    return Path(system) if system else None


class FFmpegRenderer:
    """Implements `RendererPort`.

    `resolve_asset` maps an asset id to a file on disk. Injecting it keeps the renderer
    unaware of how projects are stored, so it can be driven from a store, a temporary
    directory, or a test fixture without changing.
    """

    def __init__(
        self,
        settings: RenderSettings,
        resolve_asset: Callable[[str], Path],
        binary: Path | None = None,
    ) -> None:
        self._settings = settings
        self._resolve_asset = resolve_asset
        self._binary = binary or resolve_ffmpeg()

    # ------------------------------------------------------------------- scenes

    def render_scene(self, timeline: Timeline, scene_id: str, out: Path) -> Path:
        """Render one scene to a segment: source, trim, fit, motion, overlays.

        Segments always carry an audio track - silence when the clip is muted - because
        a mix of audio-bearing and audio-free segments cannot be concatenated.
        """
        scene = timeline.scene(scene_id)
        if scene is None:
            raise RenderError(f"no scene '{scene_id}' in timeline v{timeline.version}")

        canvas = timeline.canvas
        out.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix="genvai-text-") as scratch:
            workdir = Path(scratch)
            inputs, video_chain, audio_label, audio_chain = self._source_stage(
                scene, canvas, workdir
            )
            video_chain += self._overlay_stage(timeline, scene, workdir)
            graph = f"{video_chain},format=yuv420p[v]"
            if audio_chain:
                graph = f"{graph};{audio_chain}"
            self._encode_segment(inputs, graph, audio_label, scene, canvas, out)
        return out

    def _encode_segment(
        self,
        inputs: list[str],
        graph: str,
        audio_label: str,
        scene: Scene,
        canvas: Canvas,
        out: Path,
    ) -> None:
        """Encode one segment. Every segment uses identical settings so they concatenate."""
        command = [
            str(self._binary),
            "-y",
            "-hide_banner",
            "-loglevel",
            "error",
            *inputs,
            "-filter_complex",
            graph,
            "-map",
            "[v]",
            "-map",
            audio_label,
            "-c:v",
            self._settings.video_codec,
            "-preset",
            self._settings.preset,
            "-crf",
            str(self._settings.crf),
            "-pix_fmt",
            "yuv420p",
            "-r",
            str(canvas.fps),
            "-c:a",
            self._settings.audio_codec,
            "-ar",
            str(SAMPLE_RATE),
            "-ac",
            "2",
            "-t",
            f"{scene.duration:.4f}",
            "-movflags",
            "+faststart",
            str(out),
        ]
        self._run(command, f"rendering scene '{scene.id}'")

    def _source_stage(
        self, scene: Scene, canvas: Canvas, workdir: Path
    ) -> tuple[list[str], str, str, str]:
        """Build the input arguments and the video chain up to the fitted picture.

        Returns the ffmpeg inputs, an unlabelled video chain, the label to map for
        audio, and an audio filter chain (empty when the audio needs no processing).
        """
        visual = scene.visual
        silence = [
            "-f",
            "lavfi",
            "-t",
            f"{scene.duration:.4f}",
            "-i",
            f"anullsrc=channel_layout=stereo:sample_rate={SAMPLE_RATE}",
        ]

        if isinstance(visual, ClipVisual):
            inputs = [
                "-ss",
                f"{visual.source_start:.4f}",
                "-t",
                f"{visual.source_duration:.4f}",
                "-i",
                str(self._resolve_asset(visual.asset_id)),
            ]
            chain = (
                f"[0:v]{_crop(visual.crop, visual.crop_end, scene.duration)}"
                f"{_fit(visual.fit, canvas)}"
            )
            if visual.speed != 1.0:
                chain += f",setpts=PTS/{visual.speed}"
            chain += f",fps={canvas.fps},setsar=1"
            if visual.mute:
                return [*inputs, *silence], chain, "1:a", ""
            # Keeping a clip's own sound means retiming it too: setpts moves the video
            # but not the audio, so a speed change without atempo desyncs them.
            audio_chain = f"[0:a]{_atempo(visual.speed)},aresample={SAMPLE_RATE}[a]"
            return [*inputs, *silence], chain, "[a]", audio_chain

        if isinstance(visual, AssetVisual | GeneratedVisual):
            asset_id = visual.asset_id
            if asset_id is None:
                raise RenderError(
                    f"scene '{scene.id}' has an unresolved generated visual; "
                    "run the resolve phase before rendering"
                )
            inputs = [
                "-loop",
                "1",
                "-framerate",
                str(canvas.fps),
                "-t",
                f"{scene.duration:.4f}",
                "-i",
                str(self._resolve_asset(asset_id)),
            ]
            crop = _crop(getattr(visual, "crop", None))
            fit = _fit(getattr(visual, "fit", "cover"), canvas, oversample=2)
            motion = self._motion(scene, canvas)
            return [*inputs, *silence], f"[0:v]{crop}{fit}{motion},setsar=1", "1:a", ""

        if isinstance(visual, CardVisual | ColorVisual):
            color = visual.color if isinstance(visual, ColorVisual) else canvas.background
            inputs = [
                "-f",
                "lavfi",
                "-t",
                f"{scene.duration:.4f}",
                "-i",
                f"color=c={color}:s={canvas.width}x{canvas.height}:r={canvas.fps}",
            ]
            chain = "[0:v]null"
            if isinstance(visual, CardVisual):
                chain += "," + _drawtext(
                    visual.text,
                    "center",
                    TextStyle(),
                    canvas,
                    scene.duration,
                    0.0,
                    None,
                    workdir / "card.txt",
                )
            return [*inputs, *silence], chain, "1:a", ""

        raise RenderError(f"scene '{scene.id}' has an unsupported visual")

    def _motion(self, scene: Scene, canvas: Canvas) -> str:
        """Camera move over a still. Only Ken Burns produces movement today.

        `zoompan` is driven one input frame at a time (`d=1`) against a looped still,
        which animates reliably; feeding it a single frame with `d=N` is the variant
        that stutters.
        """
        motion = scene.motion
        if not isinstance(motion, KenBurns):
            return f",scale={canvas.width}:{canvas.height}"

        frames = max(2, round(scene.duration * canvas.fps))
        progress = _ease(f"(on/{frames - 1})", motion.easing)
        x0, y0, w0, _ = motion.start_rect
        x1, y1, w1, _ = motion.end_rect
        width = f"({w0}+({w1}-{w0})*{progress})"
        zoom = f"1/max(0.0001,{width})"
        x = f"({x0}+({x1}-{x0})*{progress})*iw"
        y = f"({y0}+({y1}-{y0})*{progress})*ih"
        return (
            f",zoompan=z='{zoom}':x='{x}':y='{y}':d=1"
            f":s={canvas.width}x{canvas.height}:fps={canvas.fps}"
        )

    def _overlay_stage(self, timeline: Timeline, scene: Scene, workdir: Path) -> str:
        """Burn text overlays, positioned inside the canvas safe area."""
        chain = ""
        for index, overlay in enumerate(scene.overlays):
            if isinstance(overlay, TextOverlay):
                style = timeline.styles.get(overlay.style_ref, TextStyle())
                chain += "," + _drawtext(
                    overlay.content,
                    overlay.position,
                    style,
                    timeline.canvas,
                    scene.duration,
                    overlay.start_offset,
                    overlay.duration,
                    workdir / f"overlay{index}.txt",
                )
            elif isinstance(overlay, ImageOverlay):
                # Image overlays need a second input, so they are a later milestone.
                continue
        return chain

    # -------------------------------------------------------------------- join

    def concat(self, segments: tuple[Path, ...], out: Path) -> Path:
        """Join segments. Stream-copies when possible, cross-fades when asked.

        `transitions` is carried on the timeline, so callers use `concat_with` when any
        scene has a non-cut transition.
        """
        return self.concat_with(segments, (), out)

    def concat_with(
        self, segments: tuple[Path, ...], transitions: tuple[tuple[str, float], ...], out: Path
    ) -> Path:
        """Join segments, applying each scene's incoming transition.

        `transitions[i]` describes how `segments[i + 1]` enters. When every transition
        is a cut the concat demuxer stream-copies, which is near-instant; otherwise an
        `xfade` chain re-encodes, which is why cuts stay the default.
        """
        if not segments:
            raise RenderError("nothing to concatenate: the timeline has no scenes")
        out.parent.mkdir(parents=True, exist_ok=True)
        if len(segments) == 1:
            shutil.copy2(segments[0], out)
            return out

        blended = [(kind, seconds) for kind, seconds in transitions if seconds > 0.0]
        if not blended:
            return self._concat_copy(segments, out)
        return self._concat_xfade(segments, transitions, out)

    def _concat_copy(self, segments: tuple[Path, ...], out: Path) -> Path:
        listing = out.parent / f"{out.stem}.concat.txt"
        listing.write_text(
            "\n".join(f"file '{p.resolve().as_posix()}'" for p in segments) + "\n",
            encoding="utf-8",
        )
        command = [
            str(self._binary),
            "-y",
            "-hide_banner",
            "-loglevel",
            "error",
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
        ]
        self._run(command, "joining segments")
        listing.unlink(missing_ok=True)
        return out

    def _concat_xfade(
        self, segments: tuple[Path, ...], transitions: tuple[tuple[str, float], ...], out: Path
    ) -> Path:
        durations = [self.probe(p).duration for p in segments]
        inputs: list[str] = []
        for segment in segments:
            inputs += ["-i", str(segment)]

        video = "[0:v]"
        audio = "[0:a]"
        steps: list[str] = []
        accumulated = durations[0]
        for index in range(1, len(segments)):
            kind, seconds = transitions[index - 1] if index - 1 < len(transitions) else ("cut", 0.0)
            seconds = max(0.0, min(seconds, durations[index] - 0.05, accumulated - 0.05))
            offset = max(0.0, accumulated - seconds)
            effect = _XFADE.get(kind, "fade")
            vlabel, alabel = f"[v{index}]", f"[a{index}]"
            steps.append(
                f"{video}[{index}:v]xfade=transition={effect}"
                f":duration={seconds:.4f}:offset={offset:.4f}{vlabel}"
            )
            steps.append(f"{audio}[{index}:a]acrossfade=d={max(seconds, 0.01):.4f}{alabel}")
            video, audio = vlabel, alabel
            accumulated = accumulated + durations[index] - seconds

        command = [
            str(self._binary),
            "-y",
            "-hide_banner",
            "-loglevel",
            "error",
            *inputs,
            "-filter_complex",
            ";".join(steps),
            "-map",
            video,
            "-map",
            audio,
            "-c:v",
            self._settings.video_codec,
            "-preset",
            self._settings.preset,
            "-crf",
            str(self._settings.crf),
            "-pix_fmt",
            "yuv420p",
            "-c:a",
            self._settings.audio_codec,
            "-ar",
            str(SAMPLE_RATE),
            "-ac",
            "2",
            "-movflags",
            "+faststart",
            str(out),
        ]
        self._run(command, "cross-fading segments")
        return out

    # ------------------------------------------------------------------- audio

    def mix_audio(
        self, video: Path, timeline: Timeline, out: Path, variant: AudioVariant = "full"
    ) -> Path:
        """Lay audio over the cut, ducking music under speech.

        `narration_only` skips the music input; `silent` drops audio entirely. All three
        reuse the same encoded video, so the extra cuts cost one mux each.
        """
        out.parent.mkdir(parents=True, exist_ok=True)
        base = [str(self._binary), "-y", "-hide_banner", "-loglevel", "error", "-i", str(video)]

        if variant == "silent":
            command = [*base, "-map", "0:v", "-an", "-c:v", "copy", str(out)]
            self._run(command, "writing the silent cut")
            return out

        music_id = timeline.music.asset_id if variant == "full" else None
        if music_id is None or timeline.music.state != "resolved":
            command = [*base, "-c", "copy", str(out)]
            self._run(command, f"writing the {variant} cut")
            return out

        gain = 10 ** (timeline.music.gain_db / 20)
        # normalize=0: amix otherwise divides every input by the input count, so the bed
        # came out 6 dB under its own gain_db and a -3 dB promo track peaked near -20.
        graph = (
            f"[1:a]volume={gain:.4f},afade=t=out:st="
            f"{max(0.0, timeline.duration - 1.5):.4f}:d=1.5[bed];"
            "[0:a][bed]amix=inputs=2:duration=first:dropout_transition=0:normalize=0[a]"
        )
        command = [
            *base,
            "-stream_loop",
            "-1",
            "-i",
            str(self._resolve_asset(music_id)),
            "-filter_complex",
            graph,
            "-map",
            "0:v",
            "-map",
            "[a]",
            "-c:v",
            "copy",
            "-c:a",
            self._settings.audio_codec,
            "-ar",
            str(SAMPLE_RATE),
            "-ac",
            "2",
            "-shortest",
            "-movflags",
            "+faststart",
            str(out),
        ]
        self._run(command, "mixing the full cut")
        return out

    # ------------------------------------------------------------------- probe

    def probe(self, media: Path) -> MediaInfo:
        """Read duration and dimensions.

        Uses ffprobe when the system provides it, and otherwise parses ffmpeg's own
        report - the bundled package ships ffmpeg alone, so the fallback is the normal
        path on a clean clone, not an edge case.
        """
        probe = resolve_ffprobe()
        if probe is not None:
            return _probe_with_ffprobe(probe, media)
        return _probe_with_ffmpeg(self._binary, media)

    # --------------------------------------------------------------------- run

    def _run(self, command: list[str], what: str) -> None:
        result = subprocess.run(command, capture_output=True, text=True, check=False)
        if result.returncode != 0:
            tail = (result.stderr or "").strip().splitlines()[-12:]
            raise RenderError(
                f"ffmpeg failed while {what} (exit {result.returncode})",
                command=command,
                stderr="\n".join(tail),
            )


# ------------------------------------------------------------------- filtergraph

_XFADE: dict[str, str] = {
    "fade": "fade",
    "dissolve": "dissolve",
    "slide": "slideleft",
    "push": "slideleft",
    "wipe": "wipeleft",
    "cut": "fade",
}

_POSITION: dict[str, tuple[str, str]] = {
    "top_left": ("{left}", "{top}"),
    "top_center": ("(w-text_w)/2", "{top}"),
    "top_right": ("w-text_w-{right}", "{top}"),
    "middle_left": ("{left}", "(h-text_h)/2"),
    "center": ("(w-text_w)/2", "(h-text_h)/2"),
    "middle_right": ("w-text_w-{right}", "(h-text_h)/2"),
    "bottom_left": ("{left}", "h-text_h-{bottom}"),
    "bottom_center": ("(w-text_w)/2", "h-text_h-{bottom}"),
    "bottom_right": ("w-text_w-{right}", "h-text_h-{bottom}"),
}


def _fit(fit: str, canvas: Canvas, *, oversample: int = 1) -> str:
    """Scale source material into the canvas.

    `oversample` renders larger than the canvas so a later zoompan has real pixels to
    crop into rather than upscaling its own output.
    """
    width, height = canvas.width * oversample, canvas.height * oversample
    if fit == "contain":
        return (
            f",scale={width}:{height}:force_original_aspect_ratio=decrease"
            f",pad={width}:{height}:(ow-iw)/2:(oh-ih)/2:color={canvas.background}"
        )
    if fit == "blur_pad":
        # Approximated as cover until the split/overlay graph lands; see TODO.md.
        return f",scale={width}:{height}:force_original_aspect_ratio=increase,crop={width}:{height}"
    return f",scale={width}:{height}:force_original_aspect_ratio=increase,crop={width}:{height}"


def _atempo(speed: float) -> str:
    """Retime audio by `speed`, chaining filters to cover the whole range.

    A single `atempo` only accepts 0.5x to 2x, so anything beyond that is expressed as
    several stages multiplied together - 4x becomes two doublings.
    """
    stages: list[float] = []
    remaining = speed
    while remaining > 2.0:
        stages.append(2.0)
        remaining /= 2.0
    while remaining < 0.5:
        stages.append(0.5)
        remaining /= 0.5
    stages.append(remaining)
    return ",".join(f"atempo={stage:.6f}" for stage in stages)


def _crop(rect: Rect | None, end: Rect | None = None, duration: float = 0.0) -> str:
    """Keep a sub-rectangle of the source, in fractions of its own frame.

    With `end`, the window travels there over the shot - a slow reframe that follows the
    subject. Only the position may move: `crop` refuses a changing output size, so a
    mismatched width or height falls back to holding the start rectangle rather than
    producing a graph ffmpeg will reject.
    """
    if rect is None:
        return "null"
    x, y, width, height = rect
    if end is None or duration <= 0 or (end[2], end[3]) != (width, height):
        return f"crop=iw*{width}:ih*{height}:iw*{x}:ih*{y}"

    # No `eval=frame` here: crop has no such option, and needs none. Its x and y
    # expressions are re-evaluated every frame already; only w and h are fixed once,
    # which is exactly the split this needs.
    progress = f"min(1,t/{duration:.4f})"
    moving_x = f"iw*({x}+({end[0]}-{x})*{progress})"
    moving_y = f"ih*({y}+({end[1]}-{y})*{progress})"
    return f"crop=w=iw*{width}:h=ih*{height}:x='{moving_x}':y='{moving_y}'"


def _ease(progress: str, easing: str) -> str:
    """Shape linear progress. Expressed inline so ffmpeg evaluates it per frame."""
    if easing == "ease_in":
        return f"({progress}*{progress})"
    if easing == "ease_out":
        return f"(1-(1-{progress})*(1-{progress}))"
    if easing == "ease_in_out":
        return f"({progress}*{progress}*(3-2*{progress}))"
    return f"({progress})"


def _drawtext(
    text: str,
    position: str,
    style: TextStyle,
    canvas: Canvas,
    scene_duration: float,
    start_offset: float,
    duration: float | None,
    textfile: Path,
) -> str:
    """One drawtext filter, wrapped to the safe area and placed inside it.

    The text goes through `textfile=` rather than `text=`. Inline text has to survive
    two levels of filtergraph parsing, where colons, quotes, percent signs and newlines
    all mean something - a caption containing an apostrophe or a line break is enough to
    corrupt the graph. A file has none of those problems and handles Unicode besides.
    """
    safe = canvas.safe_area
    size = max(8, round(canvas.height * style.size_pct / 100))
    body = text.upper() if style.uppercase else text
    # newline="\n" is load-bearing on Windows: the default translates to CRLF and
    # drawtext renders the stray carriage return as extra leading, tripling the gap
    # between wrapped lines. The same code looks correct on Linux, so it is pinned here.
    textfile.write_text(
        _wrap(body, _wrap_width(style, canvas, size)), encoding="utf-8", newline="\n"
    )

    x_template, y_template = _POSITION.get(position, _POSITION["bottom_center"])
    geometry = {
        "left": round(canvas.width * safe.left),
        "right": round(canvas.width * safe.right),
        "top": round(canvas.height * safe.top),
        "bottom": round(canvas.height * safe.bottom),
    }
    parts = [
        f"textfile='{for_ffmpeg(textfile)}'",
        f"fontsize={size}",
        f"fontcolor={style.color}",
        f"x={x_template.format(**geometry)}",
        f"y={y_template.format(**geometry)}",
        f"line_spacing={round(size * (style.line_spacing - 1))}",
    ]
    if position.endswith("center"):
        # Centre each line in the block, not just the block: a wrapped caption otherwise
        # hangs its short second line off the left edge of its long first one.
        parts.append("text_align=C")
    if style.stroke_px > 0:
        parts += [f"borderw={style.stroke_px}", f"bordercolor={style.stroke_color}"]

    font = resolve_font(style.font)
    if font is not None:
        parts.append(f"fontfile='{for_ffmpeg(font)}'")

    end_time = start_offset + duration if duration is not None else scene_duration
    if start_offset > 0.0 or end_time < scene_duration:
        parts.append(f"enable='between(t,{start_offset:.3f},{end_time:.3f})'")
    return "drawtext=" + ":".join(parts)


_GLYPH_ASPECT = 0.55
"""Average advance width of a bold sans glyph, as a fraction of its point size.

Rough, but it only has to keep text inside the frame, and erring narrow costs a line
break where erring wide costs a caption running off the screen.
"""


def _wrap_width(style: TextStyle, canvas: Canvas, size: int) -> int:
    """How many characters fit on one line inside the safe area.

    The style's `max_chars_per_line` is an editorial preference - keep lines short - but
    it cannot know the canvas. Font size is a fraction of *height* while the line has to
    fit the *width*, so on a 9:16 frame the geometric limit is usually the binding one.
    Whichever is tighter wins.
    """
    usable = canvas.width * max(0.1, 1.0 - canvas.safe_area.left - canvas.safe_area.right)
    geometric = max(8, int(usable / (size * _GLYPH_ASPECT)))
    return min(style.max_chars_per_line, geometric)


def _wrap(text: str, width: int) -> str:
    """Greedy wrap. drawtext has no wrapping of its own, so lines are pre-broken.

    A word longer than the line is broken rather than left to overflow. Without this a
    single long token - a phone number, a URL, a hashtag - runs off the side of the frame
    with nothing at all to signal it, which is worse than an ugly break.

    Line breaks already in the text are kept - a price list is written one per line.
    """
    if "\n" in text:
        return "\n".join(_wrap(line, width) for line in text.splitlines() if line.strip())

    lines: list[str] = []
    current = ""
    for word in text.split():
        remaining = word
        while len(remaining) > width:
            if current:
                lines.append(current)
                current = ""
            lines.append(remaining[:width])
            remaining = remaining[width:]
        candidate = f"{current} {remaining}".strip()
        if len(candidate) > width and current:
            lines.append(current)
            current = remaining
        else:
            current = candidate
    if current:
        lines.append(current)
    return "\n".join(lines)


def _probe_with_ffprobe(probe: Path, media: Path) -> MediaInfo:
    result = subprocess.run(
        [
            str(probe),
            "-v",
            "error",
            "-show_entries",
            "format=duration:stream=width,height,r_frame_rate,codec_type",
            "-of",
            "default=noprint_wrappers=1",
            str(media),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    fields: dict[str, str] = {}
    for line in result.stdout.splitlines():
        if "=" in line:
            key, _, value = line.partition("=")
            fields.setdefault(key.strip(), value.strip())
    return MediaInfo(
        duration=_as_float(fields.get("duration"), 0.0),
        width=int(fields["width"]) if fields.get("width", "").isdigit() else None,
        height=int(fields["height"]) if fields.get("height", "").isdigit() else None,
        fps=_as_fps(fields.get("r_frame_rate")),
        has_audio="audio" in result.stdout,
    )


def _probe_with_ffmpeg(binary: Path, media: Path) -> MediaInfo:
    """Parse ffmpeg's stream report. Exits non-zero by design - there is no output."""
    result = subprocess.run(
        [str(binary), "-hide_banner", "-i", str(media)],
        capture_output=True,
        text=True,
        check=False,
    )
    report = result.stderr
    duration = 0.0
    for line in report.splitlines():
        stripped = line.strip()
        if stripped.startswith("Duration:"):
            clock = stripped.split("Duration:", 1)[1].split(",", 1)[0].strip()
            duration = _as_clock(clock)
            break

    width = height = None
    fps = None
    for line in report.splitlines():
        if "Video:" not in line:
            continue
        for token in line.split(","):
            token = token.strip()
            if "x" in token and token.split()[0].replace("x", "").isdigit():
                dimensions = token.split()[0].split("x")
                if len(dimensions) == 2:
                    width, height = int(dimensions[0]), int(dimensions[1])
            if token.endswith("fps"):
                fps = _as_float(token[:-3].strip(), None)
        break

    return MediaInfo(
        duration=duration, width=width, height=height, fps=fps, has_audio="Audio:" in report
    )


def _as_float(value: str | None, default: float | None) -> float | None:
    try:
        return float(value) if value else default
    except ValueError:
        return default


def _as_fps(value: str | None) -> float | None:
    if not value or "/" not in value:
        return _as_float(value, None)
    numerator, _, denominator = value.partition("/")
    try:
        return float(numerator) / float(denominator) if float(denominator) else None
    except ValueError:
        return None


def _as_clock(value: str) -> float:
    try:
        hours, minutes, seconds = value.split(":")
        return int(hours) * 3600 + int(minutes) * 60 + float(seconds)
    except ValueError:
        return 0.0
