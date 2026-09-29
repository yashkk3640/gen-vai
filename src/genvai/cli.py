"""Command line surface. Thin: parsing, wiring, and presentation only.

Every command delegates to `genvai.pipeline`. No editorial logic lives here, so an
HTTP API can be added over the same core without moving anything.
"""

import importlib.util
import secrets
import shutil
import sys
from pathlib import Path

import typer
from PIL import Image
from rich.console import Console
from rich.progress import BarColumn, Progress, SpinnerColumn, TextColumn
from rich.table import Table

from genvai import __version__
from genvai.adapters.analyzer import FrameAnalyzer
from genvai.adapters.beat import FFmpegBeatDetector
from genvai.adapters.compositor import Compositor
from genvai.adapters.ffmpeg import FFmpegRenderer, resolve_ffmpeg
from genvai.adapters.fs_store import FilesystemStore
from genvai.adapters.images import image_provider
from genvai.adapters.music_local import LocalMusicProvider
from genvai.adapters.ollama import OllamaLLM
from genvai.adapters.rapid_ocr import RapidOcrVision
from genvai.adapters.yunet_faces import YunetFaces
from genvai.config import Settings, load_settings
from genvai.errors import GenvaiError, RenderError
from genvai.ocr import Box, read_brief, text_rects
from genvai.pipeline.edit import edit as run_edit
from genvai.pipeline.ingest import ingest, summarise
from genvai.pipeline.music import SOLE_TRACK_DB
from genvai.pipeline.music import approve as approve_music
from genvai.pipeline.music import attach as attach_music
from genvai.pipeline.music import decline as decline_music
from genvai.pipeline.music import suggest as suggest_music
from genvai.pipeline.plan import plan as plan_idea
from genvai.pipeline.promo import STYLES, structure_for
from genvai.pipeline.promo import build as build_promo
from genvai.pipeline.promo import choose as choose_offers
from genvai.pipeline.render import plan_render
from genvai.pipeline.render import render as render_timeline
from genvai.pipeline.resolve import resolve_visuals
from genvai.pipeline.select import make_reel
from genvai.pipeline.story import draft as draft_story
from genvai.pipeline.story import read_material
from genvai.pipeline.story import shoot as shoot_story
from genvai.pipeline.story import storybook as make_storybook
from genvai.promo import Brief
from genvai.stories import ARC_NAMES, arc_for
from genvai.timeline import Canvas, Rect, Timeline

app = typer.Typer(
    name="genvai",
    help="Generative, conversational video editor driven by a local LLM.",
    no_args_is_help=True,
    add_completion=False,
)
console = Console()


@app.command()
def doctor() -> None:
    """Report what is available on this machine.

    Run this first on any new system - it is the fastest way to see which providers
    will work and which will fall back.
    """
    settings = load_settings()
    table = Table(title=f"gen-vai {__version__}", show_header=True, header_style="bold")
    table.add_column("Component")
    table.add_column("Status")
    table.add_column("Detail", overflow="fold")

    table.add_row("Python", "ok", sys.version.split()[0])

    ffmpeg, ffmpeg_detail = _probe_ffmpeg()
    table.add_row("ffmpeg", ffmpeg, ffmpeg_detail)

    ollama, ollama_detail = _probe_ollama(settings.llm.base_url, settings.llm.model)
    table.add_row("Ollama", ollama, ollama_detail)

    for label, module, extra in (
        ("torch / CUDA", "torch", "image"),
        ("diffusers", "diffusers", "image"),
        ("piper (TTS)", "piper", "tts"),
        ("faster-whisper (ASR)", "faster_whisper", "asr"),
    ):
        installed = importlib.util.find_spec(module) is not None
        status = "ok" if installed else "missing"
        detail = _cuda_detail() if (installed and module == "torch") else f"uv sync --extra {extra}"
        table.add_row(label, status, detail)

    faces = _faces(settings)
    table.add_row(
        "faces (YuNet)",
        "ok" if faces.is_available() else "missing",
        str(settings.faces.model_path) if faces.is_available() else "uv sync --extra ocr",
    )

    table.add_row("projects dir", "ok", str(settings.projects_dir.resolve()))
    console.print(table)
    console.print(
        "\n[dim]Missing optional components are not errors - the pipeline degrades "
        "to procedural visuals and silent audio. See docs/setup.md.[/dim]"
    )


@app.command()
def add(
    project: str = typer.Argument(..., help="Project id, or a new name to create one."),
    files: list[Path] = typer.Argument(..., help="Photos and clips. Folders are walked."),
) -> None:
    """Import a camera roll and analyse it.

    The slow step, cached by content hash - re-adding the same files is free.
    """
    settings = load_settings()
    store = FilesystemStore(settings.projects_dir)
    store.create(f"reel from {files[0].name}", project)
    ffmpeg = resolve_ffmpeg()

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("{task.completed}/{task.total}"),
        console=console,
    ) as bar:
        task = bar.add_task("analysing", total=None)

        def tick(name: str, index: int, total: int) -> None:
            bar.update(task, total=total, completed=index, description=f"analysing {name}")

        library = ingest(
            tuple(files),
            project,
            FrameAnalyzer(ffmpeg, _faces(settings)),
            store,
            ffmpeg,
            on_progress=tick,
        )

    console.print(f"[bold]{project}[/bold]  {summarise(library.items)}")
    console.print("[dim]See what was found with 'genvai media " + project + "'.[/dim]")


@app.command()
def reel(
    project: str = typer.Argument(..., help="Project id."),
    duration: float = typer.Option(30.0, "--duration", "-d", help="Target length in seconds."),
    intent: str = typer.Option("", "--intent", help="Steer it: 'focus on the food'."),
    aspect: str = typer.Option("9:16", "--aspect", help="9:16, 16:9 or 1:1."),
    seed: int = typer.Option(0, "--seed", help="0 picks a random seed and records it."),
) -> None:
    """Build a reel from the project's media."""
    settings = load_settings()
    store = FilesystemStore(settings.projects_dir)
    llm: OllamaLLM | None = OllamaLLM(settings.llm)
    if llm is not None and not llm.is_available():
        console.print(
            f"[yellow]{settings.llm.model} not available[/yellow] - ordering shots by "
            "capture time instead. [dim]genvai doctor[/dim] says what is missing."
        )
        llm = None

    try:
        timeline = make_reel(
            project,
            store,
            intent=intent or f"reel from {project}",
            target_duration=duration,
            canvas=_canvas_for(aspect),
            seed=seed or _random_seed(),
            llm=llm,
            ffmpeg=resolve_ffmpeg(),
            faces=_faces(settings),
            on_note=lambda note: console.print(f"[yellow]{note}[/yellow]"),
        )
    except GenvaiError as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(code=1) from exc

    console.print(
        f"[bold]{project}[/bold] v{timeline.version}  "
        f"{len(timeline.scenes)} shots  {timeline.duration:.1f}s  "
        f"(beat {timeline.mean_scene_duration:.1f}s)"
    )
    renderer = FFmpegRenderer(settings.render, lambda asset_id: store.asset_path(project, asset_id))
    try:
        outputs = render_timeline(timeline, project, renderer, store)
    except GenvaiError as exc:
        console.print(f"[red]{exc}[/red]")
        if isinstance(exc, RenderError) and exc.stderr:
            console.print(f"[dim]{exc.stderr}[/dim]")
        raise typer.Exit(code=1) from exc

    for variant, path in outputs.items():
        console.print(f"  [green]{variant}[/green]  {path}")


@app.command()
def media(
    project: str = typer.Argument(..., help="Project id."),
    unused: bool = typer.Option(False, "--unused", help="Only what did not make the cut."),
) -> None:
    """List analysed media with quality scores and the spans that were found."""
    settings = load_settings()
    store = FilesystemStore(settings.projects_dir)
    library = store.load_media(project)
    if not library.items:
        console.print(f"[dim]Nothing imported yet. Try 'genvai add {project} <files...>'.[/dim]")
        return

    keepers = {i.asset_id for i in library.deduplicated()}
    shown = [i for i in library.items if not unused or i.asset_id not in keepers]

    table = Table(show_header=True, header_style="bold")
    table.add_column("File", overflow="fold")
    table.add_column("Kind")
    table.add_column("Sharp", justify="right")
    table.add_column("Expo", justify="right")
    table.add_column("Shake", justify="right")
    table.add_column("Best moment", overflow="fold")
    for item in shown:
        quality = item.quality
        span = item.best_span
        moment = f"{span.start:.1f}-{span.end:.1f}s  {span.reason}" if span else "-"
        if item.asset_id not in keepers:
            moment = "[dim]duplicate[/dim]"
        table.add_row(
            item.source_name,
            item.kind,
            f"{quality.sharpness:.2f}" if quality else "-",
            f"{quality.exposure:.2f}" if quality else "-",
            f"{quality.shake:.2f}" if quality else "-",
            moment,
        )
    console.print(table)
    console.print(f"[dim]{summarise(library.items)}[/dim]")


@app.command()
def create(
    intent: str = typer.Argument(..., help="What the video should be."),
    name: str = typer.Option("", "--name", help="Project id. Defaults to a slug of the idea."),
    duration: float = typer.Option(20.0, "--duration", "-d", help="Target length in seconds."),
    aspect: str = typer.Option("9:16", "--aspect", help="9:16, 16:9 or 1:1."),
    seed: int = typer.Option(0, "--seed", help="0 picks a random seed and records it."),
) -> None:
    """Idea mode: make a video from a description, with no footage at all.

    Every frame is generated. The weakest thing this does - real footage beats it every
    time - but it works when there is no camera roll to work from.
    """
    settings = load_settings()
    store = FilesystemStore(settings.projects_dir)
    project = store.create(intent, name or None)

    llm: OllamaLLM | None = OllamaLLM(settings.llm)
    if llm is not None and not llm.is_available():
        console.print(
            f"[yellow]{settings.llm.model} not available[/yellow] - writing the beats "
            "from your own words instead."
        )
        llm = None

    images = image_provider(
        settings.image,
        store.project_dir(project.id) / "cache" / "images",
        on_note=lambda note: console.print(f"[yellow]{note}[/yellow]"),
    )
    try:
        timeline = plan_idea(
            intent,
            llm,
            target_duration=duration,
            canvas=_canvas_for(aspect),
            seed=seed or _random_seed(),
        )
        # The text model is released before any image model loads; at 4 GB they cannot
        # share the card. See 'Phase-ordered model loading' in docs/decisions.md.
        if llm is not None:
            llm.unload()
        timeline = resolve_visuals(timeline, project.id, images, store)
        store.save_timeline(project.id, timeline)
    except GenvaiError as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(code=1) from exc

    console.print(
        f"[bold]{project.id}[/bold] v{timeline.version}  "
        f"{len(timeline.scenes)} beats  {timeline.duration:.1f}s  ({images.name} visuals)"
    )
    renderer = FFmpegRenderer(
        settings.render, lambda asset_id: store.asset_path(project.id, asset_id)
    )
    try:
        outputs = render_timeline(timeline, project.id, renderer, store)
    except GenvaiError as exc:
        console.print(f"[red]{exc}[/red]")
        if isinstance(exc, RenderError) and exc.stderr:
            console.print(f"[dim]{exc.stderr}[/dim]")
        raise typer.Exit(code=1) from exc

    for variant, path in outputs.items():
        console.print(f"  [green]{variant}[/green]  {path}")
    console.print(f'[dim]Change it with: genvai edit {project.id} "..."[/dim]')


@app.command()
def promo(
    project: str = typer.Argument(..., help="Project id, or a new name."),
    posters: list[Path] = typer.Argument(..., help="The offer artwork. PNG or JPEG."),
    occasion: str = typer.Option("", "--occasion", help="'Raksha Bandhan offer'."),
    business: str = typer.Option("", "--business", help="Whose offer it is."),
    duration: float = typer.Option(0.0, "--duration", "-d", help="0 fits the offers."),
    aspect: str = typer.Option("9:16", "--aspect", help="9:16, 16:9 or 1:1."),
    featured: int = typer.Option(4, "--featured", help="How many offers get their own beat."),
    seed: int = typer.Option(0, "--seed", help="0 picks a random seed and records it."),
    style: str = typer.Option(
        None, "--style", help=f"Reel shape: {', '.join(STYLES)}. Default: picked by seed."
    ),
    track: Path = typer.Option(None, "--music", help="An audio file of your own to put under it."),
    yes: bool = typer.Option(False, "--yes", "-y", help="Skip the confirmation."),
) -> None:
    """Make a reel from offer artwork - a poster, a price list, a flyer.

    Reads the prices off the artwork, shows you what it read, and only builds once you
    agree. Nothing here invents a price.
    """
    settings = load_settings()
    store = FilesystemStore(settings.projects_dir)
    if style is not None and style not in STYLES:
        console.print(f"[red]No style '{style}'.[/red] Choose from: {', '.join(STYLES)}")
        raise typer.Exit(code=1)
    if track is not None and not track.is_file():
        console.print(f"[red]No audio file at {track}[/red]")
        raise typer.Exit(code=1)
    created = store.create(occasion or f"promo from {posters[0].stem}", project)

    reader = RapidOcrVision()
    if not reader.is_available():
        console.print(
            "[red]OCR is not installed[/red] - it is what reads the prices off your "
            "poster. Install it with: [bold]uv sync --extra ocr[/bold]"
        )
        raise typer.Exit(code=1)

    brief = Brief(occasion=occasion, business=business)
    text: dict[Path, tuple[Rect, ...]] = {}
    with console.status("reading the artwork"):
        for poster in posters:
            boxes = reader.boxes(poster)
            brief = brief.merge(read_brief(boxes) if boxes else Brief())
            with Image.open(poster) as opened:
                text[poster] = text_rects(boxes, *opened.size)
    brief = brief.model_copy(
        update={
            "occasion": occasion or brief.occasion,
            "business": business or brief.business,
        }
    )

    if not brief.is_usable:
        console.print(
            "[red]No prices found on that artwork.[/red] The reader needs printed text; "
            "a photograph of a printed sheet often will not do."
        )
        raise typer.Exit(code=1)

    featured_offers = choose_offers(brief.offers, featured)
    table = Table(show_header=True, header_style="bold", title="what was read")
    table.add_column("")
    table.add_column("Service", overflow="fold")
    table.add_column("Price", justify="right")
    table.add_column("Note", overflow="fold")
    for offer in brief.offers:
        starred = "[green]*[/green]" if offer in featured_offers else " "
        table.add_row(starred, offer.service, offer.price, offer.note)
    console.print(table)
    console.print(
        f"[dim]* gets its own beat. {len(brief.offers)} offers, "
        f"phone {brief.phone or 'not found'}.[/dim]"
    )
    console.print("[yellow]Check every price against the poster before this goes out.[/yellow]")

    if not yes and not typer.confirm("Build the reel from this?", default=True):
        console.print("[dim]Nothing built. Correct the artwork or pass the offers yourself.[/dim]")
        return

    seed = seed or _random_seed()
    structure = structure_for(style, seed)
    timeline = build_promo(
        brief,
        tuple(posters),
        store.project_dir(created.id) / "cache" / "promo",
        canvas=_canvas_for(aspect),
        seed=seed,
        featured=featured,
        style=structure.name,
        store_asset=lambda image, provenance: store.store_asset(
            created.id, image, "image", provenance
        ),
        faces=_faces(settings),
        text=text,
    )
    if duration > 0:
        timeline = _scaled_to(timeline, duration)
    store.save_timeline(created.id, timeline)

    try:
        timeline = _promo_music(created.id, store, settings, track, brief, ask=not yes) or timeline
    except GenvaiError as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(code=1) from exc

    console.print(
        f"[bold]{created.id}[/bold] v{timeline.version}  [cyan]{structure.name}[/cyan] "
        f"({structure.blurb})  {len(timeline.scenes)} beats  {timeline.duration:.1f}s  "
        f"{timeline.canvas.width}x{timeline.canvas.height}"
    )
    renderer = FFmpegRenderer(
        settings.render, lambda asset_id: store.asset_path(created.id, asset_id)
    )
    try:
        outputs = render_timeline(timeline, created.id, renderer, store)
    except GenvaiError as exc:
        console.print(f"[red]{exc}[/red]")
        if isinstance(exc, RenderError) and exc.stderr:
            console.print(f"[dim]{exc.stderr}[/dim]")
        raise typer.Exit(code=1) from exc

    for variant, path in outputs.items():
        console.print(f"  [green]{variant}[/green]  {path}")
    console.print()
    if "full" not in outputs:
        console.print(
            "[dim]Silent cut only - no track was chosen. Attach a trending sound in the "
            "app, or add your own with --music.[/dim]"
        )
    else:
        console.print(
            "[dim]For reach, upload the silent cut and attach a trending sound in the app; "
            "the full cut is for places with no sound library.[/dim]"
        )


@app.command()
def story(
    project: str = typer.Argument(..., help="Project id, or a new name."),
    posters: list[Path] = typer.Argument(..., help="The offer artwork. PNG or JPEG."),
    occasion: str = typer.Option("", "--occasion", help="'Navratri offer'."),
    business: str = typer.Option("", "--business", help="Whose offer it is."),
    arc: str = typer.Option(
        None, "--arc", help=f"Story: {', '.join(ARC_NAMES)}. Default: picked by seed."
    ),
    seed: int = typer.Option(0, "--seed", help="0 picks a random seed and records it."),
    no_llm: bool = typer.Option(False, "--no-llm", help="Use the arc's own copy lines."),
    yes: bool = typer.Option(False, "--yes", "-y", help="Skip the confirmation."),
) -> None:
    """Design the reel before rendering it: a storyboard, and a storybook page to review.

    Reads the prices off the artwork, finds its pictures and colours, and lays out a short
    film from a story arc - every shot's framing, camera move, text, cut and length in
    beats. Nothing is rendered but one frame per shot. Render it with: genvai shoot
    """
    settings = load_settings()
    store = FilesystemStore(settings.projects_dir)
    if arc is not None and arc not in ARC_NAMES:
        console.print(f"[red]No arc '{arc}'.[/red] Choose from: {', '.join(ARC_NAMES)}")
        raise typer.Exit(code=1)
    reader = RapidOcrVision()
    if not reader.is_available():
        console.print("[red]OCR is not installed.[/red] Install it with: uv sync --extra ocr")
        raise typer.Exit(code=1)
    created = store.create(occasion or f"story from {posters[0].stem}", project)

    brief = Brief(occasion=occasion, business=business)
    boxes: list[tuple[Box, ...]] = []
    with console.status("reading the artwork"):
        for poster in posters:
            found = tuple(reader.boxes(poster))
            boxes.append(found)
            brief = brief.merge(read_brief(list(found)) if found else Brief())
    brief = brief.model_copy(
        update={"occasion": occasion or brief.occasion, "business": business or brief.business}
    )
    if not brief.is_usable:
        console.print("[red]No prices found on that artwork.[/red]")
        raise typer.Exit(code=1)

    table = Table(show_header=True, header_style="bold", title="what was read")
    table.add_column("Service", overflow="fold")
    table.add_column("Price", justify="right")
    table.add_column("Note", overflow="fold")
    for offer in brief.offers:
        table.add_row(offer.service, offer.price, offer.note)
    console.print(table)
    console.print("[yellow]Check every price against the poster before this goes out.[/yellow]")
    if not yes and not typer.confirm("Draft a storyboard from this?", default=True):
        return

    seed = seed or _random_seed()
    chosen = arc_for(arc, seed)
    llm: OllamaLLM | None = None if no_llm else OllamaLLM(settings.llm)
    if llm is not None and not llm.is_available():
        llm = None
    with console.status("finding the pictures and drafting the story"):
        material = read_material(created.id, store, tuple(posters), tuple(boxes), _faces(settings))
        board = draft_story(
            created.id,
            store,
            material,
            brief,
            arc=chosen,
            seed=seed,
            llm=llm,
            on_note=lambda note: console.print(f"[yellow]{note}[/yellow]"),
        )
    with console.status("drawing a key frame per shot"):
        book, checks = make_storybook(
            created.id, store, Compositor(resolve_ffmpeg(), width=720, height=1280)
        )

    shots = Table(show_header=True, header_style="bold", title=board.title)
    shots.add_column("#")
    shots.add_column("Role")
    shots.add_column("Beats", justify="right")
    shots.add_column("Shot", overflow="fold")
    shots.add_column("Text", overflow="fold")
    for shot in board.shots:
        shots.add_row(
            shot.id,
            shot.role,
            f"{shot.beats:g}",
            f"{shot.framing} {shot.layout}, {shot.move.replace('_', ' ')}, {shot.cut} in",
            " / ".join(c.text.replace(chr(10), " ") for c in shot.captions),
        )
    console.print(shots)
    console.print(
        f"[bold]{created.id}[/bold] storyboard v{board.version}  [cyan]{board.arc}[/cyan]  "
        f"{len(board.shots)} shots  {board.duration:.1f}s at {board.bpm:.0f} bpm"
    )
    console.print(f"[dim]{board.logline}[/dim]")
    _report_legibility(checks)
    console.print(f"  storybook  {book}")
    console.print(f"[dim]Render it with: genvai shoot {created.id}[/dim]")


@app.command()
def shoot(
    project: str = typer.Argument(..., help="Project id."),
    track: Path = typer.Option(None, "--music", help="An audio file to put under it."),
    preview: bool = typer.Option(False, "--preview", help="Half size, for a quick look."),
) -> None:
    """Render a project's storyboard into the reel."""
    settings = load_settings()
    store = FilesystemStore(settings.projects_dir)
    if track is not None and not track.is_file():
        console.print(f"[red]No audio file at {track}[/red]")
        raise typer.Exit(code=1)
    size = (540, 960) if preview else (1080, 1920)
    compositor = Compositor(resolve_ffmpeg(), width=size[0], height=size[1])

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("{task.completed}/{task.total}"),
        console=console,
    ) as bar:
        task = bar.add_task("shooting", total=None)

        def tick(shot_id: str, index: int, total: int) -> None:
            bar.update(task, total=total, completed=index - 1, description=f"shooting {shot_id}")

        try:
            outputs = shoot_story(project, store, compositor, track=track, on_shot=tick)
        except GenvaiError as exc:
            console.print(f"[red]{exc}[/red]")
            if isinstance(exc, RenderError) and exc.stderr:
                console.print(f"[dim]{exc.stderr}[/dim]")
            raise typer.Exit(code=1) from exc
        bar.update(task, completed=bar.tasks[0].total or 0)

    for variant, path in outputs.items():
        console.print(f"  [green]{variant}[/green]  {path}")


@app.command()
def edit(
    project: str = typer.Argument(..., help="Project id."),
    request: str = typer.Argument(..., help="What to change, in plain language."),
    yes: bool = typer.Option(False, "--yes", "-y", help="Skip the diff confirmation."),
) -> None:
    """Apply a change to an existing project and re-render what moved.

    Shows what will change before doing it. Every version is kept, so 'genvai restore'
    can always put back what was there.
    """
    settings = load_settings()
    store = FilesystemStore(settings.projects_dir)
    llm = OllamaLLM(settings.llm)
    if not llm.is_available():
        console.print(
            f"[red]{settings.llm.model} is not available[/red] - editing needs the model "
            "to read your request. [dim]genvai doctor[/dim] says what is missing."
        )
        raise typer.Exit(code=1)

    try:
        before = store.load_timeline(project)
        after, changes = run_edit(request, before, llm)
    except GenvaiError as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(code=1) from exc

    console.print(f"[bold]{project}[/bold] v{before.version} -> v{after.version}")
    for line in changes:
        console.print(f"  [cyan]{line}[/cyan]")
    console.print(
        f"  [dim]{before.duration:.1f}s -> {after.duration:.1f}s, "
        f"{len(before.scenes)} -> {len(after.scenes)} shots[/dim]"
    )

    if not yes and not typer.confirm("Apply and re-render?", default=True):
        console.print("[dim]Nothing changed.[/dim]")
        return

    store.save_timeline(project, after)
    plan = plan_render(after, project, store)
    console.print(f"[dim]{plan.summary()}[/dim]")
    renderer = FFmpegRenderer(settings.render, lambda asset_id: store.asset_path(project, asset_id))
    for variant, path in render_timeline(after, project, renderer, store).items():
        console.print(f"  [green]{variant}[/green]  {path}")
    console.print(f"[dim]Undo with: genvai restore {project} {before.version}[/dim]")


@app.command()
def render(
    project: str = typer.Argument(..., help="Project id."),
    version: int | None = typer.Option(None, "--version", help="Defaults to current."),
    preview: bool = typer.Option(False, "--preview", help="Fast low-resolution proxy."),
    dry_run: bool = typer.Option(False, "--dry-run", help="Report the work, encode nothing."),
) -> None:
    """Render a timeline to MP4.

    Only scenes whose content changed are re-encoded, so re-rendering after an edit
    costs a fraction of the first pass.
    """
    settings = load_settings()
    store = FilesystemStore(settings.projects_dir)
    try:
        timeline = store.load_timeline(project, version)
    except GenvaiError as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(code=1) from exc

    plan = plan_render(timeline, project, store)
    console.print(
        f"[bold]{project}[/bold] v{timeline.version}  "
        f"{timeline.duration:.1f}s  {len(timeline.scenes)} scenes  "
        f"{timeline.canvas.width}x{timeline.canvas.height}  -  {plan.summary()}"
    )
    if dry_run:
        return

    renderer = FFmpegRenderer(settings.render, lambda asset_id: store.asset_path(project, asset_id))
    try:
        outputs = render_timeline(timeline, project, renderer, store, preview=preview)
    except GenvaiError as exc:
        console.print(f"[red]{exc}[/red]")
        if isinstance(exc, RenderError) and exc.stderr:
            console.print(f"[dim]{exc.stderr}[/dim]")
        raise typer.Exit(code=1) from exc

    for variant, path in outputs.items():
        console.print(f"  [green]{variant}[/green]  {path}")
    if "narration_only" in outputs:
        console.print()
        console.print(
            "[dim]Upload the narration_only cut when you plan to attach a trending "
            "sound in the app - a baked-in track forfeits that reach.[/dim]"
        )


@app.command(name="list")
def list_projects() -> None:
    """List projects."""
    settings = load_settings()
    projects = FilesystemStore(settings.projects_dir).list_projects()
    if not projects:
        console.print(
            f"[dim]No projects under {settings.projects_dir.resolve()}. "
            "Start one with 'genvai add <name> <files...>'.[/dim]"
        )
        return

    table = Table(show_header=True, header_style="bold")
    table.add_column("Project")
    table.add_column("Version", justify="right")
    table.add_column("Updated")
    table.add_column("Intent", overflow="fold")
    for item in projects:
        table.add_row(item.id, str(item.current_version), item.updated_at, item.intent)
    console.print(table)


@app.command()
def restore(
    project: str = typer.Argument(..., help="Project id."),
    version: int = typer.Argument(..., help="Version to restore."),
) -> None:
    """Restore an earlier timeline version. Nothing is ever lost, so this always works."""
    settings = load_settings()
    store = FilesystemStore(settings.projects_dir)
    try:
        wanted = store.load_timeline(project, version)
    except GenvaiError as exc:
        available = ", ".join(f"v{v}" for v in store.versions(project)) or "none"
        console.print(f"[red]{exc}[/red]  [dim]Available: {available}[/dim]")
        raise typer.Exit(code=1) from exc

    # Restoring copies forward rather than rewinding, so the history stays append-only
    # and the version you came from is still there to go back to.
    restored = wanted.model_copy(update={"version": max(store.versions(project)) + 1})
    store.save_timeline(project, restored)
    console.print(f"[bold]{project}[/bold] restored v{version} as v{restored.version}")
    console.print(f"[dim]Render it with: genvai render {project}[/dim]")


@app.command()
def music(
    project: str = typer.Argument(..., help="Project id."),
    mood: str = typer.Option("", "--mood", help="What it should sound like."),
    approve_id: str = typer.Option(None, "--approve", help="Candidate id to use."),
    track: Path = typer.Option(None, "--file", help="Use an audio file of your own."),
    none: bool = typer.Option(False, "--none", help="Render without music."),
) -> None:
    """Suggest tracks, or approve one.

    Nothing is fetched until you name a candidate here. With no options it shows what was
    suggested last time.
    """
    settings = load_settings()
    store = FilesystemStore(settings.projects_dir)
    provider = LocalMusicProvider(settings.music)

    try:
        if none:
            timeline = decline_music(project, store)
            console.print(f"[bold]{project}[/bold] v{timeline.version}  no music")
            return

        if track is not None:
            timeline = attach_music(project, store, FFmpegBeatDetector(resolve_ffmpeg()), track)
            console.print(f"[bold]{project}[/bold] v{timeline.version}  music set - {track.name}")
            console.print(f"[dim]Render it with: genvai render {project}[/dim]")
            return

        if approve_id:
            timeline = approve_music(
                project,
                store,
                provider,
                FFmpegBeatDetector(resolve_ffmpeg()),
                approve_id,
                confirmed=True,
            )
            beats = timeline.music.beat_map
            tempo = f"{beats.bpm:.0f} BPM, cuts aligned" if beats else "no beat detected"
            console.print(f"[bold]{project}[/bold] v{timeline.version}  music set - {tempo}")
            console.print(f"[dim]Render it with: genvai render {project}[/dim]")
            return

        timeline = (
            suggest_music(project, store, provider, mood=mood)
            if mood
            else (store.load_timeline(project))
        )
    except GenvaiError as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(code=1) from exc

    if not timeline.music.candidates:
        console.print(
            f'[dim]No suggestions yet. Try: genvai music {project} --mood "calm piano"[/dim]'
        )
        return

    table = Table(show_header=True, header_style="bold")
    table.add_column("Id")
    table.add_column("Track", overflow="fold")
    table.add_column("Licence")
    for candidate in timeline.music.candidates:
        table.add_row(candidate.id, candidate.title, candidate.licence)
    console.print(table)
    console.print(
        f"[dim]Nothing has been downloaded. Choose one with: "
        f"genvai music {project} --approve <id>[/dim]"
    )


# --------------------------------------------------------------------------- helpers


def _not_yet(command: str, milestone: str) -> None:
    console.print(
        f"[yellow]'{command}' is not implemented yet[/yellow] - scheduled for {milestone}.\n"
        "[dim]See TODO.md for what is built and what comes next.[/dim]"
    )
    raise typer.Exit(code=2)


def _promo_music(
    project: str,
    store: FilesystemStore,
    settings: Settings,
    track: Path | None,
    brief: Brief,
    *,
    ask: bool,
) -> Timeline | None:
    """Put a track under a promo, if there is one to put.

    A file named with --music is used as given. Otherwise, when a music library exists
    and there is someone to ask, its matches are listed and one may be picked - nothing
    is ever chosen on the user's behalf. None means the reel stays silent.
    """
    detector = FFmpegBeatDetector(resolve_ffmpeg())
    if track is not None:
        return attach_music(project, store, detector, track, gain_db=SOLE_TRACK_DB)

    provider = LocalMusicProvider(settings.music)
    if not ask or not provider.is_available():
        return None

    mood = f"{brief.occasion} festive upbeat".strip()
    try:
        suggested = suggest_music(project, store, provider, mood=mood)
    except GenvaiError:
        return None

    table = Table(show_header=True, header_style="bold", title="tracks in your library")
    table.add_column("Id")
    table.add_column("Track", overflow="fold")
    table.add_column("Licence")
    for candidate in suggested.music.candidates:
        table.add_row(candidate.id, candidate.title, candidate.licence)
    console.print(table)
    choice = typer.prompt("Track id to use (blank for none)", default="", show_default=False)
    if not choice.strip():
        return decline_music(project, store)
    return approve_music(
        project, store, provider, detector, choice.strip(), confirmed=True, gain_db=SOLE_TRACK_DB
    )


def _report_legibility(checks: dict[str, tuple[tuple[str, str, float, str], ...]]) -> None:
    """Say whether every caption can be read, and list any that cannot."""
    lines = [
        (shot, text, ratio, fix)
        for shot, captions in checks.items()
        for text, role, ratio, fix in captions
        if role != "price"
    ]
    if not lines:
        return
    weak = [line for line in lines if line[2] < 4.5]
    fixed = sum(1 for line in lines if line[3] != "as designed")
    worst = min(line[2] for line in lines)
    if weak:
        console.print(f"[red]{len(weak)} caption(s) may be hard to read:[/red]")
        for shot, text, ratio, _ in weak:
            console.print(f"  {shot}  {ratio:.1f}:1  {text!r}")
    else:
        console.print(
            f"[green]All {len(lines)} captions readable[/green] - lowest contrast "
            f"{worst:.1f}:1; {fixed} needed a plate or dark ink."
        )


def _faces(settings: Settings) -> YunetFaces:
    return YunetFaces(settings.faces.model_path, min_confidence=settings.faces.min_confidence)


def _canvas_for(aspect: str) -> Canvas:
    """Turn a ratio into a canvas. Vertical is the default because reels are."""
    presets = {
        "9:16": Canvas(width=1080, height=1920),
        "16:9": Canvas(width=1920, height=1080),
        "1:1": Canvas(width=1080, height=1080),
    }
    if aspect not in presets:
        console.print(f"[yellow]unknown aspect '{aspect}', using 9:16[/yellow]")
    return presets.get(aspect, presets["9:16"])


def _scaled_to(timeline: Timeline, target: float) -> Timeline:
    """Stretch or compress every beat to hit a requested length.

    Uniform, so the pacing the beats were given is preserved rather than one shot being
    made to absorb the whole difference.
    """
    if timeline.duration <= 0:
        return timeline
    factor = target / timeline.duration
    return timeline.model_copy(
        update={
            "scenes": tuple(
                s.model_copy(update={"duration": max(0.4, s.duration * factor)})
                for s in timeline.scenes
            )
        }
    )


def _random_seed() -> int:
    """A recorded seed, so the same reel can be rebuilt exactly."""
    return secrets.randbelow(2**31)


def _probe_ffmpeg() -> tuple[str, str]:
    system = shutil.which("ffmpeg")
    if system:
        return "ok", f"system: {system}"
    try:
        import imageio_ffmpeg

        return "ok", f"bundled: {imageio_ffmpeg.get_ffmpeg_exe()}"
    except Exception as exc:  # noqa: BLE001 - diagnostics must never raise
        return "missing", f"no system ffmpeg and bundled lookup failed: {exc}"


def _probe_ollama(base_url: str, model: str) -> tuple[str, str]:
    try:
        import httpx

        response = httpx.get(f"{base_url}/api/tags", timeout=3.0)
        response.raise_for_status()
        names = [m.get("name", "") for m in response.json().get("models", [])]
    except Exception as exc:  # noqa: BLE001
        return "unreachable", f"{base_url} - {type(exc).__name__}. Start it with 'ollama serve'."

    if not names:
        return "no models", f"{base_url} - pull one: ollama pull {model}"
    # Ollama reports an untagged model as "name:latest", so a bare name in the config
    # matches it. Comparing the strings directly reports a working model as missing.
    if model in names or f"{model}:latest" in names:
        return "ok", f"{model} available ({len(names)} model(s))"
    return "model missing", f"has {', '.join(sorted(names)[:3])} - pull: ollama pull {model}"


def _cuda_detail() -> str:
    try:
        import torch

        if not torch.cuda.is_available():
            return "CUDA unavailable - image generation would run on CPU (slow)"
        name = torch.cuda.get_device_name(0)
        vram_gb = torch.cuda.get_device_properties(0).total_memory / 1024**3
        note = " - set GENVAI_IMAGE__LOW_VRAM=true" if vram_gb < 6 else ""
        return f"{name}, {vram_gb:.1f} GB{note}"
    except Exception as exc:  # noqa: BLE001
        return f"probe failed: {exc}"


if __name__ == "__main__":
    app()
