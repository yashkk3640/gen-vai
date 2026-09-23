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
from rich.console import Console
from rich.progress import BarColumn, Progress, SpinnerColumn, TextColumn
from rich.table import Table

from genvai import __version__
from genvai.adapters.analyzer import FrameAnalyzer
from genvai.adapters.beat import FFmpegBeatDetector
from genvai.adapters.ffmpeg import FFmpegRenderer, resolve_ffmpeg
from genvai.adapters.fs_store import FilesystemStore
from genvai.adapters.music_local import LocalMusicProvider
from genvai.adapters.ollama import OllamaLLM
from genvai.config import load_settings
from genvai.errors import GenvaiError, RenderError
from genvai.pipeline.edit import edit as run_edit
from genvai.pipeline.ingest import ingest, summarise
from genvai.pipeline.music import approve as approve_music
from genvai.pipeline.music import decline as decline_music
from genvai.pipeline.music import suggest as suggest_music
from genvai.pipeline.render import plan_render
from genvai.pipeline.render import render as render_timeline
from genvai.pipeline.select import make_reel
from genvai.timeline import Canvas

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
            tuple(files), project, FrameAnalyzer(ffmpeg), store, ffmpeg, on_progress=tick
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
    aspect: str = typer.Option("9:16", "--aspect", help="9:16, 16:9 or 1:1."),
    seed: int = typer.Option(0, "--seed", help="0 picks a random seed and records it."),
) -> None:
    """Idea mode: generate from a description, no footage. Parked - see roadmap."""
    _not_yet("create", "M7")


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
    if model in names:
        return "ok", f"{model} available ({len(names)} model(s))"
    return "model missing", f"has {', '.join(names[:3])} - pull with: ollama pull {model}"


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
