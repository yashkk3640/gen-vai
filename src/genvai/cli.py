"""Command line surface. Thin: parsing, wiring, and presentation only.

Every command delegates to `genvai.pipeline`. No editorial logic lives here, so an
HTTP API can be added over the same core without moving anything.
"""

import importlib.util
import shutil
import sys
from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

from genvai import __version__
from genvai.config import load_settings

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
        "to procedural visuals and silent audio. See docs/05-setup.md.[/dim]"
    )


@app.command()
def add(
    project: str = typer.Argument(..., help="Project id, or a new name to create one."),
    files: list[Path] = typer.Argument(..., help="Photos and clips. Folders are walked."),
) -> None:
    """Import a camera roll and analyse it.

    The slow step, cached by content hash - re-adding the same files is free.
    """
    _not_yet("add", "M2")


@app.command()
def reel(
    project: str = typer.Argument(..., help="Project id."),
    duration: float = typer.Option(30.0, "--duration", "-d", help="Target length in seconds."),
    intent: str = typer.Option("", "--intent", help="Steer it: 'focus on the food'."),
    aspect: str = typer.Option("9:16", "--aspect", help="9:16, 16:9 or 1:1."),
    seed: int = typer.Option(0, "--seed", help="0 picks a random seed and records it."),
) -> None:
    """Build a reel from the project's media."""
    _not_yet("reel", "M3")


@app.command()
def media(
    project: str = typer.Argument(..., help="Project id."),
    unused: bool = typer.Option(False, "--unused", help="Only what did not make the cut."),
) -> None:
    """List analysed media with quality scores and the spans that were found."""
    _not_yet("media", "M2")


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
    """Apply a change to an existing project and re-render what moved."""
    _not_yet("edit", "M4")


@app.command()
def render(
    project: str = typer.Argument(..., help="Project id."),
    version: int | None = typer.Option(None, "--version", help="Defaults to current."),
    preview: bool = typer.Option(False, "--preview", help="Fast low-resolution proxy."),
) -> None:
    """Render a timeline to MP4."""
    _not_yet("render", "M1")


@app.command(name="list")
def list_projects() -> None:
    """List projects."""
    _not_yet("list", "M1")


@app.command()
def restore(
    project: str = typer.Argument(..., help="Project id."),
    version: int = typer.Argument(..., help="Version to restore."),
) -> None:
    """Restore an earlier timeline version. Nothing is ever lost, so this always works."""
    _not_yet("restore", "M4")


@app.command()
def music(
    project: str = typer.Argument(..., help="Project id."),
    approve: str | None = typer.Option(None, "--approve", help="Candidate id to download."),
) -> None:
    """Show suggested tracks, or approve one for download.

    Nothing is fetched until a candidate is approved here.
    """
    _not_yet("music", "M5")


# --------------------------------------------------------------------------- helpers


def _not_yet(command: str, milestone: str) -> None:
    console.print(
        f"[yellow]'{command}' is not implemented yet[/yellow] - scheduled for {milestone}.\n"
        "[dim]See docs/04-roadmap.md for what is built and what comes next.[/dim]"
    )
    raise typer.Exit(code=2)


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
