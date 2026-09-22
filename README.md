# gen-vai

**Turn a camera roll into a reel. Locally.**

Point it at your photos and short clips. It picks the good ones, trims each to its best
moment, sequences them, and renders a vertical video. Then you say what to change and it
patches the result instead of starting over.

Nothing is uploaded. Ollama makes the editorial calls, ffmpeg does the render.

## Install

Needs [uv](https://docs.astral.sh/uv/). It brings its own Python, and ffmpeg ships with
the dependencies — nothing else to install.

```bash
git clone <repo> gen-vai && cd gen-vai
uv sync
uv run genvai doctor
```

`doctor` reports what your machine has. Anything missing degrades gracefully rather than
breaking. For the LLM features, install [Ollama](https://ollama.com) and
`ollama pull qwen2.5:7b-instruct`.

## Use

```bash
genvai add trip ~/Pictures/trip/        # import and analyse a camera roll
genvai reel trip --duration 30          # pick, trim, sequence, caption
genvai edit trip "drop the third clip"  # change it
genvai render trip                      # re-render (only what changed)
```

| Command | What it does | Ready |
| --- | --- | --- |
| `doctor` | What this machine can do | yes |
| `list` | Your projects | yes |
| `render` | Timeline → MP4. `--preview` for a fast proxy, `--dry-run` to see the work | yes |
| `add` | Import and analyse media | M2 |
| `media` | Show what was found, with scores | M2 |
| `reel` | Build a reel | M3 |
| `edit` | Change it in plain English | M4 |
| `music` | Suggest and approve a track | M5 |

A project is a plain folder under `projects/`. Open it and you can see every asset, every
version of the edit, and every render. Copy it to another machine and it still works.

## Status

**M1 done** — timelines render. Ingest and selection are next. See [TODO.md](TODO.md).

## Docs

[docs/](docs/) — [vision](docs/vision.md) · [architecture](docs/architecture.md) ·
[timeline format](docs/timeline.md) · [setup](docs/setup.md) ·
[decisions](docs/decisions.md) · [backlog](docs/backlog.md)
