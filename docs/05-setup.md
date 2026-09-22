# 05 - Setup and portability

Portability was an explicit requirement: the project must move to another machine
cleanly. The rule is **`git clone` + `uv sync` and nothing else.**

## Prerequisites

| | Required | Notes |
| --- | --- | --- |
| Python 3.11-3.12 | yes | pinned by `.python-version`; `uv` installs it if missing |
| [uv](https://docs.astral.sh/uv/) | yes | dependency + venv manager |
| ffmpeg | **no** | bundled via `imageio-ffmpeg` into the venv |
| [Ollama](https://ollama.com) | for LLM features | any machine with `localhost:11434` |
| NVIDIA GPU | for image generation | falls back to procedural visuals without one |

## First-time setup

```bash
git clone <repo> gen-vai
cd gen-vai
uv sync                      # creates .venv, installs base deps + ffmpeg
uv run genvai --help
```

Optional extras, installed only where the hardware supports them:

```bash
uv sync --extra image        # torch + diffusers (~3 GB, needs a GPU)
uv sync --extra tts          # Piper narration (CPU)
uv sync --extra asr          # faster-whisper (footage mode)
uv sync --extra dev          # pytest, ruff, mypy
```

A machine without a GPU simply runs `uv sync` and gets procedural visuals.
Nothing breaks; the pipeline degrades.

## LLM setup

```bash
ollama serve                         # if not already running as a service
ollama pull qwen2.5:7b-instruct      # ~4.7 GB, good structured-output behaviour
```

On 4 GB VRAM a 7B Q4 model fits with room for context. Smaller alternatives if it
is tight: `qwen2.5:3b-instruct`, `llama3.2:3b`.

Point the tool at a different model or host by copying `.env.example` to `.env`:

```
GENVAI_LLM__MODEL=qwen2.5:3b-instruct
GENVAI_LLM__BASE_URL=http://localhost:11434
```

## Verifying the install

```bash
uv run genvai doctor
```

Reports Python and package versions, the resolved ffmpeg binary path, whether
Ollama is reachable and which models it has, CUDA availability and VRAM, and which
optional extras are installed. Run this first on any new machine - it is the
fastest way to see what is and is not available.

## Moving to another system

Committed, so it travels: `pyproject.toml`, `uv.lock`, `.python-version`, `src/`,
`docs/`, `.env.example`.

Not committed, and intentionally so: `.venv/` (rebuilt by `uv sync`), `.env`
(machine-specific), `projects/` (large binaries), `models/` (multi-GB weights).

```bash
# on the new machine
git clone <repo> && cd gen-vai
uv sync
cp .env.example .env         # adjust if the model or host differs
uv run genvai doctor
```

`uv.lock` pins exact versions, so the second machine resolves to the same
dependency graph as the first.

### Moving a project's work too

Project directories are self-contained and path-free, so a project folder can be
copied to another machine and reopened:

```bash
cp -r projects/<id> /path/to/other/machine/gen-vai/projects/
```

They are not in git because they hold large binaries. Use a normal file copy, or
add a specific project directory with `git add -f` if you want to version one.

## Troubleshooting

**`ffmpeg not found`** - `imageio-ffmpeg` should provide it; check with
`uv run python -c "import imageio_ffmpeg; print(imageio_ffmpeg.get_ffmpeg_exe())"`.
A system ffmpeg on PATH takes precedence if present.

**Ollama unreachable** - start `ollama serve`, confirm with
`curl http://localhost:11434/api/tags`.

**CUDA out of memory** - expected at 4 GB if both models load at once. The pipeline
is phase-ordered to prevent this; if it still occurs, set
`GENVAI_IMAGE__LOW_VRAM=true` and reduce resolution.

**`torch` install is slow or huge** - it is ~3 GB. Skip `--extra image` and use
procedural visuals if the machine has no GPU.
