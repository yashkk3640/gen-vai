# Setup

The rule: **`git clone` + `uv sync`, nothing else.**

## Install

```bash
git clone <repo> gen-vai && cd gen-vai
uv sync
uv run genvai doctor
```

`uv` installs Python if it is missing. ffmpeg arrives as a dependency, so there is no
system install to do. `doctor` reports what is available and what will fall back.

## Optional extras

Install only what the machine can use:

```bash
uv sync --extra image   # torch + diffusers (~3 GB, needs a GPU)
uv sync --extra tts     # Piper narration (CPU)
uv sync --extra asr     # faster-whisper
uv sync --extra dev     # pytest, ruff, mypy
```

Without them you get procedural visuals and silent audio. Nothing breaks.

## LLM

```bash
ollama serve
ollama pull llama3.2       # ~2 GB, the default
```

Model size matters more than you would expect on a 4 GB card, and these are measured
rather than guessed:

| Model | Warm request | Verdict |
| --- | --- | --- |
| `llama3.2` (3B) | ~12s | fits in VRAM; the default |
| 9B-class | >180s, times out | spills to CPU, unusable here |

Bigger is better where there is VRAM for it. Point at something else by copying
`.env.example` to `.env` and setting `GENVAI_LLM__MODEL`.

## Another machine

Committed and travels: `pyproject.toml`, `uv.lock`, `.python-version`, `src/`, `docs/`.
Not committed: `.venv/` (rebuilt), `.env` (machine-specific), `projects/` (large).

```bash
git clone <repo> && cd gen-vai && uv sync
cp .env.example .env
uv run genvai doctor
```

`uv.lock` pins exact versions, so both machines resolve the same graph.

Project folders are self-contained and path-free — copy `projects/<id>/` across and it
reopens. They are not in git because they hold large binaries.

## When something is wrong

| Symptom | Fix |
| --- | --- |
| `ffmpeg not found` | `uv sync`. A system ffmpeg on PATH takes precedence if present |
| Ollama unreachable | `ollama serve`, check `curl localhost:11434/api/tags` |
| CUDA out of memory | Set `GENVAI_IMAGE__LOW_VRAM=true` and reduce resolution |
| torch install is huge | It is ~3 GB. Skip `--extra image` and use procedural visuals |
