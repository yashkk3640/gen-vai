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
`ollama pull llama3.2`.

## Use

```bash
genvai add trip ~/Pictures/trip/        # import and analyse a camera roll
genvai reel trip --duration 30          # pick, trim, sequence, caption
genvai music trip --mood "calm piano"   # suggest tracks (downloads nothing)
genvai music trip --approve <id>        # use one, and cut to its beat
genvai edit trip "drop the third clip"  # change it in plain English

genvai create "why compound interest matters" -d 20   # no footage needed
```

| Command | What it does | Ready |
| --- | --- | --- |
| `doctor` | What this machine can do | yes |
| `list` | Your projects | yes |
| `render` | Timeline → MP4. `--preview` for a fast proxy, `--dry-run` to see the work | yes |
| `add` | Import and analyse media | yes |
| `media` | Show what was found, with scores | yes |
| `reel` | Build a reel — picks, trims, orders, captions | yes |
| `edit` | Change it in plain English, with a diff first | yes |
| `restore` | Put back an earlier version | yes |
| `music` | Suggest a track, approve one, align cuts to its beat | yes |
| `create` | No footage? Make one from a description alone | yes |

A project is a plain folder under `projects/`. Open it and you can see every asset, every
version of the edit, and every render. Copy it to another machine and it still works.

## Worth knowing before you trust it

- **Captions are invented, not observed.** The model never sees your pictures, only
  measurements, so it writes plausible copy from your `--intent`. Ask for "a weekend by
  the sea" and you may get "seagulls soar" whether or not there is one.
- **Avoid `genvai edit --yes`.** A small model occasionally adds a command you did not
  ask for; the diff it prints before applying is the safeguard.
- **Use a 3B model.** A 9B one spills off a 4 GB card and times out - measurements in
  [docs/setup.md](docs/setup.md).
- **Bring your own music.** Nothing ships with the project.

Fuller list in [TODO.md](TODO.md).

## Status

**All eight milestones are done** — import, select, reframe, beat-align, render,
edit, undo, and idea mode. What is left is refinement; see [TODO.md](TODO.md).

## Docs

[docs/](docs/) — [vision](docs/vision.md) · [architecture](docs/architecture.md) ·
[timeline format](docs/timeline.md) · [setup](docs/setup.md) ·
[decisions](docs/decisions.md) · [backlog](docs/backlog.md)
