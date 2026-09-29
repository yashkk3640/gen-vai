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
genvai music trip --file song.mp3       # or a track of your own
genvai edit trip "drop the third clip"  # change it in plain English

genvai create "why compound interest matters" -d 20   # no footage needed

genvai promo gk offer.png --occasion "Diwali offer"   # a reel from a price list
genvai promo gk offer.png --style countdown --music song.mp3

genvai story gk offer.png --occasion "Navratri offer" --arc glow-up   # design it first
genvai shoot gk --music song.mp3                                      # then render it
```

`promo` has five shapes - `classic`, `question`, `from`, `menu-first`, `countdown` -
picked by seed unless you name one, so two reels for one client do not come out alike.

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
| `promo` | A reel from offer artwork — reads the prices off it | yes |
| `story` | Design a promo as a short film: storyboard + storybook page to review | yes |
| `shoot` | Render the storyboard — layered, animated, cinematic | yes |

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
- **Bring your own music.** Nothing ships with the project. Without a track, `promo`
  exports only the silent cut rather than a "full" one with nothing in it.
- **`promo` shows you what it read before building.** OCR pairs a service to its
  price by position, which is imperfect on unusual layouts. Check every price.

Fuller list in [TODO.md](TODO.md).

## Status

**All ten milestones are done** — import, select, reframe, beat-align, render,
edit, undo, and idea mode. What is left is refinement; see [TODO.md](TODO.md).

## Docs

Picking this up fresh? Start with **[CONTEXT.md](CONTEXT.md)** — what was measured, what
was tried and rejected, and the gotchas that cost time.

[docs/](docs/) — [vision](docs/vision.md) · [architecture](docs/architecture.md) ·
[timeline format](docs/timeline.md) · [setup](docs/setup.md) ·
[decisions](docs/decisions.md) · [backlog](docs/backlog.md)
