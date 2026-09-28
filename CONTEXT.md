# Context handoff

Everything a fresh session needs. Written 2026-09-28, at commit `fbe612b`.

Read [README.md](README.md) for usage and [TODO.md](TODO.md) for what is open. This file
carries what those two do not: why things are the way they are, what was measured, and
what has already been tried and rejected.

---

## What this is

`gen-vai` — a local video editor. Three ways in:

| Command | Input | Status |
| --- | --- | --- |
| `genvai add` + `reel` | a camera roll of photos and short clips | works |
| `genvai promo` | designed artwork — a poster, a price list | works |
| `genvai create` | a text description, no footage | works |

Plus `edit` (change it in plain English), `restore`, `music`, `render`, `media`, `list`,
`doctor`. Nine milestones M0–M8 complete. **545 tests passing**, ruff clean.

Everything runs locally. Nothing is uploaded. The only network call is a music download,
and that is confirmation-gated.

---

## The machine this was built for

Facts that shaped almost every decision:

- **RTX 3050 Laptop, 4 GB VRAM.** This is the binding constraint throughout.
- Python 3.11.9, `uv` 0.11.2, Windows 11, Git Bash available.
- ffmpeg is **bundled** via `imageio-ffmpeg` — a full gyan.dev build with libass,
  libfreetype, x264, zoompan, xfade. No system install needed.
- Ollama runs locally with ~10 models pulled. `llama3.2` (3B) is the working one.

### Measured, not assumed

| Thing | Result |
| --- | --- |
| 9B model for planning | **times out at 180s** — spills to CPU, unusable |
| `llama3.2` (3B) warm | ~12s per request — the default |
| Ollama `keep_alive` at `0s` | reloads every call, ~10s wasted each time. Now `5m` |
| `moondream` reading a poster | 138s, **invented prices**, dollars on a rupee poster |
| RapidOCR reading the same poster | **4.4s, every price correct**, conf 0.94–1.00 |
| Stable Diffusion adapter | **written but never run** — torch is not installed |

---

## Decisions that should not be re-litigated

Full list in [docs/decisions.md](docs/decisions.md). The load-bearing ones:

**Camera roll over text-to-video.** Generating video from a prompt is the crowded end of
the market and where 4 GB loses hardest. Real footage sidesteps image quality entirely.
Idea mode exists but is not the pitch.

**The timeline is the source of truth.** Immutable, versioned JSON. Every edit is a pure
`Timeline → Timeline` transform; rendering is the only side effect. This is what makes
"change shot 2" a function on data rather than a re-generation.

**Typed edit operations, not timeline rewrites.** A model handed a whole timeline drifts
on fields nobody asked it to touch.

**Small schemas for small models.** Established in M3 and confirmed twice since. Asking a
3B model for order + duration + role + caption produced two hooks and a rationale of
`"6"`. Cutting it to order + caption + one hook id gave 3/3 valid. **Never ask a model
for a number that has already been measured.**

**OCR, not a vision model, for reading prices.** A language model asked for a number
always produces one. OCR either resolves the characters or does not. See the measurement
above — this was tried the other way first and it hallucinated badly.

**Music is suggested, never fetched implicitly.** `search` → `approve` cannot be
collapsed; a single call has no moment at which to ask. `candidates_ready` is
deliberately unrenderable.

**Phase-ordered model loading.** 4 GB cannot hold a text model and an image model at
once. All LLM work completes before any image model loads. A correctness constraint.

---

## Gotchas learned the hard way

Each of these cost real time. They are in the code as comments, but collected here:

- **ffmpeg `crop` has no `eval` option** and needs none — its x/y expressions are already
  re-evaluated every frame while w/h are fixed once. Passing `eval=frame` fails the graph.
- **Inline `text=` in a filtergraph is unusable.** Colons, quotes and newlines all mean
  something to the parser; a wrapped caption rendered its line break as a literal "n".
  Use `textfile=`.
- **`Path.write_text` produces CRLF on Windows** and `drawtext` renders the stray
  carriage return as extra leading, tripling line gaps. The same code looks fine on
  Linux. Always pass `newline="\n"`.
- **Writing files with the platform default encoding corrupts them.** Two docs ended up
  with a stray `0x97` cp1252 em-dash and became invalid UTF-8. Always
  `encoding="utf-8"`.
- **drawtext has no colour-emoji fallback** — an emoji renders as a tofu box.
- **A long token with no spaces overflows the frame silently.** `_wrap` now hard-breaks.
- **Caption wrap width must come from the canvas**, not the style: font size is a
  fraction of *height* but the line must fit the *width*.
- **`thumbnail` only shrinks, never enlarges.** A motif smaller than its target box
  silently stayed at source size.
- **`ollama pull` can exit 0 on a network failure.** Check the output, not the code.
- **Ollama reports an untagged model as `name:latest`** — a bare config name needs both
  forms checked.
- Bash heredocs in this environment mangle `\n` escapes in Python strings. Use the
  Write/Edit tools for anything containing escapes.

---

## Architecture in one screen

Ports and adapters around a pure core. Dependencies point inward.

```
CLI → pipeline (orchestration) → PURE CORE → ports (Protocols) → adapters
```

**Pure modules** (no I/O, heavily tested): `timeline.py` (the schema), `ops.py` (24 edit
operations), `media.py` (ingest observations), `analysis.py` (sharpness, exposure, shake,
spans, perceptual hash), `beats.py`, `reframe.py`, `regions.py`, `palette.py`, `ocr.py`,
`promo.py`, `fingerprint.py`.

**All protocols live in `ports.py`** — one file, so the whole boundary reads in a sitting.

**A project is a directory.** `projects/<id>/` holds `project.json`, `media.json`,
`timelines/v1.json…` (append-only), content-addressed `assets/`, a fingerprinted
`cache/segments/`, and `renders/`. No database. Copy the folder to another machine and it
opens.

**Incremental rendering** is the reason the edit loop feels fast: each scene has a
fingerprint over everything affecting its pixels, so changing shot 2 of twelve re-encodes
one segment.

---

## Numbers that were tuned, and against what

All of these were set by measurement, but **measured against synthetic fixtures and two
real posters** — not against a real camera roll. Expect them to move.

| Constant | Where | Value | Why |
| --- | --- | --- | --- |
| `DUPLICATE_THRESHOLD` | `analysis.py` | 12 of 256 | true dupes measure 0–6; closest false pair measured 20. Was 40 and wrongly merged two photos |
| `MIN_STRUCTURE` | `analysis.py` | 24 bits | a near-black frame sets ~6 and matches anything |
| `SHARPNESS_MIDPOINT` | `analysis.py` | 300 | Laplacian variance scoring 0.5 |
| `EDGE_SECONDS` | `analysis.py` | 0.6 | phone clips are shakiest with a thumb on the button |
| `DRIFT_THRESHOLD` | `reframe.py` | 0.06 | below this a moving crop reads as a render fault |
| `BACKGROUND_PERCENTILE` | `reframe.py` | 60 | without it a flat pedestal drags focus to centre |
| `SNAP_TOLERANCE` | `select.py` | 0.25 beat | beyond it, alignment cuts the good moment out |
| `MAX_SPEED_NUDGE` | `select.py` | 0.12 | imperceptible on a 2s shot |
| `SPANNING` | `ocr.py` | 1.4× | a price pill taller than the label spans a wrapped label |
| `INSET` | `pipeline/promo.py` | 0.14 | trims the caption beside a poster photo |

---

## Client work in progress

`beauty parlor/` is **gitignored** — client artwork and finished cuts stay off a public
repo. Four things live there:

- `gracy skin offer.png`, `gracy nail art.png` — Raksha Bandhan offer posters for Gracy
  Khatri (beauty) and Nails by Gracy. Phone 7043641428. Prices are already discounted.
- `reel/` — four MP4s: a hand-built combined reel and an auto-built nails reel, each in
  full and silent cuts.
- `build_backdrops.py`, `build_reel.py`, `HOW-THIS-REEL-WAS-MADE.md` — the hand-authored
  method that produced the first reel. Still the higher-quality output; edit the `BEATS`
  list to change copy.

Both posters now read completely: **4/4 rows on nails, 10/10 on skin**.

### What the client said last

Three things, all open:

1. **The two reels feel the same** — same concept, same sequence, same shape. Wants
   something new and pleasant. The eight-beat structure in `pipeline/promo.py` is
   hardcoded: hook → prices → menu card → CTA. It needs variety, or several templates.
2. **The hand-built reel had better backdrops** — specifically the ones with a person in
   them. The automatic region finder scores on "picture-likeness" and does not know a
   face from a flower. Faces in the saliency map would fix both this and `face_area`.
3. **The reels have no sound** — confirmed: the "full" cut measures −91 dB, digital
   silence, because `promo` never asks about music. `genvai music` exists but promo does
   not invoke it, and no music library is configured.

---

## Open items, in the order I would do them

1. **Make promo reels varied.** Several beat structures, chosen by seed or flag. Right
   now every promo has the same shape and it shows immediately when you make two.
2. **Wire music into `promo`.** Either call the existing suggest/approve flow, or stop
   emitting a "full" variant that is identical to the silent one — the current output
   promises audio and delivers silence.
3. **Faces in the saliency map.** Fixes the backdrop quality complaint and makes
   `ClipQuality.face_area` non-zero. Costs a dependency: OpenCV Haar is cheap, a small
   ONNX detector is better.
4. **Check what is actually trending** before designing the new structures. This session
   could not — knowledge runs to May 2026 and trends move weekly. Worth searching for
   current Reels pacing conventions, hook patterns, and Raksha Bandhan audio.
5. **Tune thresholds against a real camera roll.** Still the highest-value item for the
   camera-roll path, and still not done.
6. **Nail-art add-ons are not read** — a grid where the price pill sits under a *wrapped*
   label. `_label_above` handles grids, `join_wrapped` handles wrapped labels, but not
   both at once.
7. Narration/TTS (`resolve_narration` is a stub), verifying the Stable Diffusion adapter,
   trending-audio discovery.

---

## How to check it still works

```bash
uv sync --extra dev --extra ocr
uv run pytest -q          # expect 545 passing
uv run ruff check .
uv run genvai doctor      # what this machine can do

ollama serve              # for anything using the LLM
```

An end-to-end smoke test, which exercises most of the system:

```bash
GENVAI_PROJECTS_DIR=/tmp/smoke uv run genvai promo demo \
  "beauty parlor/rakshabandhan offer/gracy nail art.png" \
  --occasion "Raksha Bandhan offer" --business "Nails by Gracy" --yes
```

Expect a confirmation table with four services, then a 7-beat 17.8s reel.

---

## Working agreements from this session

- Commit step by step, not in one lump. Push when asked.
- Keep docs short — they were cut from 1272 lines to ~520 once for being unreadable.
- Client assets never go in the public repo.
- Verify claims before making them. Several times "I updated X" turned out to be a string
  replacement that silently did not match — check, do not assume.
- Prefer measuring to guessing. Most of the good decisions here came from running the
  thing and looking at the output, and most of the bad ones came from reasoning about
  what should work.
