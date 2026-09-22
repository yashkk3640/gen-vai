# 07 - Backlog

Work that is understood but deliberately not built, with the reason it is parked.
Everything here came out of assessing the design against short-form platform
requirements (Reels, Shorts, TikTok).

Items that were cheap enough to fold straight into the schema are **not** here - they
are already in `timeline.py` and listed at the bottom for reference.

> Platform mechanics move faster than this document. The reach and distribution claims
> below were accurate as understood in mid-2026; re-check them before betting a
> milestone on one.

---

## B1 - Image fidelity ceiling

**Blocked by hardware.**

SD-Turbo at 4 GB produces imagery that reads as AI-generated. That is the single
largest limit on output quality, and no amount of pipeline work fixes it.

Options, in order of appeal:

1. A hosted image adapter behind the existing `ImageProvider` port - FLUX-class quality,
   no local VRAM. Costs the offline-first property, which is why it is not the default.
2. A larger local GPU. Changes nothing in the code; `diffusers_image.py` already takes
   resolution and step count from settings.
3. Stay procedural and lean into a designed, typographic look rather than competing on
   photorealism. Cheapest, and for text-forward niches arguably the *better* result -
   a deliberate graphic style ages better than mediocre generated photos.

The port boundary means this is an adapter swap whenever it becomes worth doing.

---

## B2 - Character and scene consistency

**Hard, not blocked.**

Plain diffusion will not hold the same person or place across scenes. `style_suffix`
plus a pinned seed narrows the drift but does not solve it.

Real fixes: IP-Adapter, a reference image per character, or a small LoRA. All need the
`image` extra and more VRAM headroom than 4 GB leaves. Until then, plan around it -
narration-driven videos over abstract or atmospheric imagery work; anything with a
recurring character does not.

---

## B3 - Motion that does not read as a slideshow

**Partially addressed.**

Uniform Ken Burns over stills is the recognisable "AI slideshow" look. Scene-level
motion variety and shorter beats are already in the schema, which helps. What is still
missing:

- depth-aware parallax (`ParallaxMotion` is defined but needs a depth estimator)
- motion cut on the beat of the audio bed
- occasional real motion - a generated or stock clip among the stills breaks the pattern

Beat-synced cutting is the best value here: it needs onset detection, not a model.

---

## B4 - Trending audio workflow

**Partially addressed.**

The `narration_only` export variant exists so a trending sound can be attached in-app,
which is where the reach actually comes from. Still open:

- surfacing *what* is trending, which needs a platform data source and is the hard part
- pacing cuts to a track the tool cannot see
- a "sound-first" mode: pick the audio, then cut the video to it

Until a trends source exists, the honest workflow is: render narration-only, attach the
sound manually, accept that cut timing will not match the track.

---

## B5 - Mode C should probably outrank Mode A

**A prioritisation question, not a technical one.**

For competitive short-form output on this hardware, editing existing footage beats
generating stills. The source material is already real, so B1 and B2 stop mattering
entirely, and the LLM is left doing what it is genuinely good at: choosing moments,
pacing, and writing captions.

Mode C is currently M7, last. If short-form output is the actual goal rather than
text-to-video as such, it belongs at M2, right after the render path.

Deliberately not reordered yet - that is a product call, and the roadmap notes it as
open rather than assuming an answer.

---

## B6 - Retention feedback loop

**Not started.**

Nothing in the system knows whether a video performed. Everything above is a prior
about what works, never a measurement.

A minimal version: record what was published, let the user paste back retention or
view figures, and let the planner weight its choices accordingly. Small, and it would
turn a set of assumptions into something that actually learns.

---

## B7 - Auto-reframe and face tracking

**Needed for Mode C.** Cropping a 16:9 talking head to 9:16 without cutting the
speaker's head off needs face detection and a smoothed crop path. Blocked on Mode C
being scheduled.

---

## Already folded into the schema

These were cheap, so they were built into the contract before implementation rather
than parked:

| | Where |
| --- | --- |
| Default beat length 2.2s, not 4.0s | `Scene.duration` |
| `hook` / `body` / `payoff` / `cta` scene roles | `Scene.role` |
| Word-level kinetic captions | `Captions.mode`, `TextStyle.highlight_*` |
| Narration-only and silent export cuts | `Export.audio_variants` |
| Seamless-loop flag | `Export.seamless_loop` |
| Platform UI safe areas | `Canvas.safe_area` |
| Shared style phrase across prompts | `Timeline.style_suffix` |
| Heavier caption weight and stroke for small screens | `TextStyle` defaults |

They cost nothing now because no renderer had been written yet. Adding them after M1
would have meant reworking the ffmpeg filtergraph, so it was worth doing first.
