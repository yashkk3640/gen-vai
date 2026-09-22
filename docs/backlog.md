# Backlog

Understood, deliberately not built, with the reason.

> Platform mechanics move faster than this file. Reach and distribution claims were
> accurate as understood in mid-2026; re-check before betting a milestone on one.

### Image fidelity ceiling — *blocked by hardware*

SD-Turbo at 4 GB reads as AI-generated. No pipeline work fixes it. Options: a hosted
adapter behind the existing `ImageProvider` port (costs offline-first), a bigger GPU
(no code change), or lean into a designed typographic look instead of competing on
photorealism. Only matters for idea mode.

### Character and scene consistency — *hard*

Plain diffusion will not hold the same person across scenes. `style_suffix` plus a pinned
seed narrows drift but does not solve it. Real fixes — IP-Adapter, a reference image, a
small LoRA — need more VRAM than 4 GB leaves.

### Motion that does not read as a slideshow — *partly done*

Scene-level motion variety and short beats are in the schema. Still missing: depth-aware
parallax (needs a depth estimator), and cutting on the beat. **Beat-synced cutting is the
best value here** — it needs onset detection, not a model.

### Trending audio — *partly done*

The `narration_only` export exists so a sound can be attached in-app. Still open:
discovering *what* is trending needs a platform data source, and that is the hard part.

### Retention feedback — *not started*

Nothing knows whether a video performed. Everything above is a prior, never a
measurement. A minimal version — paste back view figures, let the planner weight its
choices — would turn assumptions into something that learns.

### Auto-reframe and face tracking

Cropping landscape to 9:16 decapitates people. Needs face/saliency detection and a
smoothed crop path. Scheduled as M6.

### Idea mode — *deprioritised on purpose*

Generating a video from a description was the original headline. It is now last, because
the market is crowded, the incumbents have better models and voices, and 4 GB is where
this loses hardest.

**Survives:** generated title cards and gap-fillers. `GeneratedVisual` is in the same
union as `ClipVisual`, so a generated shot drops into a real timeline at any point.
**Parked:** whole videos made of generated stills.
