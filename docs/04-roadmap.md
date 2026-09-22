# 04 - Roadmap

## Status - 2026-09-22

**Milestone 0 complete: documentation, portable environment, typed scaffold.**

The schema has since had a short-form pass: pacing defaults, scene roles, kinetic
captions, export variants and safe areas were folded into the contract before any
renderer existed, because adding them after M1 would have meant reworking the
filtergraph. What could not be folded in is parked in [07-backlog.md](07-backlog.md).

Nothing renders yet. Every adapter is a stub that raises `NotImplementedError`.
This is deliberate - the schema and the port boundary were agreed before any
implementation, so implementation does not thrash.

| | Milestone | State |
| --- | --- | --- |
| M0 | Docs, env, ports, timeline and media schemas | **done** |
| M1 | Render path: timeline -> MP4, trims and crops included | next |
| M2 | Ingest and analysis: camera roll -> media.json | |
| M3 | Selection and assembly: library + intent -> timeline | |
| M4 | Conversational edit loop + incremental re-render | |
| M5 | Music, beat detection, beat-aligned cuts | |
| M6 | Auto-reframe: keep the subject in frame when cropping to 9:16 | |
| M7 | Idea mode: generated visuals and narration | parked |

## M1 - Render path first

A hand-written timeline renders to MP4. Deliberately before any analysis or LLM work,
because everything downstream produces timelines and nothing can be checked until one
can be played.

- ffmpeg adapter: trim a clip span, crop/fit to canvas, speed change, concat
- photos: Ken Burns motion; clips: straight playback within the trim
- ASS subtitle generation for word-level kinetic captions
- audio variants: one video pass, separate muxes for full / narration_only / silent
- safe-area-aware overlay placement
- filesystem store; content-addressed assets
- fingerprint-based segment cache

*Done when:* a committed sample timeline mixing photos and trimmed clips renders to a
correct MP4 on a clean clone, and changing one scene re-encodes exactly one segment.

## M2 - Ingest and analysis

The slow phase, cached forever by content hash.

- probe: dimensions, duration, fps, EXIF capture time and rotation
- measure: sharpness, exposure, motion, shake, face area
- spans: find the good seconds inside each clip
- dedup: perceptual hashing groups near-identical takes
- `genvai add <project> <files...>`

All classical CV on the CPU. No model, no GPU, no optional extra.

*Risk:* span detection is the part that decides whether the output is good. A first cut
can be crude - prefer stable, sharp, well-exposed stretches away from the clip's start
and end, since phone clips are shakiest when a thumb is on the button.

## M3 - Selection and assembly

- `shortlist`: pure scored filter, no model
- `assemble`: LLM orders the shortlist, sets trims, roles and on-screen text
- chronological fallback when Ollama is unavailable, so the tool still works
- `genvai reel <project> --duration 30`

*Done when:* a real camera roll produces a watchable reel without hand-editing.

## M4 - Conversational editing

- edit-op extraction from free text, including the clip ops
- atomic validate-then-apply, human-readable diff, version restore
- `genvai edit <project> "drop the third clip and slow the sunset one down"`

## M5 - Music and beat

- music query -> candidates -> confirm -> download -> licence recorded
- beat detection on the chosen track
- snap cuts to the grid

## M6 - Auto-reframe

Phone clips are often landscape; reels are vertical. Centre-cropping decapitates people.
Face/saliency-aware crop paths, smoothed over time so the frame does not jitter.

## M7 - Idea mode

Parked, not cancelled. Generated stills and narration for title cards and gaps where no
footage exists. See the backlog for why it is not the headline.

## Known open questions

1. **Should footage mode be built before idea mode?** For competitive short-form output on 4 GB,
   editing real footage sidesteps the image-fidelity ceiling entirely and leaves the
   LLM doing what it is best at. That argues for moving M7 to M2. It is a product
   call, not a technical one, so it is recorded rather than assumed
   ([07-backlog.md](07-backlog.md) B5).
2. **Small-model JSON reliability** - decides whether M2 needs the two-step fallback.
3. **Beat snapping versus span integrity** - forcing a cut onto a beat can clip the
   good moment it was trimmed to. Which wins is a judgement call that needs footage.
4. **Music source** - which CC library to integrate first, and whether to ship a small
   bundled set so M5 works fully offline.
5. **Preview loop** - is a proxy render fast enough for iteration, or is a frame-accurate still preview needed for the edit loop to feel conversational?
