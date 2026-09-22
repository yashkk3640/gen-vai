# TODO

**Now: M2 — ingest and analysis.**

## Done

- [x] **M0** Docs, portable env, timeline + media schemas, port boundary
- [x] **M1** Render path — trims, crops, Ken Burns, transitions, audio variants,
      fingerprint cache, `genvai render`

## M2 — ingest and analysis

Read a camera roll once, cache forever by content hash. All CPU, no model.

- [ ] Probe: dimensions, duration, fps, capture time, rotation
- [ ] Quality: sharpness, exposure, motion, shake
- [ ] Spans: find the good seconds inside each clip
- [ ] Dedup: group near-identical takes
- [ ] `genvai add` and `genvai media`

## Next

- [ ] **M3** Selection — shortlist (pure) + LLM ordering → timeline. `genvai reel`
- [ ] **M4** Edit loop — free text → typed ops → re-render what moved. `genvai edit`
- [ ] **M5** Music — suggest, confirm, download, beat-align cuts
- [ ] **M6** Auto-reframe — keep the subject in frame when cropping to 9:16

## Known gaps in what is built

- [ ] `blur_pad` fit is approximated as `cover`
- [ ] Image overlays (logo/watermark) parse but do not draw
- [ ] Captions not burned from narration — needs real audio timings (M5)

## Parked

**Idea mode** (generate a video from a text prompt) — the crowded end of the market, and
where a 4 GB GPU loses. See [docs/backlog.md](docs/backlog.md).
