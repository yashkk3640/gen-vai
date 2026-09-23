# TODO

**All eight milestones are done.** What follows is refinement, not missing work.

## Done

- [x] **M0** Docs, portable env, timeline + media schemas, port boundary
- [x] **M1** Render path — trims, crops, Ken Burns, transitions, audio variants,
      fingerprint cache, `genvai render`
- [x] **M2** Ingest and analysis — probe, quality, spans, dedup, `genvai add` / `genvai media`
- [x] **M3** Selection — shortlist, LLM ordering and captions, `genvai reel`
- [x] **M4** Edit loop — free text → typed ops, atomic apply, `genvai edit` / `genvai restore`
- [x] **M5** Music — suggest, approve, beat detection, beat-aligned cuts, `genvai music`
- [x] **M6** Auto-reframe — saliency crop, moving when the subject moves
- [x] **M7** Idea mode — text → storyboard → generated backdrops, `genvai create`

## Known limitations

Things that work as designed but that you should know before trusting the output.

- **Captions are invented, not observed.** The model never sees the pictures - only
  measurements - so it writes plausible copy from your `--intent`. Ask for "a weekend by
  the sea" and you get "seagulls soar" whether or not there is a seagull. Grounding this
  needs a `ContentTagger`; the port exists and is unimplemented.
- **A small model sometimes adds a command you did not ask for.** Asked to drop a shot,
  a 3B model also emitted "drop the music". The diff shown before applying is the
  safeguard, so **avoid `genvai edit --yes`** unless you are scripting something you
  have already checked.
- **A 9B model is unusable on 4 GB.** It spills to CPU and times out. `llama3.2` (3B)
  answers in about twelve seconds; measurements are in [docs/setup.md](docs/setup.md).
- **No music ships with the project.** Point `GENVAI_MUSIC__LIBRARY_DIR` at your own.
- **Every threshold is tuned against synthetic footage.** Span weights, duplicate
  distance, drift, saliency floor. A dry run on a synthetic roll already moved the
  duplicate distance from 40 to 12 and stopped reels being padded with blur; real photos
  will move them again.

## Known gaps in what is built

- [ ] `blur_pad` fit is approximated as `cover`
- [ ] Image overlays (logo/watermark) parse but do not draw
- [ ] Captions are not burned from narration — needs real audio timings, which needs TTS
- [ ] `resolve_narration` is a stub; the TTS port has no implementation
- [ ] `face_area` is always 0 — reframing uses detail saliency, not faces
- [ ] The Stable Diffusion provider is written but has never been run

## Worth doing next

- [ ] Tune the thresholds against a **real** camera roll. Highest value of anything here
- [ ] Narration: implement the Piper TTS adapter and `resolve_narration`
- [ ] Verify the Stable Diffusion adapter on real hardware
- [ ] Faces weighted into the saliency map, which would also make `face_area` non-zero
- [ ] Trending-audio discovery — see [docs/backlog.md](docs/backlog.md)
