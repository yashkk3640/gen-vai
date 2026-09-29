# TODO

**All nine milestones are done.** What follows is refinement, not missing work.

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
- [ ] The Stable Diffusion provider is written but has never been run
- [ ] Promo backdrops keep a word when the ground around it is not flat - a caption
      touching a photo, a banner on a gradient. Everything on flat ground is erased
- [ ] Pairing a service to its price is imperfect on unusual layouts — hence the
      confirmation table

- [x] **M8** Promo mode — a reel from designed artwork, no script to write.
      `genvai promo`

## Done - from the client's feedback on the two reels

- [x] **Five promo structures** - classic, question, from, menu-first, countdown -
      picked by seed or `--style`. Each differs in order, pacing and transition
- [x] **Music in promo.** `--music FILE`, or pick from the library when one exists.
      No track, no "full" cut. The bed mix no longer loses 6 dB to `amix`
- [x] **Faces in the saliency map** (YuNet, bundled). Backdrops, reframing, `face_area`
- [x] **Printed words erased from backdrops** using the OCR boxes already read
- [x] **Trends checked** (Sept 2026): hook in the first 1-2 s, question hooks,
      countdowns. Trending *audio* could not be established - pick it in the app
- [x] **Every poster reads completely**: nail add-on grids, two-price columns with
      their headers, labels whose price sits by their second line. 14/14, 10/10,
      11/11, 10/10 on the four client posters

## M9 - storyboard layer (done 2026-09-29)

- [x] Cast: each picture on a poster as an object, named by the label beside it
- [x] Storyboard schema, four story arcs, model-written copy with a claims filter
- [x] Frame compositor: cards over blurred backdrops, parallax, petals, bokeh, light
      leaks, grain, animated type, whip / flash / zoom / fade cuts
- [x] Storybook page; `genvai story` and `genvai shoot`; shots cached by content

## Storyboard - known gaps

- [ ] 0.35 s a frame at full size - a 20 s reel takes about three minutes
- [ ] Offers with no matching picture fall back to any person, sometimes a repeat
- [ ] Shots cannot yet be edited in plain English; edit `story/vN.json` and re-shoot
- [ ] Combining two posters in one storyboard is untested

## Worth doing next

- [ ] Tune the thresholds against a **real** camera roll. Highest value of anything here
- [ ] Narration: implement the Piper TTS adapter and `resolve_narration`
- [ ] Verify the Stable Diffusion adapter on real hardware
- [ ] Trending-audio discovery — see [docs/backlog.md](docs/backlog.md)
