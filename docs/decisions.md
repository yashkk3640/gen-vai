# Decisions

Why the project looks the way it does. The rejected option is usually the part that gets
forgotten and re-argued, so it is recorded too.

### Camera roll over text-to-video

Generating video from a prompt is crowded, and a 4 GB GPU loses there. Real footage
sidesteps image quality entirely and leaves the LLM doing what it is good at.
**Rejected:** idea mode as the headline — parked, not deleted.

### Motion from stills, not generative video

Video diffusion needs 12–24 GB. Ken Burns and transitions on stills produce real motion
and read as deliberate when chosen per scene.
**Revisit** when hardware grows; the `ImageProvider` port extends without pipeline change.

### Timeline JSON as the single source of truth

Makes conversational editing tractable — "change scene 2" is a function on data, not a
replay of a generation. Inspectable, diffable, restorable, testable with no model.
**Rejected:** re-prompting to regenerate the whole video; non-deterministic and it
silently changes parts the user liked.

### Typed edit operations, not timeline rewrites

A model handed a whole timeline drifts on fields nobody asked it to touch. Ops are small,
individually validatable, atomic, reviewable, reversible.
**Rejected:** RFC-6902 JSON Patch — LLMs handle pointer paths badly and a wrong path
writes silently to the wrong place.

### Measurement separate from judgement

Sharpness, exposure, shake and duplicates are objective and cheap on the CPU.
Shortlisting is pure and deterministic. Only then does the LLM see a compact description.
Keeps context small, edits reproducible, and the whole thing runnable on modest hardware.

### Music consent

The LLM describes a track, the provider returns candidates, nothing is fetched until the
user approves one. Licence and source URL recorded. `MusicProvider.fetch` takes
`confirmed` as an argument rather than reading config, so no call site can reach the
network invisibly.
**Rejected:** auto-fetching a best match — surprising, and a licensing risk.

### Ports and adapters, protocols in one file

Pipeline is testable with zero models installed; backends swap without touching core
logic. One file means the whole boundary reads in one sitting.
**Rejected:** a package of seven two-line files — more structure, less clarity.

### Phase-ordered model loading

4 GB cannot hold a 7B LLM and a diffusion model at once. All LLM work completes before
any image generates. A correctness constraint, not tuning — violating it is an OOM crash
mid-render. Consequence: the planner emits a complete plan in one pass.

### uv, heavy stacks as extras

`uv sync` reproduces the environment exactly and installs Python itself. torch in the
base would force 3 GB on machines that will only ever use procedural visuals.
**Rejected:** conda (heavy), pip + requirements.txt (no real lock), poetry (slower).

### Bundled ffmpeg

The difference between "clone and run" and "clone, then install ffmpeg and fix PATH". A
system binary wins when present, for the wider codec set.

### Filesystem as the store

Inspectable, copyable, diffable, no service to run. The fingerprint *is* the filename, so
there is no cache to invalidate.
**Rejected:** SQLite or Postgres — real overhead, no concurrent writers, no queries.

### Frozen pydantic models

Immutability by construction, validation at the boundary, and JSON Schema export for
free — which is exactly what constrained LLM decoding needs.
**Rejected:** dataclasses (no validation), dicts (no contract).

### A storyboard before the render (M9)

A promo reel is designed as a short film first: `story/vN.json` holds a logline, an arc
and one entry per shot - picture, framing, angle, camera move, captions and how each
arrives, cut, effects, length in beats. It is reviewed as a storybook page, then drawn.
For promos it is the source of truth; the timeline is not involved.
**Rejected:** going straight from the brief to a timeline - nobody ever decided what the
reel was about, and every reel came out the same shape.

### Templates write the structure, the model writes three lines

Arcs (glow-up, countdown, treat, reveal) carry the craft - where the hook lands, when the
price appears, which cuts sit where. The 3B model writes only a hook, a promise and a
call to action, and each is filtered: no digits, no repeating the occasion, no claims
the poster does not make. It wrote "BOOK NOW BEFORE SOLD OUT" for a salon with no limit.
**Rejected:** the model drafting whole storyboards - small models drift on multi-field
schemas, measured in M3.

### Frames drawn in Python, not filtergraphs

A storyboard shot is layers - a card over a blurred copy of itself, particles in front,
type popping on the beat. `adapters/compositor.py` prepares each shot once and draws it
frame by frame with Pillow and numpy, piping raw frames to x264. About 0.35 s a frame at
1080x1920; shots are cached by content, so editing one shot redraws one shot.
**Rejected:** extending the ffmpeg filtergraph renderer - correct for footage, unreadable
for motion graphics.

### Poster imagery only, depth from layers

The client chose not to film and not to use generated images. Poster photos are small
(a tile is ~135 px), so they are never shown full screen: a card at no more than 5x,
over a blurred, brand-tinted copy of itself, moving against it for parallax, with grain.

### Every caption is measured, not trusted

The compositor draws each shot without its text, samples what sits behind every caption
as it arrives and as the shot ends, and computes the WCAG contrast ratio. Under 4.5:1 it
tries the brand's deep colour as ink, then a soft plate. Measured before this existed:
19 of 24 captions on the two Navratri reels were under 4.5:1, the worst at 1.0:1.
Captions also must stay on screen, fully arrived, for 0.45 s plus 0.24 s a word; shots
too short for their text are lengthened in half beats. Text never arrives during a cut
into a shot and leaves before a whip or fade, so no transition smears it.
**Rejected:** a fixed dark scrim on every shot - it flattens the pictures that did not
need it, and still fails over a busy one.
