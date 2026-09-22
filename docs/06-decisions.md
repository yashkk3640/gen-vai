# 06 - Decision log

Why the project looks the way it does. Each entry records what was rejected, because
that is usually the part that gets forgotten and re-litigated.

- [Motion from stills, not generative video](#motion-from-stills-not-generative-video)
- [Timeline JSON as the single source of truth](#timeline-json-as-the-single-source-of-truth)
- [Typed edit operations, not free-form timeline rewrites](#typed-edit-operations-not-free-form-timeline-rewrites)
- [Music consent: suggested, downloaded only on confirmation](#music-consent-suggested-downloaded-only-on-confirmation)
- [Ports and adapters, protocols in one file](#ports-and-adapters-protocols-in-one-file)
- [uv, with heavy stacks as optional extras](#uv-with-heavy-stacks-as-optional-extras)
- [Bundled ffmpeg via imageio-ffmpeg](#bundled-ffmpeg-via-imageio-ffmpeg)
- [Phase-ordered model loading](#phase-ordered-model-loading)
- [Filesystem as the store](#filesystem-as-the-store)
- [Frozen pydantic models](#frozen-pydantic-models)

---

### Motion from stills, not generative video

**Decision.** Video is produced by animating stills (Ken Burns, parallax, transitions)
via ffmpeg, not by a video diffusion model.

**Why.** The target machine has 4 GB VRAM. SVD, AnimateDiff and similar need 12-24 GB.
Still generation at 512-768px fits comfortably, and camera motion on well-chosen
stills reads as deliberate when the move is picked per scene.

**Rejected.** Generative video locally (does not fit); cloud video APIs (breaks the
offline-first requirement).

**Revisit when** hardware grows - the `ImageProvider` port extends to a `ClipProvider`
without changing the pipeline.

---

### Timeline JSON as the single source of truth

**Decision.** An immutable, versioned `Timeline` document holds all editorial state.
Planning writes it, edits patch it, rendering reads it.

**Why.** It makes conversational editing tractable. "Change scene 2" is a function on
data, not a replay of a generation process. It is inspectable, diffable, restorable,
and testable without any model installed.

**Rejected.** Re-prompting the LLM to regenerate the whole video per request -
non-deterministic, expensive, and it silently changes parts the user was happy with.

---

### Typed edit operations, not free-form timeline rewrites

**Decision.** On an edit the LLM returns a validated list of ops from a closed
vocabulary.

**Why.** A model returning a whole new timeline will drift on fields nobody asked it
to touch. Ops are small, individually validatable, atomically applicable, reviewable
before execution, and trivially reversible. A malformed op is rejected with a precise
reason instead of quietly corrupting a project.

**Rejected.** RFC-6902 JSON Patch - generic, but LLMs handle JSON pointer paths badly
and a wrong path silently writes to the wrong place.

---

### Music consent: suggested, downloaded only on confirmation

**Decision.** The LLM describes a track; the provider returns candidates; nothing is
fetched until the user approves a specific one. Licence and source URL are recorded.

**Why.** Directly requested. It is also the right default: downloading is an outward-facing
network action with licensing consequences, so it should never be implicit. Encoding it
as a state machine on the timeline makes "did the user agree to this?" answerable by
looking at the data.

**Rejected.** Auto-fetching a best match (surprising, and a licence risk); bundling a
large music library in git (repo bloat).

---

### Ports and adapters, protocols in one file

**Decision.** Every external dependency sits behind a `typing.Protocol` in
`genvai/ports.py`; adapters are the only impure code.

**Why.** The pipeline is testable with zero models installed. Backends swap without
touching core logic. Keeping all protocols in one file means the whole system boundary
is readable in one sitting - a package of seven two-line files would be more structure
for less clarity.

**Rejected.** Direct calls into ffmpeg/torch/Ollama from pipeline code - untestable and
welds the design to today's tools.

---

### uv, with heavy stacks as optional extras

**Decision.** `uv` manages the environment. Base install is small; torch, TTS and ASR
are extras.

**Why.** Portability was an explicit requirement. `uv sync` reproduces the environment
exactly from `uv.lock`, and installs Python itself if absent. Putting torch in the base
would force a 3 GB download on machines that will only ever use procedural visuals.

**Rejected.** conda (heavier, slower, awkward to commit); bare pip + requirements.txt
(no real lockfile); poetry (`uv` is faster and handles the Python install too).

---

### Bundled ffmpeg via imageio-ffmpeg

**Decision.** ffmpeg arrives as a pip dependency. A system ffmpeg on PATH wins if present.

**Why.** It is the difference between "clone and run" and "clone, then install ffmpeg
and put it on PATH". The bundled build covers H.264/AAC, which is all the render path
needs.

**Trade-off.** The bundled build has a narrower codec set than a full system install.
The PATH-first lookup means anyone needing more just installs ffmpeg normally.

---

### Phase-ordered model loading

**Decision.** All LLM work for a run completes before any diffusion model loads.
The two never share the GPU.

**Why.** 4 GB cannot hold both. This is a correctness constraint, not tuning - violating
it means an OOM crash mid-render. It also forces the planner to emit a complete
storyboard in one pass, which is cheaper and more deterministic than interleaving.

**Consequence.** The planner cannot look at a generated image and reconsider. If
image-aware replanning is ever needed, it runs as a separate later phase with the
diffusion model unloaded.

---

### Filesystem as the store

**Decision.** A project is a directory. Content-addressed assets, one JSON file per
timeline version.

**Why.** Inspectable with a file manager, copyable between machines, diffable in git,
no service to run. Matches the "objects are obviously alive or dead, no hidden state"
principle - there is no cache to invalidate, because the fingerprint *is* the filename.

**Rejected.** SQLite or Postgres - real overhead for something with no concurrent
writers and no query needs.

---

### Frozen pydantic models

**Decision.** Domain models are frozen pydantic v2 models.

**Why.** Immutability by construction, validation at the boundary, and JSON Schema
generation for free - which is exactly what constrained LLM decoding needs. The house
style prefers immutable value objects over mutable structures, and this is the Python
equivalent.

**Rejected.** Dataclasses (no validation, no schema export); dicts (no contract at all).
