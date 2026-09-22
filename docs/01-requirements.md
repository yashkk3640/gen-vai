# 01 - Requirements

## Functional

### F1 - Project lifecycle
- F1.1 Create a project from a free-text intent, with zero or more input assets.
- F1.2 A project is a self-contained directory: timeline versions, assets, renders, log.
- F1.3 A project is reopenable across machines and sessions; no state lives in memory.
- F1.4 Every timeline version is retained. Nothing is overwritten, so any version can be restored.

### F2 - Planning (LLM)
- F2.1 Turn an intent into a **storyboard**: scenes, narration text, on-screen text, duration, visual prompt.
- F2.2 Honour explicit constraints from the intent (total duration, aspect ratio, tone, language).
- F2.3 Emit a schema-valid `Timeline`. Invalid output is rejected and retried with the validation error fed back; it is never silently patched.
- F2.4 When the user supplies images, plan around them rather than generating new ones.
- F2.5 Choose a motion style per scene, not one global effect.

### F3 - Visual generation
- F3.1 Generate one still per scene from the LLM-written prompt.
- F3.2 Deterministic: a recorded seed reproduces the image exactly.
- F3.3 Fall back to a procedural card (typography/gradient) when no image backend is available.
- F3.4 Never silently regenerate an image the user approved.

### F4 - Audio
- F4.1 Synthesise narration per scene from the script.
- F4.2 **Suggest** music: the LLM describes the desired track (mood, genre, tempo, energy) and the provider returns candidates.
- F4.3 **Never download without explicit confirmation.** Candidates are presented; the user approves one; only then is it fetched.
- F4.4 Record licence and source URL for every downloaded track.
- F4.5 Duck music under narration automatically.
- F4.6 Reuse an already-downloaded track across projects via a local cache.

### F5 - Rendering
- F5.1 Render the timeline to MP4 (H.264 + AAC) via ffmpeg.
- F5.2 Support 9:16, 16:9, 1:1 canvases.
- F5.3 Burn captions with a configurable style.
- F5.4 **Incremental:** re-render only scenes whose content fingerprint changed.
- F5.5 Emit a proxy/preview render (low res, fast) distinct from the final render.

### F6 - Conversational editing
- F6.1 Accept a free-text change request against an existing project.
- F6.2 The LLM emits a **typed edit-op list**, not a replacement timeline.
- F6.3 Ops are validated against the current timeline before application; an invalid op is rejected with a clear reason.
- F6.4 Show the user what will change before re-rendering.
- F6.5 Any version can be restored; an unwanted edit is always reversible.

### F7 - Interface
- F7.1 CLI is the primary surface, over a pure core library.
- F7.2 Core contains no CLI or I/O concerns, so an API or UI can be added without refactoring.

## Non-functional

### N1 - Portability *(explicit user requirement)*
- N1.1 `git clone` + `uv sync` reproduces the environment on any machine.
- N1.2 ffmpeg ships with the environment; no manual system install.
- N1.3 Heavy optional stacks (torch, TTS, ASR) are extras, not base dependencies.
- N1.4 No absolute paths in committed files or in project state.

### N2 - Offline first
- N2.1 Everything runs locally. No cloud service is required.
- N2.2 The only network calls are music fetches, which are confirmation-gated.

### N3 - Resource ceiling
- N3.1 Must run within 4 GB VRAM. Never hold the LLM and the diffusion model on the GPU simultaneously.
- N3.2 Degrade gracefully to CPU where feasible; report the cost rather than failing.

### N4 - Determinism
- N4.1 Identical timeline + seeds produce a byte-comparable render, modulo encoder nondeterminism.
- N4.2 Every LLM call, prompt, and response is logged to the project for auditability.

### N5 - Quality
- N5.1 Timeline schema and pure transforms have full test coverage; they are the correctness core.
- N5.2 Providers are mockable via protocols, so the pipeline is testable with no models installed.
- N5.3 `mypy --strict` clean.

## Explicit non-goals

- No manual timeline editing UI.
- No JPA-style ORM or database; the filesystem is the store.
- No cloud rendering, no SaaS accounts.
- No generative video diffusion until hardware supports it.
- No general-purpose plugin/scripting engine.
