# 04 - Roadmap

## Status - 2026-09-22

**Milestone 0 complete: documentation, portable environment, typed scaffold.**

Nothing renders yet. Every adapter is a stub that raises `NotImplementedError`.
This is deliberate - the schema and the port boundary were agreed before any
implementation, so implementation does not thrash.

| | Milestone | State |
| --- | --- | --- |
| M0 | Docs, env, ports, timeline schema | **done** |
| M1 | Render a timeline to MP4 (procedural visuals, no LLM) | next |
| M2 | LLM planner: intent -> timeline | |
| M3 | Conversational edit loop + incremental re-render | |
| M4 | Local diffusion visuals | |
| M5 | Narration TTS + music suggestion/confirm/download | |
| M6 | Mode B: user-supplied images | |
| M7 | Mode C: existing footage -> shorts | |

## M1 - Render path first

Deliberately before the LLM. A hand-written timeline JSON renders to MP4 using
procedural cards and Ken Burns motion.

- ffmpeg adapter: per-scene segment, `zoompan`, transitions, concat
- procedural image provider: gradient + typography
- filesystem store; content-addressed assets
- fingerprint-based segment cache
- `genvai render <project>`

*Done when:* a committed sample timeline renders to a correct MP4 on a clean clone,
and re-rendering after changing one scene re-encodes exactly one segment.

## M2 - The planner

- Ollama adapter with schema-constrained structured output
- intent -> storyboard -> `Timeline`, with validate-and-retry on malformed output
- prompt/response logging to `log/llm.jsonl`
- `genvai create "<intent>"`

*Risk:* small local models produce invalid JSON. Mitigation: constrained decoding via
Ollama's `format` parameter, a compact schema, and bounded retry that feeds the
validation error back. If a 7B model proves unreliable, the fallback is a two-step
prompt - prose storyboard first, then JSON conversion - which is far easier for
small models than one-shot structured output.

## M3 - Conversational editing

The milestone that makes it the product described in the vision.

- `EditOp` extraction from free text
- atomic validate-then-apply
- human-readable diff before re-render
- version restore
- `genvai edit <project> "<request>"` / `genvai restore <project> <version>`

*Done when:* "make scene 2 longer and drop the music" applies correctly, shows the
diff, re-encodes only the affected segment, and is reversible.

## M4 - Diffusion visuals

- `diffusers` adapter, SD-Turbo default (1-4 steps, fast, fits 4 GB)
- attention slicing + sequential CPU offload; VAE slicing for the 4 GB ceiling
- explicit LLM unload before load - the phase boundary from the architecture doc
- seed recorded per image for reproducibility

*Risk:* 4 GB is genuinely tight. If SD-Turbo at 768px OOMs, drop to 512px and
upscale with ffmpeg `lanczos`. Procedural fallback stays available throughout.

## M5 - Audio

- Piper TTS per scene; timing reconciled against planned scene duration
  (scenes stretch to fit narration rather than truncating it)
- music query -> candidates -> **confirm** -> download -> licence recorded
- sidechain ducking under narration

## M6 - User images

- ingest, hash, EXIF orientation, aspect-fit to canvas
- LLM orders them and assigns motion from actual image content
- requires a vision model, or captioning at ingest so the text LLM can reason about them

## M7 - Existing footage

- faster-whisper transcript with word timings
- LLM highlight selection -> cut list
- auto-reframe to 9:16, face-aware where possible
- one long video -> several shorts

## Known open questions

1. **Small-model JSON reliability** - decides whether M2 needs the two-step fallback.
2. **Scene/narration timing** - stretch the scene, speed the speech, or trim the text? Leaning stretch-the-scene; it is the least destructive.
3. **Music source** - which CC library to integrate first, and whether to ship a small bundled set so M5 works fully offline.
4. **Preview loop** - is a proxy render fast enough for iteration, or is a frame-accurate still preview needed for the edit loop to feel conversational?
