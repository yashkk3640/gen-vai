# 02 - Architecture

## Shape

Ports and adapters around a pure core. The core knows nothing about ffmpeg,
Ollama, or torch - only about protocols.

```
                    CLI  (genvai.cli)
                          |
        +-----------------+------------------+
        |            pipeline layer          |   orchestration, impure
        |  plan -> resolve -> render -> edit |
        +-----------------+------------------+
                          |
        +-----------------+------------------+
        |              PURE CORE             |
        |   timeline schema  +  ops  +  patch|   no I/O, fully testable
        +-----------------+------------------+
                          |
        +-----------------+------------------+
        |               ports                |   Protocols only
        | LLM Image Speech Music Render Store|
        +-----------------+------------------+
                          |
        +-----------------+------------------+
        |              adapters              |   the only impure code
        | ollama diffusers piper ffmpeg fs   |
        +------------------------------------+
```

Dependencies point **inward**. The core never imports an adapter.

## Data flow - creation

```
intent (text) + optional assets
        |
        v
  [1] PLAN            LLM  ->  Storyboard  ->  Timeline v1
        |                      (validated, rejected+retried if malformed)
        v
  [2] RESOLVE         for each unresolved slot:
        |               visual    -> ImageProvider   -> asset
        |               narration -> SpeechProvider  -> asset
        |               music     -> MusicProvider   -> CANDIDATES ONLY
        v
  [3] CONFIRM         music candidates shown to user
        |               approved -> downloaded -> asset
        |               declined -> render silent
        v
  [4] RENDER          Timeline -> ffmpeg filtergraph -> MP4
        |               per-scene segments, cached by fingerprint
        v
     output.mp4  +  timeline_v1.json persisted
```

## Data flow - conversational edit

This is the flow that defines the product.

```
"make scene 2 slower and change the music"
        |
        v
  LLM  <- current Timeline (summarised) + op vocabulary schema
        |
        v
  EditOp[]   e.g. [SetSceneDuration(s2, 6.0), SetMusicQuery(...)]
        |
        v
  VALIDATE against current timeline   -- unknown scene id? bad range? reject with reason
        |
        v
  APPLY    Timeline v_n -> Timeline v_n+1     (pure, total)
        |
        v
  DIFF     show the user what changed, ask to proceed
        |
        v
  RE-RENDER  fingerprint every scene; re-encode only what changed; concat
```

The LLM never returns a whole timeline on an edit. It returns **operations**.
That keeps edits cheap, reviewable, reversible, and impossible to corrupt by
hallucinating an unrelated field.

## Incremental rendering

Each scene gets a **content fingerprint**: a hash over everything that affects its pixels
and samples - resolved asset hashes, duration, motion, overlays, transitions, canvas, seed.

```
scene.fingerprint  ==  cached segment on disk   ->  reuse
                   !=                            ->  re-encode that scene only
```

Segments are concatenated with ffmpeg's concat demuxer. Changing scene 2 of a
12-scene video re-encodes one segment, not twelve. This is what makes
conversational editing feel interactive rather than like a fresh render each time.

The music/narration bed is mixed in a final pass, so an audio-only change skips
video re-encoding entirely.

## Ports

All protocols live in one file (`genvai/ports.py`) so the entire system boundary
is readable in a single sitting.

| Port | Responsibility | Default adapter | Fallback |
| --- | --- | --- | --- |
| `LLMPort` | text completion + schema-constrained structured output | Ollama HTTP | none (required) |
| `ImageProvider` | prompt -> still image | `diffusers` local SD | procedural card |
| `SpeechProvider` | text -> narration audio | Piper (CPU) | silence |
| `MusicProvider` | describe -> candidates; fetch on approval | local library | no music |
| `TranscriptProvider` | audio -> timed transcript | faster-whisper | not required until footage mode |
| `RendererPort` | timeline -> video file | ffmpeg | none (required) |
| `ProjectStore` | load/save project + versions | filesystem | none (required) |

Every port is a `typing.Protocol`. Tests substitute fakes; no model needs to be
installed to exercise the pipeline.

## VRAM discipline

4 GB does not fit a 7B LLM and Stable Diffusion at once. The pipeline is therefore
**phase-ordered**, and this is a correctness constraint, not an optimisation:

```
PLAN phase      LLM loaded         diffusion NOT loaded
                        |
                  (LLM released)
                        |
RESOLVE phase   diffusion loaded   LLM NOT loaded
```

All LLM decisions for a run are made **before** any image is generated. The planner
therefore emits a complete storyboard in one pass rather than interleaving
generation and reasoning. Ollama's `keep_alive` is set to release the model at the
phase boundary.

## Project layout on disk

A project is a plain directory. Open it in a file manager and everything is visible -
no database, no hidden state.

```
projects/<project-id>/
  project.json            # id, intent, created_at, current version pointer
  timelines/
    v1.json
    v2.json               # every version kept; restore = point at an older file
  assets/
    images/<sha256>.png
    audio/<sha256>.wav
    music/<sha256>.mp3    # + .license.json alongside
  cache/
    segments/<fingerprint>.mp4
  renders/
    v2-final.mp4
    v2-preview.mp4
  log/
    llm.jsonl             # every prompt and response
    events.jsonl
```

Assets are content-addressed by hash, so the same image is never stored twice and
a timeline referencing it is verifiable.

## Module map

```
src/genvai/
  cli.py            Typer commands - thin, no logic
  config.py         Settings from env + .env (pydantic-settings)
  errors.py         Typed error hierarchy
  timeline.py       PURE  frozen models: Timeline, Scene, Visual, Motion, Audio...
  ops.py            PURE  EditOp union + apply() : Timeline -> Timeline
  fingerprint.py    PURE  content hashing for incremental render
  ports.py          Protocols - the whole system boundary in one file
  pipeline/
    plan.py         intent -> Timeline
    resolve.py      fill unresolved asset slots
    render.py       Timeline -> MP4
    edit.py         request -> EditOp[] -> new Timeline
  adapters/
    ollama.py       LLMPort
    diffusers.py    ImageProvider (optional extra)
    procedural.py   ImageProvider fallback
    piper.py        SpeechProvider (optional extra)
    music_local.py  MusicProvider
    ffmpeg.py       RendererPort
    fs_store.py     ProjectStore
```
