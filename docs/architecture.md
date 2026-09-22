# Architecture

Ports and adapters around a pure core. The core knows nothing about ffmpeg, Ollama or
torch — only about protocols. Dependencies point inward.

```
CLI → pipeline (orchestration) → PURE CORE (timeline, ops, fingerprint)
                               → ports (Protocols) → adapters (ffmpeg, ollama, fs)
```

## The flow

```
photos + clips
   ↓
INGEST      once per file ever, keyed by content hash
            probe · measure quality · find good spans · group duplicates  → media.json
   ↓
SHORTLIST   pure, deterministic, no model
   ↓
ASSEMBLE    LLM orders it, picks trims, roles and text                    → Timeline v1
   ↓
BEAT FIT    nudge cuts onto the music grid
   ↓
RENDER      trim · crop · concat · mix                                    → MP4
   ↓
EDIT        "drop the third clip" → typed ops → Timeline v2 → re-render what moved
```

## Incremental rendering

Every scene gets a fingerprint hashing everything that affects its pixels — assets,
duration, motion, overlays, canvas, styles. Match a cached segment, reuse it; otherwise
re-encode that scene alone. Changing scene 2 of twelve re-encodes one segment.

The audio mix is a separate pass, so an audio-only change skips video entirely.

## Ports

| Port | Job | Default | Fallback |
| --- | --- | --- | --- |
| `LLMPort` | structured output | Ollama | required |
| `MediaAnalyzer` | quality, spans, duplicates | ffmpeg + numpy, CPU | required |
| `ContentTagger` | what is in a shot | CLIP | quality + chronology |
| `BeatDetector` | music → beat grid | onset detection | cuts on scene bounds |
| `RendererPort` | timeline → video | ffmpeg | required |
| `ProjectStore` | load/save | filesystem | required |
| `ImageProvider` | prompt → still | diffusers | procedural card |
| `SpeechProvider` | text → audio | Piper | silence |
| `MusicProvider` | search, fetch on approval | local library | no music |

All protocols live in one file, `genvai/ports.py`, so the whole boundary reads in one
sitting. Tests substitute fakes; no model needs installing to exercise the pipeline.

## A project on disk

```
projects/<id>/
  project.json          id, intent, current version
  media.json            ingest results: quality, spans, dedup groups
  timelines/v1.json     every version kept — restore points at an older file
  assets/               images/ clips/ audio/, named by content hash
  cache/segments/       rendered scenes, named by fingerprint
  renders/              the output
  log/llm.jsonl         every prompt and response
```

No database. Copy the folder to another machine and it opens.

## Modules

```
timeline.py   PURE  frozen models — the source of truth
media.py      PURE  ingest observations, kept apart from editorial decisions
ops.py        PURE  edit operations + apply
fingerprint.py PURE content hashing
ports.py            the whole system boundary
pipeline/           ingest · select · render · edit · plan · resolve
adapters/           ffmpeg · fs_store · ollama · diffusers · piper · music
```
