# 00 - Vision

## The goal

A video tool where the **user states an outcome** and the system produces the video,
then keeps refining it through conversation.

Not "a timeline UI with AI features bolted on". The user never touches a timeline.
They describe what they want, watch the result, and say what to change.

## What it is for

**You have a camera roll. You want a reel.**

Forty clips and photos from a weekend, a shoot, an event. Fifteen of them are worth
using. Each clip is fifteen seconds of which two are good. Finding those two seconds,
forty times over, then sequencing them to music, is the tedious part of making a reel -
and it is the part that automates well.

| Mode | You have | The system does | Status |
| --- | --- | --- | --- |
| **Camera roll mode** | photos and short clips | picks the keepers, trims each to its good moment, sequences, cuts to the beat, captions | **the product** |
| **Idea mode** | nothing but a description | generates visuals and narration from a prompt | parked - see backlog |

### Camera roll mode

> "Here are 40 clips and photos from the trip. Make me a 30-second reel."

The system reads every file, measures it, throws out the blurry and the duplicated,
finds the good seconds inside each clip, orders them into something with a shape, cuts
on the beat, and burns captions.

Photos and clips are the same problem and go through the same pipeline. A photo is
just a clip with no span to choose.

### Why this and not text-to-video

Generating video from a prompt is the crowded end of the market, and on a 4 GB GPU it
is also the end where this loses. Camera-roll editing inverts that: the footage is
already real, so image fidelity stops being a constraint, and the work that remains -
selection, trimming, pacing - is genuinely tedious by hand and genuinely automatable.

Idea mode is not deleted. It stays useful for title cards and for filling a gap where
no footage exists. It is simply not the pitch.

### The mode that matters most: refinement

The edit is never finished in one pass:

> "The second scene is too fast, and use a different song."

This is not a new request. It is a **patch to the video that already exists**.
The system must keep the project, apply the change, and re-render only what moved.

## What "AI generates it" actually means here

Two different model families run locally, and conflating them causes confusion:

| Model | Runs on | Produces |
| --- | --- | --- |
| **Text LLM** (Ollama) | `localhost:11434` | The plan: script, scene breakdown, image *prompts*, timing, music description, edit decisions |
| **Diffusion model** (`diffusers`) | GPU, in-process | The actual pixels for each scene |

The LLM cannot draw. It decides *what* should be drawn and hands a prompt to the
image model. Ollama does not serve diffusion models, so these are separate stacks.

## Hardware reality

Target machine: **RTX 3050 Laptop, 4 GB VRAM**.

| Capability | Viable at 4 GB? |
| --- | --- |
| Text LLM (7B, quantised) | Yes |
| Still image generation (SD 1.5 / SD-Turbo, 512-768px) | Yes, with attention slicing / sequential offload |
| Narration TTS (Piper, CPU) | Yes, no VRAM cost |
| **Generative video diffusion** (SVD, AnimateDiff, Sora-class) | **No** - needs 12-24 GB |

So motion comes from **camera moves on stills** - Ken Burns pan/zoom, parallax,
crossfades, push/slide transitions - executed by ffmpeg. This is how most
"AI text-to-video" products actually work, and it looks intentional rather than cheap
when the motion is chosen per scene instead of applied uniformly.

A real video-diffusion backend can be added later behind the same `ImageProvider` /
`ClipProvider` port without touching the pipeline.

## Design principles

Inherited from the global engineering conventions, and they fit this problem well:

- **The timeline is immutable data.** Every edit is a pure `Timeline -> Timeline` function.
- **Rendering is the only side effect.** Planning, patching, and validation are pure.
- **Explicit ownership.** A project is a directory on disk you can open and read.
  No hidden state, no implicit session context.
- **Deterministic given a seed.** Same timeline + same seed = same video.
- **The pipeline never hard-fails on a missing provider.** Every provider has a
  degraded fallback (procedural cards, silence, no music).
