# 00 - Vision

## The goal

A video tool where the **user states an outcome** and the system produces the video,
then keeps refining it through conversation.

Not "a timeline UI with AI features bolted on". The user never touches a timeline.
They describe what they want, watch the result, and say what to change.

## Three input modes

Named after what the user already has. The system handles all three with one pipeline.

| Mode | You have | The system does |
| --- | --- | --- |
| **Idea mode** | nothing but a description | writes, generates and renders everything |
| **Photo mode** | images | orders them, times them, adds text and music |
| **Footage mode** | a long recording | transcribes, finds the good parts, cuts clips |

### Idea mode - you have nothing but an idea

> "Make me a 30-second vertical video explaining why compound interest matters,
> calm tone, soft piano."

Nothing is provided. The system writes the script, generates every visual,
synthesises narration, proposes music, and renders.

### Photo mode - you have images

> "Here are 12 photos from the trip. Make a 45-second reel."

The system orders them, decides timing and motion, writes on-screen text,
and proposes music if none was given.

### Footage mode - you have a recording

> "Here is a 40-minute talk. Cut it into three shorts."

The system transcribes, finds highlights, cuts, reframes, captions.
*(Later milestone - see roadmap.)*

### The mode that matters most: refinement

Every mode is followed by:

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
