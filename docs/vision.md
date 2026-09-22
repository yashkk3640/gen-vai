# Vision

**You have a camera roll. You want a reel.**

Forty clips and photos from a weekend. Fifteen are worth using, each clip holds about two
good seconds, and finding those two seconds forty times over is the tedious part. That is
what this automates.

## The three inputs

| Mode | You have | Status |
| --- | --- | --- |
| **Camera roll** | photos and short clips | the product |
| **Idea** | nothing but a description | parked |

Photos and clips go through one pipeline — a photo is just a clip with no span to choose.

## Why not text-to-video

Generating video from a prompt is the crowded end of the market, and on a 4 GB GPU it is
the end where this loses. Camera-roll editing inverts that: the footage is already real,
so image fidelity stops being a constraint, and the work that remains — selection,
trimming, pacing — is tedious by hand and automates well.

Idea mode is not deleted. It stays useful for title cards and gaps where no footage
exists. It is just not the pitch.

## What the machine does vs. what the model does

Keeping these apart is the core idea.

- **Measuring** is objective and cheap: sharpness, exposure, shake, duplicates. Classical
  CV on the CPU. No model, no GPU.
- **Shortlisting** is pure and deterministic: drop the unusable, keep the best of each
  duplicate group, rank by span score.
- **Judging** is the LLM's job, and only that: what order tells a story, what the opening
  shot should be, what the text says.

Small context, reproducible edits, and it works on modest hardware.

## Principles

- The timeline is immutable data. Every edit is a pure `Timeline → Timeline` function.
- Rendering is the only side effect.
- A project is a directory you can read. No hidden state.
- Same timeline and seed, same video.
- No provider is ever required. Everything has a fallback.
