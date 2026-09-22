# gen-vai

**You have a camera roll. You want a reel.**

Forty clips and photos from a weekend. Fifteen are worth using, each clip has about two
good seconds in it, and finding those two seconds forty times over is the tedious part.
That is the part this automates.

```bash
genvai add trip ~/Pictures/trip/          # import and analyse
genvai reel trip --duration 30            # pick, trim, sequence, caption
genvai edit trip "drop the third clip and slow the sunset one down"
```

It reads every file, measures it, throws out the blurry and the duplicated, finds the
good seconds inside each clip, orders them into something with a shape, cuts on the
beat, and burns captions. Then you tell it what to change and it patches the result
instead of starting over.

Everything runs locally - Ollama for the editorial judgement, OpenCV for the analysis,
ffmpeg for the render. Your footage never leaves the machine. The only network call is
fetching a music track, and that never happens without your explicit confirmation.

**Generating video from a text prompt** is also in here, but it is deliberately not the
headline: it is the crowded end of the market and the end where a 4 GB GPU loses.
Camera-roll editing inverts that - the footage is already real, so image quality stops
being a constraint. See [docs/00-vision.md](docs/00-vision.md).

## Status

**Milestone 0** - documentation, portable environment, and typed scaffold.
Nothing renders yet; adapters are stubs. See [docs/04-roadmap.md](docs/04-roadmap.md).

## Quick start

```bash
uv sync
uv run genvai doctor      # reports what is available on this machine
```

Full setup, optional extras, and how to move the project to another system:
[docs/05-setup.md](docs/05-setup.md).

## Documentation

Start at **[docs/README.md](docs/README.md)**.

| | |
| --- | --- |
| [Vision](docs/00-vision.md) | What this is for, and why camera roll over text-to-video |
| [Requirements](docs/01-requirements.md) | Functional and non-functional |
| [Architecture](docs/02-architecture.md) | Ports and adapters, data flow, incremental rendering |
| [Timeline schema](docs/03-timeline-schema.md) | The core data contract and the edit-op vocabulary |
| [Backlog](docs/07-backlog.md) | Understood but deliberately not built, and why |
| [Roadmap](docs/04-roadmap.md) | Milestones and open questions |
| [Setup](docs/05-setup.md) | Environment and portability |
| [Decisions](docs/06-decisions.md) | What was chosen, what was rejected, and why |

## Design notes

The **timeline is immutable data**. Planning produces one, edits are pure
`Timeline -> Timeline` transforms, and rendering is the only side effect. Every version
is kept on disk, so any edit is reversible and any result is reproducible from its seed.

A project is a **plain directory** - open it and you can see every asset, every timeline
version, and every LLM prompt that produced them.

The work is split so the machine does the measuring and the model does the judging.
Sharpness, exposure, shake and duplicate detection are classical CV on the CPU - no
model, no GPU. The LLM is only asked what it is actually good at: what order tells a
story, what the opening shot should be, what the text says. That keeps the edit
reproducible and works on modest hardware.
