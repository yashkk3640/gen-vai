# gen-vai

A generative, conversational video editor driven by a **local** LLM.

You describe the video you want. The system plans it, generates the visuals and
narration, proposes music, and renders it. Then you say what to change, and it
patches the result instead of starting over.

```bash
genvai create "30 second vertical explainer on compound interest, calm, soft piano"
genvai edit  <project> "scene 2 is too fast, and try a different song"
```

Three ways in:

- **Vision only** - no files at all; the system writes and generates everything
- **Images provided** - it arranges them and suggests music if you have none
- **Footage provided** - it transcribes, finds the good parts, and cuts *(later milestone)*

Everything runs locally: Ollama for the planning, Stable Diffusion for the visuals,
Piper for narration, ffmpeg for the render. The only network call is fetching a music
track, and that never happens without your explicit confirmation.

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
| [Vision](docs/00-vision.md) | What this is, and what "AI generates the video" actually means here |
| [Requirements](docs/01-requirements.md) | Functional and non-functional |
| [Architecture](docs/02-architecture.md) | Ports and adapters, data flow, incremental rendering |
| [Timeline schema](docs/03-timeline-schema.md) | The core data contract and the edit-op vocabulary |
| [Roadmap](docs/04-roadmap.md) | Milestones and open questions |
| [Setup](docs/05-setup.md) | Environment and portability |
| [Decisions](docs/06-decisions.md) | What was chosen, what was rejected, and why |

## Design notes

The **timeline is immutable data**. Planning produces one, edits are pure
`Timeline -> Timeline` transforms, and rendering is the only side effect. Every version
is kept on disk, so any edit is reversible and any result is reproducible from its seed.

A project is a **plain directory** - open it and you can see every asset, every timeline
version, and every LLM prompt that produced them.

Two constraints shaped the architecture, both documented in
[decisions](docs/06-decisions.md): 4 GB of VRAM cannot hold a text LLM and a diffusion
model at once, and nothing should reach the network without being asked.
