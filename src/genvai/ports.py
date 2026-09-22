"""The system boundary, in one file.

Every external dependency - the LLM, the diffusion model, TTS, music, ffmpeg, the
filesystem - sits behind a Protocol here. Core and pipeline code imports only from
this module, never from `adapters`.

Keeping all of them together is deliberate: the entire surface the project depends on
should be readable in one sitting. See docs/06-decisions.md D5.
"""

from pathlib import Path
from typing import Protocol, TypeVar, runtime_checkable

from pydantic import BaseModel

from genvai.timeline import Asset, MusicCandidate, MusicQuery, Project, Timeline

T = TypeVar("T", bound=BaseModel)


@runtime_checkable
class LLMPort(Protocol):
    """A local text model. The only required provider - nothing plans without it."""

    def complete(self, prompt: str, *, system: str | None = None) -> str:
        """Free-form completion. Used for rationales and summaries, not for structure."""
        ...

    def structured(
        self,
        prompt: str,
        schema: type[T],
        *,
        system: str | None = None,
        max_retries: int = 3,
    ) -> T:
        """Completion constrained to `schema`.

        Invalid output is retried with the validation error fed back to the model.
        Raises `PlanningError` once the retry budget is exhausted - malformed output is
        never silently repaired, because a quietly patched plan is worse than a failure.
        """
        ...

    def unload(self) -> None:
        """Release the model from VRAM.

        Called at the plan/resolve phase boundary; at 4 GB the LLM and the diffusion
        model cannot coexist. See docs/06-decisions.md D8.
        """
        ...


@runtime_checkable
class ImageProvider(Protocol):
    """Prompt to still image. Falls back to procedural cards when no backend exists."""

    @property
    def name(self) -> str: ...

    def is_available(self) -> bool:
        """False when the backend is missing. The caller degrades rather than failing."""
        ...

    def generate(
        self,
        prompt: str,
        *,
        width: int,
        height: int,
        seed: int,
        negative_prompt: str | None = None,
    ) -> Path:
        """Render one image and return its path. Deterministic for a given seed."""
        ...

    def unload(self) -> None: ...


@runtime_checkable
class SpeechProvider(Protocol):
    """Text to narration audio."""

    @property
    def name(self) -> str: ...

    def is_available(self) -> bool: ...

    def synthesise(self, text: str, *, voice: str | None = None, speed: float = 1.0) -> Path:
        """Speak `text` to an audio file."""
        ...


@runtime_checkable
class MusicProvider(Protocol):
    """Music search and fetch.

    Split into two calls on purpose. `search` is read-only and safe to run freely;
    `fetch` touches the network and is guarded by an explicit confirmation flag.
    """

    @property
    def name(self) -> str: ...

    def is_available(self) -> bool: ...

    def search(self, query: MusicQuery, *, limit: int = 5) -> tuple[MusicCandidate, ...]:
        """Find candidate tracks. Must not download anything."""
        ...

    def fetch(self, candidate: MusicCandidate, *, confirmed: bool) -> Path:
        """Download an approved track.

        Raises `ConfirmationRequired` unless `confirmed` is True. The flag is passed
        explicitly rather than read from config so that no call site can reach the
        network without the consent being visible in the code. See docs/06-decisions.md D4.
        """
        ...


@runtime_checkable
class TranscriptProvider(Protocol):
    """Audio to timed transcript. Needed only for Mode C (existing footage)."""

    def is_available(self) -> bool: ...

    def transcribe(self, media: Path, *, language: str | None = None) -> "Transcript": ...


class TranscriptWord(BaseModel):
    text: str
    start: float
    end: float


class TranscriptSegment(BaseModel):
    text: str
    start: float
    end: float
    words: tuple[TranscriptWord, ...] = ()


class Transcript(BaseModel):
    language: str
    segments: tuple[TranscriptSegment, ...]

    @property
    def text(self) -> str:
        return " ".join(s.text.strip() for s in self.segments)


@runtime_checkable
class RendererPort(Protocol):
    """Timeline to video file. The only component that touches ffmpeg."""

    def render_scene(self, timeline: Timeline, scene_id: str, out: Path) -> Path:
        """Render one scene to a segment, for the fingerprint cache."""
        ...

    def concat(self, segments: tuple[Path, ...], out: Path) -> Path:
        """Join rendered segments, applying transitions between them."""
        ...

    def mix_audio(self, video: Path, timeline: Timeline, out: Path) -> Path:
        """Lay narration and music over the cut, ducking music under speech.

        Separate from the video pass so an audio-only change skips re-encoding video.
        """
        ...

    def probe(self, media: Path) -> "MediaInfo": ...


class MediaInfo(BaseModel):
    duration: float
    width: int | None = None
    height: int | None = None
    fps: float | None = None
    has_audio: bool = False


@runtime_checkable
class ProjectStore(Protocol):
    """Persistence. A project is a directory; see docs/02-architecture.md."""

    def create(self, intent: str) -> Project: ...

    def load(self, project_id: str) -> Project:
        """Raises `ProjectNotFound` if it does not exist."""
        ...

    def list_projects(self) -> tuple[Project, ...]: ...

    def save_timeline(self, project_id: str, timeline: Timeline) -> None:
        """Write a new version. Never overwrites an existing one, so edits stay reversible."""
        ...

    def load_timeline(self, project_id: str, version: int | None = None) -> Timeline:
        """Load a version, or the current one when `version` is None."""
        ...

    def versions(self, project_id: str) -> tuple[int, ...]: ...

    def store_asset(self, project_id: str, source: Path, asset: Asset) -> Asset:
        """Move a file into the project, addressed by content hash.

        Idempotent: storing identical content twice yields one file.
        """
        ...

    def asset_path(self, project_id: str, asset_id: str) -> Path: ...

    def project_dir(self, project_id: str) -> Path: ...
