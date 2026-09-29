"""The system boundary, in one file.

Every external dependency - the LLM, the diffusion model, TTS, music, ffmpeg, the
filesystem - sits behind a Protocol here. Core and pipeline code imports only from
this module, never from `adapters`.

Keeping all of them together is deliberate: the entire surface the project depends on
should be readable in one sitting. See 'Ports and adapters' in docs/decisions.md.
"""

from pathlib import Path
from typing import Any, Literal, Protocol, TypeVar, runtime_checkable

import numpy as np
from numpy.typing import NDArray
from pydantic import BaseModel

from genvai.media import Face, MediaItem, MediaLibrary
from genvai.promo import Brief
from genvai.storyboard import Shot, Storyboard
from genvai.timeline import (
    Asset,
    AssetProvenance,
    AudioVariant,
    BeatMap,
    MusicCandidate,
    MusicQuery,
    Project,
    Timeline,
)

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
        model cannot coexist. See 'Phase-ordered model loading' in docs/decisions.md.
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
        network without the consent being visible in the code.
        See 'Music consent' in docs/decisions.md.
        """
        ...


@runtime_checkable
class MediaAnalyzer(Protocol):
    """Measure a photo or clip: quality, good spans, duplicates.

    The workhorse of camera-roll editing, and almost entirely classical CV - sharpness,
    exposure, motion, shake, perceptual hashing. Runs on CPU, needs no model, and is
    the reason selection works on a 4 GB machine.
    """

    def is_available(self) -> bool: ...

    def analyse(self, media: Path, item: MediaItem) -> MediaItem:
        """Return the item with `quality` and `spans` filled in.

        Pure with respect to the library: it observes one file and returns a new
        item, so analysis can be parallelised and cached per content hash.
        """
        ...

    def duplicate_groups(self, items: tuple[MediaItem, ...]) -> dict[str, str]:
        """Map asset_id -> group id for near-duplicates, via perceptual hashing.

        Only ids that belong to a group of two or more appear in the result.
        """
        ...


@runtime_checkable
class FaceDetector(Protocol):
    """Find faces in a picture.

    Optional. Everything that uses it - reframing, backdrop choice, `face_area` - works
    from detail saliency alone without one; faces are weighted on top when present.
    """

    @property
    def name(self) -> str: ...

    def is_available(self) -> bool: ...

    def detect(self, image: NDArray[np.uint8]) -> tuple[Face, ...]:
        """Faces in an RGB array, most confident first. Empty rather than raising."""
        ...


@runtime_checkable
class ShotRenderer(Protocol):
    """Storyboard shots to video, drawn frame by frame.

    Separate from `RendererPort`, which describes scenes as ffmpeg filtergraphs over
    footage. A storyboard shot is layers, particles and animated type - drawing.
    """

    @property
    def canvas(self) -> tuple[int, int]: ...

    def stage(
        self, board: Storyboard, shot: Shot, plates: tuple[Any, ...], posters: tuple[Any, ...]
    ) -> Any:
        """Prepare a shot once: backgrounds, card, text, particles."""
        ...

    def frame(self, stage: Any, t: float, *, cut_in: str, cut_out: str | None) -> Any:
        """One frame of a prepared shot, as a PIL image."""
        ...

    def legibility(self, stage: Any) -> tuple[tuple[str, str, float, str], ...]:
        """Each caption's text, role, contrast ratio against what is behind it, and the
        fix applied to reach it."""
        ...

    def render_shot(
        self, stage: Any, out: Path, *, frames: int, cut_in: str, cut_out: str | None
    ) -> Path: ...

    def join(self, segments: tuple[Path, ...], out: Path) -> Path: ...

    def with_music(self, video: Path, track: Path, out: Path, *, seconds: float) -> Path: ...


@runtime_checkable
class ContentTagger(Protocol):
    """Say what is in a photo or clip, so intent can be matched against it.

    Needed only for requests like "focus on the food". CLIP-style embedding
    comparison is enough and is cheap; a captioning model is better and is not.
    Optional - without it, selection falls back to quality and chronology.
    """

    def is_available(self) -> bool: ...

    def tag(self, media: Path, *, vocabulary: tuple[str, ...] = ()) -> tuple[str, ...]: ...

    def caption(self, media: Path) -> str | None: ...


@runtime_checkable
class BeatDetector(Protocol):
    """Find the beat grid of a music track.

    Onset detection, not a model. Cutting on the beat is most of what separates a
    reel that feels edited from one that feels assembled.
    """

    def is_available(self) -> bool: ...

    def detect(self, audio: Path) -> BeatMap: ...


@runtime_checkable
class VisionPort(Protocol):
    """Reads what is printed on a piece of artwork.

    Needed only for promo mode, where the content of the reel is on a poster rather than
    in a camera roll. Optional: without it the user supplies the offers instead, which is
    slower but never wrong.
    """

    @property
    def name(self) -> str: ...

    def is_available(self) -> bool: ...

    def read(self, image: Path) -> Brief:
        """Extract offers, business, occasion and phone from a poster.

        Returns an empty brief rather than raising when the model answers with nothing
        usable - losing the automation is a smaller failure than losing the command.
        """
        ...

    def unload(self) -> None: ...


@runtime_checkable
class TranscriptProvider(Protocol):
    """Audio to timed transcript.

    Only needed when a clip's own speech matters - a piece to camera that should be
    captioned. Not required for the ordinary camera-roll path, where clip audio is
    muted and the bed is music.
    """

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

    def concat_with(
        self, segments: tuple[Path, ...], transitions: tuple[tuple[str, float], ...], out: Path
    ) -> Path:
        """Join rendered segments, applying each scene's incoming transition.

        `transitions[i]` describes how `segments[i + 1]` enters. An all-cut timeline
        stream-copies; anything else cross-fades and therefore re-encodes.
        """
        ...

    def mix_audio(
        self, video: Path, timeline: Timeline, out: Path, variant: AudioVariant = "full"
    ) -> Path:
        """Lay audio over the cut, ducking music under speech.

        `full` mixes narration and music, `narration_only` omits the music bed, and
        `silent` writes no audio track at all. The variants are cheap - the video pass
        is already done - and the narration-only cut is the one to upload when the
        trending sound will be attached in-app.
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
    """Persistence. A project is a directory; see docs/architecture.md."""

    def create(self, intent: str, project_id: str | None = None) -> Project: ...

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

    def store_asset(
        self,
        project_id: str,
        source: Path,
        kind: Literal["image", "audio", "video"],
        provenance: AssetProvenance,
    ) -> Asset:
        """Copy a file into the project, addressed by content hash.

        Returns the stored Asset with its path and digest filled in. Idempotent:
        storing identical content twice yields one file.
        """
        ...

    def load_asset(self, project_id: str, asset_id: str) -> Asset:
        """Rebuild an Asset record for a stored file."""
        ...

    def asset_path(self, project_id: str, asset_id: str) -> Path: ...

    def save_media(self, project_id: str, library: MediaLibrary) -> None:
        """Persist ingest results to `media.json`.

        Written once per ingest and read on every re-selection, so that changing
        your mind about the edit never re-analyses the footage.
        """
        ...

    def load_media(self, project_id: str) -> MediaLibrary: ...

    def project_dir(self, project_id: str) -> Path: ...

    def segment_cache_dir(self, project_id: str) -> Path:
        """Where fingerprinted scene segments live. Created on demand."""
        ...

    def renders_dir(self, project_id: str) -> Path: ...
