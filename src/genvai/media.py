"""What is known about each photo and clip the user dropped in.

Kept separate from `timeline.py` on purpose. The timeline holds *editorial decisions* -
what made the cut and how it is treated. This module holds *observations* - how sharp a
clip is, whether it duplicates another, what is in it. Ingest writes these once; the
selector reads them; edits never change them.

Nearly all of it is ordinary signal processing on the CPU. Sharpness, exposure, shake
and duplicate detection need no model at all, which matters on a 4 GB machine: the
expensive part of camera-roll editing is the judgement, not the measurement.
"""

from typing import Literal

from pydantic import Field

from genvai.timeline import Frozen, Rect, Seconds

MEDIA_SCHEMA_VERSION = 1


class Face(Frozen):
    """A face found in a picture, in fractions of the frame."""

    rect: Rect
    confidence: float = Field(ge=0.0, le=1.0)

    @property
    def area(self) -> float:
        return max(0.0, self.rect[2]) * max(0.0, self.rect[3])

    @property
    def centre(self) -> tuple[float, float]:
        x, y, w, h = self.rect
        return x + w / 2, y + h / 2


class ClipQuality(Frozen):
    """Objective measures, all normalised to 0-1 so they can be weighted and summed.

    These answer "is this shot usable", never "is this shot interesting". Keeping the
    two apart matters: a blurry shot of the thing you care about should lose on
    sharpness but still be findable, rather than being silently dropped at ingest.
    """

    sharpness: float = Field(ge=0.0, le=1.0, description="Laplacian variance, normalised.")
    exposure: float = Field(
        ge=0.0, le=1.0, description="0 is crushed black, 1 is blown out, ~0.5 is good."
    )
    motion: float = Field(ge=0.0, le=1.0, description="Subject/camera movement magnitude.")
    shake: float = Field(ge=0.0, le=1.0, description="Handheld instability. Lower is better.")
    face_area: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Largest face as a fraction of frame. Faces hold attention.",
    )
    audio_peak_db: float | None = Field(
        default=None, description="None when the clip has no audio track."
    )

    @property
    def usable(self) -> bool:
        """A cheap first filter. Deliberately permissive - it rejects, it does not choose."""
        return self.sharpness > 0.25 and 0.15 < self.exposure < 0.9 and self.shake < 0.8


class Span(Frozen):
    """A stretch of a clip, with a score for how good it is.

    Analysis proposes several per clip. Selection picks at most one. This is where most
    of the value is: a twenty-second phone clip usually contains two seconds worth
    keeping, and finding them by hand is the tedious part of making a reel.
    """

    start: Seconds
    end: Seconds
    score: float = Field(ge=0.0, le=1.0)
    reason: str = Field(default="", description="Why this span scored well. Shown to the user.")

    @property
    def duration(self) -> float:
        return max(0.0, self.end - self.start)


class MediaItem(Frozen):
    """One ingested photo or clip.

    `asset_id` is the content hash, so re-dropping the same file is free and a camera
    roll imported twice does not double.
    """

    asset_id: str
    kind: Literal["image", "video"]
    source_name: str = Field(description="Original filename, for talking to the user about it.")
    width: int
    height: int
    duration: Seconds | None = Field(default=None, description="None for stills.")
    fps: float | None = None
    captured_at: str | None = Field(
        default=None,
        description=(
            "EXIF/container capture time, ISO-8601. Chronological order is the default "
            "narrative for a camera roll, and it is usually the right one."
        ),
    )
    rotation: int = Field(default=0, description="EXIF orientation, degrees clockwise.")

    quality: ClipQuality | None = Field(default=None, description="None until analysed.")
    spans: tuple[Span, ...] = Field(
        default=(), description="Candidate good moments, best first. Empty for stills."
    )
    tags: tuple[str, ...] = Field(
        default=(),
        description=(
            "Content labels, for intent matching - 'make it about the food'. From CLIP "
            "or a small vision model; empty when neither is installed."
        ),
    )
    caption: str | None = None
    fingerprint: int | None = Field(
        default=None,
        description=(
            "Perceptual hash of a representative frame. Kept on the item so duplicate "
            "grouping can be redone without re-reading the files."
        ),
    )
    dedup_group: str | None = Field(
        default=None,
        description=(
            "Near-duplicates share a group id. Camera rolls are full of five near "
            "identical takes; the selector keeps the best of each group and the rest "
            "stay available rather than being deleted."
        ),
    )

    @property
    def is_vertical(self) -> bool:
        return self.height > self.width

    @property
    def best_span(self) -> Span | None:
        return max(self.spans, key=lambda s: s.score) if self.spans else None


class MediaLibrary(Frozen):
    """Everything ingested for one project. Persisted as `media.json`.

    Separate from the timeline so that re-selecting - "use more of the beach ones" -
    costs no re-analysis. Analysis is the slow part; it happens once per file, ever.
    """

    schema_version: int = MEDIA_SCHEMA_VERSION
    items: tuple[MediaItem, ...] = ()

    def item(self, asset_id: str) -> MediaItem | None:
        return next((i for i in self.items if i.asset_id == asset_id), None)

    @property
    def unanalysed(self) -> tuple[MediaItem, ...]:
        return tuple(i for i in self.items if i.quality is None)

    @property
    def clips(self) -> tuple[MediaItem, ...]:
        return tuple(i for i in self.items if i.kind == "video")

    @property
    def photos(self) -> tuple[MediaItem, ...]:
        return tuple(i for i in self.items if i.kind == "image")

    def deduplicated(self) -> tuple[MediaItem, ...]:
        """One representative per duplicate group - the sharpest - plus everything ungrouped.

        Nothing is discarded; this is a view. The user can always ask for a take that
        lost, and the selector should be able to explain why it did.
        """
        best: dict[str, MediaItem] = {}
        loose: list[MediaItem] = []
        for item in self.items:
            if item.dedup_group is None:
                loose.append(item)
                continue
            incumbent = best.get(item.dedup_group)
            sharper = (
                incumbent is None
                or item.quality is None
                or incumbent.quality is None
                or item.quality.sharpness > incumbent.quality.sharpness
            )
            if sharper:
                best[item.dedup_group] = item
        return tuple(loose) + tuple(best.values())
