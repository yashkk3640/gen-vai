"""Camera roll -> analysed media library.

Runs once per file, ever. Results are keyed by content hash, so re-importing the same
photos is free and changing your mind about the edit never re-analyses anything.

This is the slow phase and the one worth caching hardest: a hundred phone clips is a
few minutes of CPU, and the user should pay it exactly once.
"""

from pathlib import Path

from genvai.media import MediaLibrary
from genvai.ports import ContentTagger, MediaAnalyzer, ProjectStore


def ingest(
    sources: tuple[Path, ...],
    project_id: str,
    analyzer: MediaAnalyzer,
    store: ProjectStore,
    *,
    tagger: ContentTagger | None = None,
) -> MediaLibrary:
    """Import files, measure them, and group near-duplicates.

    Photos and clips go through the same path; the difference is only that a still has
    no spans. `tagger` is optional - without it the library carries no content labels
    and selection falls back to quality and chronology, which is a weaker but perfectly
    usable edit.

    Already-ingested files are skipped by hash.
    """
    raise NotImplementedError


def probe_only(sources: tuple[Path, ...]) -> MediaLibrary:
    """Read dimensions, duration and capture time without analysing.

    Fast enough to run on drop, so the user sees what was picked up before committing
    to the slow pass.
    """
    raise NotImplementedError
