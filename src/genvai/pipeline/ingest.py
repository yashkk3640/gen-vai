"""Camera roll -> analysed media library.

Runs once per file, ever. Results are keyed by content hash, so re-importing the same
photos is free and changing your mind about the edit never re-analyses anything.

This is the slow phase and the one worth caching hardest: a hundred phone clips is a few
minutes of CPU, and the user should pay it exactly once.
"""

from collections.abc import Callable, Iterable
from pathlib import Path

from genvai.adapters.probe import probe, walk
from genvai.fingerprint import hash_file
from genvai.media import MediaItem, MediaLibrary
from genvai.ports import ContentTagger, MediaAnalyzer, ProjectStore
from genvai.timeline import AssetProvenance

Progress = Callable[[str, int, int], None]
"""Called with (filename, index, total) as each file is handled."""


def ingest(
    sources: tuple[Path, ...],
    project_id: str,
    analyzer: MediaAnalyzer,
    store: ProjectStore,
    ffmpeg: Path,
    *,
    tagger: ContentTagger | None = None,
    on_progress: Progress | None = None,
) -> MediaLibrary:
    """Import files, measure them, and group near-duplicates.

    Photos and clips take the same path; the only difference is that a still has no span
    to choose. `tagger` is optional - without it the library carries no content labels
    and selection falls back to quality and chronology, which is weaker but usable.

    Files already in the library are skipped by content hash, so re-running after adding
    a few more photos only pays for the new ones.
    """
    library = store.load_media(project_id)
    known = {item.asset_id for item in library.items}
    paths = walk(sources)

    added: list[MediaItem] = []
    for index, path in enumerate(paths, start=1):
        if on_progress:
            on_progress(path.name, index, len(paths))
        digest = hash_file(path)
        if digest in known:
            continue

        store.store_asset(
            project_id,
            path,
            "image" if path.suffix.lower() in _IMAGE else "video",
            AssetProvenance(provider="import", source_url=path.name),
        )
        item = analyzer.analyse(path, probe(path, digest, ffmpeg))
        if tagger is not None and tagger.is_available():
            item = item.model_copy(
                update={"tags": tagger.tag(path), "caption": tagger.caption(path)}
            )
        added.append(item)
        known.add(digest)

    merged = MediaLibrary(items=(*library.items, *added))
    regrouped = _regroup(merged, analyzer)
    store.save_media(project_id, regrouped)
    return regrouped


def probe_only(sources: tuple[Path, ...], ffmpeg: Path) -> MediaLibrary:
    """Read dimensions, duration and capture time without analysing.

    Fast enough to run on drop, so the user sees what was picked up before committing to
    the slow pass.
    """
    items = [probe(path, hash_file(path), ffmpeg) for path in walk(sources)]
    return MediaLibrary(items=tuple(items))


def _regroup(library: MediaLibrary, analyzer: MediaAnalyzer) -> MediaLibrary:
    """Redo duplicate grouping across the whole library.

    Over the whole library, not just the new files: a photo imported today may be a
    second take of one imported last week, and grouping only within a batch would miss
    it. Fingerprints are stored on the items, so this costs no file reads.
    """
    groups = analyzer.duplicate_groups(library.items)
    return MediaLibrary(
        schema_version=library.schema_version,
        items=tuple(
            item.model_copy(update={"dedup_group": groups.get(item.asset_id)})
            for item in library.items
        ),
    )


def summarise(items: Iterable[MediaItem]) -> str:
    """A one-line count for the CLI: what came in, and what is worth using."""
    listed = list(items)
    clips = sum(1 for i in listed if i.kind == "video")
    usable = sum(1 for i in listed if i.quality and i.quality.usable)
    duplicates = sum(1 for i in listed if i.dedup_group)
    parts = [f"{len(listed)} files ({clips} clips, {len(listed) - clips} photos)"]
    parts.append(f"{usable} usable")
    if duplicates:
        parts.append(f"{duplicates} in duplicate groups")
    return ", ".join(parts)


_IMAGE = frozenset(
    {".jpg", ".jpeg", ".png", ".heic", ".heif", ".webp", ".bmp", ".tif", ".tiff", ".gif"}
)
