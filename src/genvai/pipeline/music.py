"""Choosing a music bed, in two steps that cannot be collapsed into one.

`suggest` searches and presents. `approve` fetches, and only ever for a track the user
named. The split is the whole point: searching is free and reversible, downloading is
neither, and a design where one call does both has no moment at which to ask.

`attach` is the third way: a file the user already has, named by path. Naming the file
is the consent, and nothing touches the network.

See 'Music consent' in docs/decisions.md.
"""

from pathlib import Path

from genvai.beats import snap  # noqa: F401  (re-exported for callers that align cuts)
from genvai.errors import GenvaiError
from genvai.pipeline.select import fit_to_beats
from genvai.ports import BeatDetector, MusicProvider, ProjectStore
from genvai.timeline import AssetProvenance, Export, MusicQuery, Timeline

SOLE_TRACK_DB = -3.0
"""Gain for a track that is the only sound in the cut.

The default bed level, -18 dB, is set to sit under narration. A promo has none, and at
-18 the track is barely audible on a phone speaker.
"""


def suggest(
    project_id: str,
    store: ProjectStore,
    provider: MusicProvider,
    *,
    mood: str,
    limit: int = 5,
) -> Timeline:
    """Find candidate tracks and record them on the timeline.

    Read-only with respect to the network. The timeline moves to `candidates_ready`,
    which the renderer refuses - deliberately, because reaching the renderer in that
    state would mean nobody was ever asked which track to use.
    """
    timeline = store.load_timeline(project_id)
    if not provider.is_available():
        raise GenvaiError(
            "No music library found. Drop some audio files into the library directory, "
            "or point GENVAI_MUSIC__LIBRARY_DIR somewhere that has them."
        )

    query = MusicQuery(mood=mood)
    candidates = provider.search(query, limit=limit)
    if not candidates:
        raise GenvaiError(f"Nothing in the music library matches '{mood}'.")

    return _save(
        store,
        project_id,
        timeline.model_copy(
            update={
                "music": timeline.music.model_copy(
                    update={
                        "query": query,
                        "candidates": candidates,
                        "state": "candidates_ready",
                        "selected_candidate_id": None,
                    }
                )
            }
        ),
    )


def approve(
    project_id: str,
    store: ProjectStore,
    provider: MusicProvider,
    detector: BeatDetector,
    candidate_id: str,
    *,
    confirmed: bool,
    gain_db: float | None = None,
) -> Timeline:
    """Fetch an approved track, analyse its beat, and align the cuts to it.

    `confirmed` is passed through to the provider rather than assumed here, so the
    consent travels with the call instead of being inferred from having got this far.
    """
    timeline = store.load_timeline(project_id)
    candidate = next((c for c in timeline.music.candidates if c.id == candidate_id), None)
    if candidate is None:
        offered = ", ".join(c.id for c in timeline.music.candidates) or "none yet"
        raise GenvaiError(f"No candidate '{candidate_id}'. Suggested: {offered}")

    path = provider.fetch(candidate, confirmed=confirmed)
    asset = store.store_asset(
        project_id,
        path,
        "audio",
        AssetProvenance(
            provider=provider.name,
            source_url=candidate.source_url,
            licence=candidate.licence,
        ),
    )
    beats = detector.detect(path) if detector.is_available() else None

    resolved = timeline.model_copy(
        update={
            "assets": {**timeline.assets, asset.sha256: asset},
            "music": timeline.music.model_copy(
                update={
                    "selected_candidate_id": candidate_id,
                    "asset_id": asset.sha256,
                    "state": "resolved",
                    "beat_map": beats,
                    **({"gain_db": gain_db} if gain_db is not None else {}),
                }
            ),
        }
    )
    return _save(store, project_id, fit_to_beats(_with_full_cut(resolved)))


def attach(
    project_id: str,
    store: ProjectStore,
    detector: BeatDetector,
    track: Path,
    *,
    gain_db: float | None = None,
) -> Timeline:
    """Use a track the user handed over by path, and align the cuts to its beat.

    `gain_db` overrides the bed level - pass `SOLE_TRACK_DB` when nothing else in the
    cut makes a sound.
    """
    if not track.is_file():
        raise GenvaiError(f"No audio file at {track}")

    timeline = store.load_timeline(project_id)
    asset = store.store_asset(
        project_id, track, "audio", AssetProvenance(provider="user", prompt=track.name)
    )
    beats = detector.detect(track) if detector.is_available() else None
    music = timeline.music.model_copy(
        update={
            "state": "resolved",
            "asset_id": asset.sha256,
            "selected_candidate_id": None,
            "beat_map": beats,
            **({"gain_db": gain_db} if gain_db is not None else {}),
        }
    )
    resolved = timeline.model_copy(
        update={"assets": {**timeline.assets, asset.sha256: asset}, "music": music}
    )
    return _save(store, project_id, fit_to_beats(_with_full_cut(resolved)))


def decline(project_id: str, store: ProjectStore) -> Timeline:
    """Render without music. A real answer, not an absence of one."""
    timeline = store.load_timeline(project_id)
    return _save(
        store,
        project_id,
        timeline.model_copy(
            update={
                "music": timeline.music.model_copy(update={"state": "declined", "asset_id": None})
            }
        ),
    )


def _with_full_cut(timeline: Timeline) -> Timeline:
    """Make sure a cut with the music is exported, now that there is music.

    A promo exports only the silent cut until a track is chosen - a "full" cut with no
    track in it is the same file twice, and it promised sound it did not have.
    """
    variants = timeline.export.audio_variants
    if "full" in variants:
        return timeline
    export: Export = timeline.export.model_copy(update={"audio_variants": ("full", *variants)})
    return timeline.model_copy(update={"export": export})


def _save(store: ProjectStore, project_id: str, timeline: Timeline) -> Timeline:
    """Persist as the next version, so choosing music is as reversible as any edit."""
    versions = store.versions(project_id)
    numbered = timeline.model_copy(update={"version": (max(versions) + 1) if versions else 1})
    store.save_timeline(project_id, numbered)
    return numbered
