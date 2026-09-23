"""MusicProvider over a local, tagged library with confirmation-gated downloads.

`search` is read-only and safe to run freely. `fetch` is the only place in the project
that writes a network resource to disk, and it refuses to run without an explicit
`confirmed=True` from a call site that actually asked. The flag is a parameter rather
than a config lookup precisely so that consent is visible in the code at every call.

See 'Music consent' in docs/decisions.md.
"""

import json
import re
from pathlib import Path

import httpx

from genvai.config import MusicSettings
from genvai.errors import ConfirmationRequired, ProviderUnavailable
from genvai.timeline import MusicCandidate, MusicQuery

AUDIO_SUFFIXES = frozenset({".mp3", ".wav", ".m4a", ".aac", ".ogg", ".opus", ".flac"})
DOWNLOAD_TIMEOUT = 60.0
MAX_DOWNLOAD_BYTES = 40 * 1024 * 1024
"""A music bed that will not fit in forty megabytes is not a music bed."""

_WORD = re.compile(r"[a-z0-9]+")


class LocalMusicProvider:
    """Implements `MusicProvider`.

    A track is any audio file in the library directory. An optional sidecar `.json`
    beside it carries what the file itself cannot say - mood, genre, tempo, licence:

        quiet-hours.mp3
        quiet-hours.json   {"mood": "calm", "genre": "piano", "bpm": 72,
                            "licence": "CC-BY-4.0", "attribution": "..."}

    Without a sidecar the filename is used as the title and matched against the query,
    which is weak but means dropping files into a folder is enough to get started.
    """

    name = "local"

    def __init__(self, settings: MusicSettings) -> None:
        self._settings = settings

    @property
    def library(self) -> Path:
        return Path(self._settings.library_dir)

    def is_available(self) -> bool:
        return self.library.exists()

    def search(self, query: MusicQuery, *, limit: int = 5) -> tuple[MusicCandidate, ...]:
        """Rank the library against a description. Downloads nothing.

        Everything is returned, best first, rather than only what matches: a small
        library would otherwise answer "calm piano" with nothing at all, and a track
        that is merely the closest available beats silence.
        """
        if not self.is_available():
            return ()
        tracks = [self._read(path) for path in self._audio_files()]
        scored = sorted(tracks, key=lambda t: -_score(t, query))
        return tuple(_as_candidate(t) for t in scored[:limit])

    def fetch(self, candidate: MusicCandidate, *, confirmed: bool) -> Path:
        """Make an approved track available as a file.

        A candidate already in the library is returned as-is - no network, no
        confirmation needed, because nothing leaves the machine.

        A remote candidate is downloaded only when `confirmed` is True *and* downloads
        are enabled in settings. Both gates exist on purpose: the setting is the
        standing policy, the flag is this particular consent.
        """
        local = Path(candidate.source_url)
        if local.exists():
            return local

        if not confirmed:
            raise ConfirmationRequired(
                f"'{candidate.title}' has not been approved for download. "
                f"Approve it first: genvai music <project> --approve {candidate.id}"
            )
        if not self._settings.allow_download:
            raise ConfirmationRequired(
                "Downloading is disabled. Set GENVAI_MUSIC__ALLOW_DOWNLOAD=true to "
                "permit it, then approve the track again."
            )
        return self._download(candidate)

    # ------------------------------------------------------------------- reading

    def _audio_files(self) -> list[Path]:
        return sorted(p for p in self.library.rglob("*") if p.suffix.lower() in AUDIO_SUFFIXES)

    def _read(self, path: Path) -> dict:
        """A track's metadata: its sidecar, filled out from the filename."""
        meta: dict = {}
        sidecar = path.with_suffix(".json")
        if sidecar.exists():
            try:
                meta = json.loads(sidecar.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                meta = {}
        meta.setdefault("title", path.stem.replace("-", " ").replace("_", " "))
        meta.setdefault("licence", "unknown")
        meta["path"] = str(path)
        return meta

    def _download(self, candidate: MusicCandidate) -> Path:
        """Fetch an approved track and record where it came from.

        The licence sidecar is written alongside, because a track whose provenance is
        lost is a track that cannot safely be published with.
        """
        self.library.mkdir(parents=True, exist_ok=True)
        suffix = Path(httpx.URL(candidate.source_url).path).suffix or ".mp3"
        destination = self.library / f"{_slug(candidate.title)}-{candidate.id}{suffix}"

        try:
            with httpx.stream(
                "GET", candidate.source_url, timeout=DOWNLOAD_TIMEOUT, follow_redirects=True
            ) as response:
                response.raise_for_status()
                written = 0
                with destination.open("wb") as handle:
                    for chunk in response.iter_bytes():
                        written += len(chunk)
                        if written > MAX_DOWNLOAD_BYTES:
                            raise ProviderUnavailable(
                                self.name, f"'{candidate.title}' is larger than the size limit"
                            )
                        handle.write(chunk)
        except httpx.HTTPError as exc:
            destination.unlink(missing_ok=True)
            raise ProviderUnavailable(
                self.name, f"could not download '{candidate.title}': {type(exc).__name__}"
            ) from exc
        except ProviderUnavailable:
            destination.unlink(missing_ok=True)
            raise

        destination.with_suffix(".json").write_text(
            json.dumps(
                {
                    "title": candidate.title,
                    "licence": candidate.licence,
                    "source_url": candidate.source_url,
                    "attribution": candidate.attribution,
                },
                indent=2,
            ),
            encoding="utf-8",
            newline="\n",
        )
        return destination


# ------------------------------------------------------------------------- scoring


def _score(track: dict, query: MusicQuery) -> float:
    """How well a track answers the description, 0-1.

    Mood carries the most weight because it is what the user actually said; tempo is a
    band rather than a target, since "around 70" should not reject 74.
    """
    text = " ".join(
        str(track.get(key, "")) for key in ("title", "mood", "genre", "tags", "description")
    ).lower()

    score = 0.0
    if query.mood and _mentions(text, query.mood):
        score += 0.45
    if query.genre and _mentions(text, query.genre):
        score += 0.3
    if query.instruments and any(_mentions(text, i) for i in query.instruments):
        score += 0.15

    bpm = track.get("bpm")
    if query.bpm and isinstance(bpm, int | float):
        low, high = query.bpm
        score += 0.1 if low <= float(bpm) <= high else 0.0
    return score


def _mentions(text: str, phrase: str) -> bool:
    return any(word in text for word in _WORD.findall(phrase.lower()))


def _as_candidate(track: dict) -> MusicCandidate:
    return MusicCandidate(
        id=_slug(track["title"])[:24] or "track",
        title=str(track["title"]),
        source_url=str(track["path"]),
        licence=str(track.get("licence", "unknown")),
        duration=float(track.get("duration", 0.0)),
        attribution=track.get("attribution"),
    )


def _slug(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")
