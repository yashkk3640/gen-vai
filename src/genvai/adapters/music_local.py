"""MusicProvider over a local, tagged library with confirmation-gated downloads.

`search` scans the local cache first and only consults remote catalogues for metadata.
`fetch` is the sole place in the project that writes a network resource to disk, and it
refuses to run without an explicit `confirmed=True` from a call site that asked the
user. See docs/06-decisions.md D4.
"""

from pathlib import Path

from genvai.config import MusicSettings
from genvai.timeline import MusicCandidate, MusicQuery


class LocalMusicProvider:
    """Implements `MusicProvider`."""

    name = "local"

    def __init__(self, settings: MusicSettings) -> None:
        self._settings = settings

    def is_available(self) -> bool:
        return self._settings.library_dir.exists()

    def search(self, query: MusicQuery, *, limit: int = 5) -> tuple[MusicCandidate, ...]:
        """Rank candidates by mood, genre, tempo and energy. Downloads nothing.

        Already-cached tracks rank first so a repeat request costs no network at all.
        """
        raise NotImplementedError

    def fetch(self, candidate: MusicCandidate, *, confirmed: bool) -> Path:
        """Download an approved track and write a sidecar `.license.json` beside it.

        Raises `ConfirmationRequired` when `confirmed` is False, and when
        `settings.allow_download` is False.
        """
        raise NotImplementedError
