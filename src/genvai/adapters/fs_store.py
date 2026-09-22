"""ProjectStore over the filesystem.

A project is a directory you can open in a file manager: every timeline version, every
asset, and every LLM exchange is a visible file. Assets are content-addressed, so
identical content is stored once and any timeline reference is verifiable.

All stored paths are relative to the project directory, which is what lets a project
folder be copied to another machine and reopened.
"""

from pathlib import Path

from genvai.timeline import Asset, Project, Timeline


class FilesystemStore:
    """Implements `ProjectStore`."""

    def __init__(self, root: Path) -> None:
        self._root = root

    def create(self, intent: str) -> Project:
        raise NotImplementedError

    def load(self, project_id: str) -> Project:
        raise NotImplementedError

    def list_projects(self) -> tuple[Project, ...]:
        raise NotImplementedError

    def save_timeline(self, project_id: str, timeline: Timeline) -> None:
        """Write `timelines/v{n}.json` and advance the current-version pointer.

        Refuses to overwrite an existing version file - history is append-only, which
        is what makes every edit reversible.
        """
        raise NotImplementedError

    def load_timeline(self, project_id: str, version: int | None = None) -> Timeline:
        raise NotImplementedError

    def versions(self, project_id: str) -> tuple[int, ...]:
        raise NotImplementedError

    def store_asset(self, project_id: str, source: Path, asset: Asset) -> Asset:
        """Move a file in under its content hash. Idempotent."""
        raise NotImplementedError

    def asset_path(self, project_id: str, asset_id: str) -> Path:
        raise NotImplementedError

    def project_dir(self, project_id: str) -> Path:
        return self._root / project_id
