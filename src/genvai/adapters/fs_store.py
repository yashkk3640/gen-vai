"""ProjectStore over the filesystem.

A project is a directory you can open in a file manager: every timeline version, every
asset, and every LLM exchange is a visible file. Assets are content-addressed, so
identical content is stored once and any timeline reference is verifiable.

All stored paths are relative to the project directory, which is what lets a project
folder be copied to another machine and reopened.
"""

import json
import re
import shutil
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

from genvai.errors import ProjectNotFound
from genvai.fingerprint import hash_file
from genvai.media import MediaLibrary
from genvai.timeline import Asset, AssetProvenance, Project, Timeline

_ASSET_DIRS: dict[str, str] = {"image": "images", "video": "clips", "audio": "audio"}
_SLUG = re.compile(r"[^a-z0-9]+")


class FilesystemStore:
    """Implements `ProjectStore`."""

    def __init__(self, root: Path) -> None:
        self._root = Path(root)

    # ------------------------------------------------------------------ projects

    def create(self, intent: str, project_id: str | None = None) -> Project:
        """Create a project directory. An existing id is returned untouched."""
        identifier = project_id or _slugify(intent) or "project"
        directory = self.project_dir(identifier)
        if (directory / "project.json").exists():
            return self.load(identifier)

        now = _now()
        project = Project(id=identifier, intent=intent, created_at=now, updated_at=now)
        for sub in ("timelines", "cache/segments", "renders", "log"):
            (directory / sub).mkdir(parents=True, exist_ok=True)
        for sub in _ASSET_DIRS.values():
            (directory / "assets" / sub).mkdir(parents=True, exist_ok=True)
        self._write_json(directory / "project.json", project.model_dump(mode="json"))
        return project

    def load(self, project_id: str) -> Project:
        path = self.project_dir(project_id) / "project.json"
        if not path.exists():
            raise ProjectNotFound(f"no project '{project_id}' under {self._root}")
        return Project.model_validate_json(path.read_text(encoding="utf-8"))

    def list_projects(self) -> tuple[Project, ...]:
        if not self._root.exists():
            return ()
        found = [
            self.load(child.name)
            for child in sorted(self._root.iterdir())
            if (child / "project.json").exists()
        ]
        return tuple(found)

    # ----------------------------------------------------------------- timelines

    def save_timeline(self, project_id: str, timeline: Timeline) -> None:
        """Write `timelines/v{n}.json` and advance the current-version pointer.

        Refuses to overwrite an existing version file - history is append-only, which
        is what makes every edit reversible.
        """
        directory = self.project_dir(project_id) / "timelines"
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / f"v{timeline.version}.json"
        if path.exists():
            existing = Timeline.model_validate_json(path.read_text(encoding="utf-8"))
            if existing != timeline:
                raise FileExistsError(
                    f"timeline v{timeline.version} already exists for '{project_id}' "
                    "with different content; bump the version instead of overwriting"
                )
            return

        self._write_json(path, timeline.model_dump(mode="json"))
        project = self.load(project_id)
        updated = project.model_copy(
            update={"current_version": timeline.version, "updated_at": _now()}
        )
        self._write_json(
            self.project_dir(project_id) / "project.json", updated.model_dump(mode="json")
        )

    def load_timeline(self, project_id: str, version: int | None = None) -> Timeline:
        resolved = version if version is not None else self.load(project_id).current_version
        path = self.project_dir(project_id) / "timelines" / f"v{resolved}.json"
        if not path.exists():
            raise ProjectNotFound(f"no timeline v{resolved} for project '{project_id}'")
        return Timeline.model_validate_json(path.read_text(encoding="utf-8"))

    def versions(self, project_id: str) -> tuple[int, ...]:
        directory = self.project_dir(project_id) / "timelines"
        if not directory.exists():
            return ()
        numbers = sorted(int(p.stem[1:]) for p in directory.glob("v*.json") if p.stem[1:].isdigit())
        return tuple(numbers)

    # -------------------------------------------------------------------- assets

    def store_asset(
        self,
        project_id: str,
        source: Path,
        kind: Literal["image", "audio", "video"],
        provenance: AssetProvenance,
    ) -> Asset:
        """Copy a file in under its content hash. Idempotent.

        The id *is* the hash, so importing the same photo twice costs one file and the
        second import is a no-op.
        """
        digest = hash_file(source)
        relative = Path("assets") / _ASSET_DIRS[kind] / f"{digest}{source.suffix.lower()}"
        destination = self.project_dir(project_id) / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        if not destination.exists():
            shutil.copy2(source, destination)
        return Asset(
            kind=kind,
            path=relative.as_posix(),
            sha256=digest,
            provenance=provenance,
        )

    def asset_path(self, project_id: str, asset_id: str) -> Path:
        """Resolve an asset id to a file, whatever extension it was stored with."""
        base = self.project_dir(project_id) / "assets"
        for sub in _ASSET_DIRS.values():
            matches = sorted((base / sub).glob(f"{asset_id}.*"))
            if matches:
                return matches[0]
        raise ProjectNotFound(f"asset '{asset_id}' not found in project '{project_id}'")

    # --------------------------------------------------------------------- media

    def save_media(self, project_id: str, library: MediaLibrary) -> None:
        """Persist ingest results to `media.json`.

        Written once per ingest and read on every re-selection, so that changing your
        mind about the edit never re-analyses the footage.
        """
        self._write_json(
            self.project_dir(project_id) / "media.json", library.model_dump(mode="json")
        )

    def load_media(self, project_id: str) -> MediaLibrary:
        path = self.project_dir(project_id) / "media.json"
        if not path.exists():
            return MediaLibrary()
        return MediaLibrary.model_validate_json(path.read_text(encoding="utf-8"))

    # ------------------------------------------------------------------- layout

    def project_dir(self, project_id: str) -> Path:
        return self._root / project_id

    def segment_cache_dir(self, project_id: str) -> Path:
        path = self.project_dir(project_id) / "cache" / "segments"
        path.mkdir(parents=True, exist_ok=True)
        return path

    def renders_dir(self, project_id: str) -> Path:
        path = self.project_dir(project_id) / "renders"
        path.mkdir(parents=True, exist_ok=True)
        return path

    @staticmethod
    def _write_json(path: Path, payload: object) -> None:
        """Write atomically, so an interrupted save cannot truncate project state."""
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(path.suffix + ".tmp")
        temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        temporary.replace(path)


def _now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def _slugify(text: str, *, limit: int = 40) -> str:
    return _SLUG.sub("-", text.lower()).strip("-")[:limit].strip("-")
