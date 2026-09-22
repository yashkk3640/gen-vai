"""Filesystem store: content addressing, append-only history, media persistence."""

from pathlib import Path

import pytest

from genvai.adapters.fs_store import FilesystemStore
from genvai.errors import ProjectNotFound
from genvai.media import ClipQuality, MediaItem, MediaLibrary
from genvai.timeline import AssetProvenance, Timeline


@pytest.fixture
def store(tmp_path: Path) -> FilesystemStore:
    return FilesystemStore(tmp_path / "projects")


def _file(tmp_path: Path, name: str, content: bytes) -> Path:
    path = tmp_path / name
    path.write_bytes(content)
    return path


def test_create_is_idempotent(store: FilesystemStore) -> None:
    first = store.create("a trip reel", "trip")
    second = store.create("something else", "trip")
    assert first == second, "an existing project is returned untouched"


def test_project_id_defaults_to_a_slug(store: FilesystemStore) -> None:
    assert store.create("A Trip! To  the Sea").id == "a-trip-to-the-sea"


def test_missing_project_raises(store: FilesystemStore) -> None:
    with pytest.raises(ProjectNotFound, match="nope"):
        store.load("nope")


def test_identical_content_stores_once(store: FilesystemStore, tmp_path: Path) -> None:
    """Re-importing the same photo must not double the project's size."""
    store.create("t", "t")
    a = _file(tmp_path, "a.jpg", b"same bytes")
    b = _file(tmp_path, "b.jpg", b"same bytes")
    first = store.store_asset("t", a, "image", AssetProvenance(provider="test"))
    second = store.store_asset("t", b, "image", AssetProvenance(provider="test"))

    assert first.sha256 == second.sha256
    stored = list((store.project_dir("t") / "assets" / "images").glob("*.jpg"))
    assert len(stored) == 1


def test_different_content_stores_separately(store: FilesystemStore, tmp_path: Path) -> None:
    store.create("t", "t")
    first = store.store_asset(
        "t", _file(tmp_path, "a.jpg", b"one"), "image", AssetProvenance(provider="test")
    )
    second = store.store_asset(
        "t", _file(tmp_path, "b.jpg", b"two"), "image", AssetProvenance(provider="test")
    )
    assert first.sha256 != second.sha256


def test_asset_paths_are_relative_to_the_project(store: FilesystemStore, tmp_path: Path) -> None:
    """Absolute paths would not survive copying the project to another machine."""
    store.create("t", "t")
    asset = store.store_asset(
        "t", _file(tmp_path, "a.jpg", b"x"), "image", AssetProvenance(provider="test")
    )
    assert not Path(asset.path).is_absolute()
    assert asset.path.startswith("assets/")


def test_asset_path_resolves_across_kinds(store: FilesystemStore, tmp_path: Path) -> None:
    store.create("t", "t")
    clip = store.store_asset(
        "t", _file(tmp_path, "c.mp4", b"video"), "video", AssetProvenance(provider="test")
    )
    assert store.asset_path("t", clip.sha256).exists()


def test_timeline_history_is_append_only(store: FilesystemStore) -> None:
    store.create("t", "t")
    store.save_timeline("t", Timeline(intent="t", version=1))
    store.save_timeline("t", Timeline(intent="t", version=2, seed=7))

    assert store.versions("t") == (1, 2)
    assert store.load_timeline("t", 1).seed == 0, "v1 survives the later write"
    assert store.load_timeline("t").version == 2, "current follows the newest save"


def test_rewriting_a_version_with_different_content_is_refused(store: FilesystemStore) -> None:
    """Silently overwriting history would make an edit irreversible."""
    store.create("t", "t")
    store.save_timeline("t", Timeline(intent="t", version=1))
    with pytest.raises(FileExistsError, match="already exists"):
        store.save_timeline("t", Timeline(intent="t", version=1, seed=99))


def test_resaving_identical_content_is_allowed(store: FilesystemStore) -> None:
    store.create("t", "t")
    timeline = Timeline(intent="t", version=1)
    store.save_timeline("t", timeline)
    store.save_timeline("t", timeline)
    assert store.versions("t") == (1,)


def test_media_library_roundtrips(store: FilesystemStore) -> None:
    store.create("t", "t")
    library = MediaLibrary(
        items=(
            MediaItem(
                asset_id="a",
                kind="video",
                source_name="a.mov",
                width=1920,
                height=1080,
                duration=8.0,
                quality=ClipQuality(sharpness=0.8, exposure=0.5, motion=0.2, shake=0.1),
            ),
        )
    )
    store.save_media("t", library)
    assert store.load_media("t") == library


def test_media_defaults_to_empty(store: FilesystemStore) -> None:
    store.create("t", "t")
    assert store.load_media("t").items == ()


def test_listing_skips_directories_that_are_not_projects(
    store: FilesystemStore, tmp_path: Path
) -> None:
    store.create("real", "real")
    (tmp_path / "projects" / "stray").mkdir(parents=True, exist_ok=True)
    assert tuple(p.id for p in store.list_projects()) == ("real",)


def test_listing_an_absent_root_is_empty(tmp_path: Path) -> None:
    assert FilesystemStore(tmp_path / "nothing-here").list_projects() == ()
