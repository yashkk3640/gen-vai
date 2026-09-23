"""Idea mode: a video from a description, with no footage.

The planner and the procedural provider run here for real. The diffusion provider is
covered only for how it *degrades*, since torch is not installed in the test environment
- which is also the common case on a user's machine.
"""

from pathlib import Path

import pytest
from PIL import Image

from genvai.adapters.fs_store import FilesystemStore
from genvai.adapters.images import image_provider
from genvai.adapters.procedural_image import PALETTES, ProceduralImageProvider
from genvai.config import ImageSettings
from genvai.errors import PlanningError
from genvai.pipeline.plan import Beat, Storyboard, plan
from genvai.pipeline.resolve import resolve_visuals
from genvai.timeline import Canvas, GeneratedVisual


class _FakeLLM:
    def __init__(self, board: Storyboard | None = None) -> None:
        self._board = board

    def structured(self, prompt: str, schema: type, **kw: object) -> Storyboard:
        if self._board is None:
            raise RuntimeError("model is having a bad day")
        return self._board

    def complete(self, prompt: str, **kw: object) -> str:
        return ""

    def unload(self) -> None:
        return None


def _board(*pairs: tuple[str, str]) -> Storyboard:
    return Storyboard(beats=tuple(Beat(text=t, visual=v) for t, v in pairs))


# ------------------------------------------------------------------------ planning


def test_a_storyboard_becomes_a_timeline() -> None:
    llm = _FakeLLM(_board(("One", "a sunrise"), ("Two", "a city"), ("Three", "a lake")))
    timeline = plan("a short film", llm, target_duration=9.0)  # type: ignore[arg-type]

    assert len(timeline.scenes) == 3
    assert timeline.duration == pytest.approx(9.0)
    assert [s.role for s in timeline.scenes] == ["hook", "body", "payoff"]


def test_the_text_becomes_an_overlay() -> None:
    llm = _FakeLLM(_board(("Hello there", "a sunrise")))
    scene = plan("x", llm, target_duration=4.0).scenes[0]  # type: ignore[arg-type]
    assert scene.overlays[0].content == "Hello there"


def test_the_backdrop_becomes_a_prompt() -> None:
    llm = _FakeLLM(_board(("", "a quiet harbour at dawn")))
    visual = plan("x", llm, target_duration=4.0).scenes[0].visual  # type: ignore[arg-type]
    assert isinstance(visual, GeneratedVisual)
    assert visual.prompt == "a quiet harbour at dawn"
    assert visual.asset_id is None, "planning describes; resolving generates"


def test_a_beat_with_no_text_gets_no_overlay() -> None:
    llm = _FakeLLM(_board(("", "a sunrise")))
    assert plan("x", llm, target_duration=4.0).scenes[0].overlays == ()  # type: ignore[arg-type]


def test_camera_moves_alternate() -> None:
    """Identical motion on every beat is what makes a generated sequence a slideshow."""
    llm = _FakeLLM(_board(("a", "x"), ("b", "y"), ("c", "z")))
    scenes = plan("x", llm, target_duration=9.0).scenes  # type: ignore[arg-type]
    assert scenes[0].motion.start_rect != scenes[1].motion.start_rect  # type: ignore[union-attr]


def test_the_style_suffix_reaches_every_prompt() -> None:
    llm = _FakeLLM(_board(("a", "a harbour"), ("b", "a street")))
    timeline = plan("x", llm, target_duration=6.0, style_suffix="muted film grain")  # type: ignore[arg-type]
    assert all("muted film grain" in s.visual.prompt for s in timeline.scenes)  # type: ignore[union-attr]


def test_beat_count_follows_the_target() -> None:
    llm = _FakeLLM(_board(*[(f"b{i}", "x") for i in range(12)]))
    short = plan("x", llm, target_duration=6.0)  # type: ignore[arg-type]
    assert 2 <= len(short.scenes) <= 12


def test_an_empty_idea_is_refused() -> None:
    with pytest.raises(PlanningError, match="Describe what you want"):
        plan("   ", None)


# ------------------------------------------------------------------- degrading


def test_without_a_model_the_users_words_are_used() -> None:
    """A video rather than an error - the whole promise is that it runs locally."""
    timeline = plan("Start here. Then this. Finally that.", None, target_duration=9.0)
    assert len(timeline.scenes) == 3
    assert timeline.scenes[0].overlays[0].content == "Start here"


def test_a_failing_model_falls_back_rather_than_raising() -> None:
    timeline = plan("One thing. Another thing.", _FakeLLM(None), target_duration=6.0)  # type: ignore[arg-type]
    assert len(timeline.scenes) == 2


def test_an_idea_with_no_full_stops_still_works() -> None:
    assert plan("just one idea", None, target_duration=6.0).scenes


def test_empty_beats_are_discarded() -> None:
    llm = _FakeLLM(_board(("real", "a harbour"), ("", ""), ("also real", "a street")))
    assert len(plan("x", llm, target_duration=6.0).scenes) == 2  # type: ignore[arg-type]


# --------------------------------------------------------------- procedural images


def test_a_backdrop_is_produced(tmp_path: Path) -> None:
    provider = ProceduralImageProvider(tmp_path)
    out = provider.generate("a calm ocean", width=120, height=200, seed=3)
    with Image.open(out) as image:
        assert image.size == (120, 200)


def test_the_same_prompt_and_seed_give_the_same_image(tmp_path: Path) -> None:
    provider = ProceduralImageProvider(tmp_path)
    first = provider.generate("a calm ocean", width=64, height=64, seed=3)
    second = provider.generate("a calm ocean", width=64, height=64, seed=3)
    assert first.read_bytes() == second.read_bytes()


def test_different_prompts_give_different_images(tmp_path: Path) -> None:
    provider = ProceduralImageProvider(tmp_path)
    a = provider.generate("a calm ocean", width=64, height=64, seed=3)
    b = provider.generate("a busy street", width=64, height=64, seed=3)
    assert a.read_bytes() != b.read_bytes()


def test_size_is_part_of_the_cache_key(tmp_path: Path) -> None:
    """Otherwise a preview render silently overwrites the full-size image."""
    provider = ProceduralImageProvider(tmp_path)
    small = provider.generate("a calm ocean", width=64, height=64, seed=3)
    large = provider.generate("a calm ocean", width=128, height=128, seed=3)
    assert small != large
    with Image.open(small) as image:
        assert image.size == (64, 64)


def test_the_prompt_steers_the_palette(tmp_path: Path) -> None:
    """'A calm ocean' should not come out orange."""
    import numpy as np

    from genvai.adapters.procedural_image import _palette_for

    rng = np.random.default_rng(0)
    assert _palette_for("a calm ocean at dusk", rng) == "cool"
    assert _palette_for("a warm coffee in autumn", rng) == "warm"
    assert _palette_for("stars at night", rng) == "night"


def test_an_unrecognised_prompt_still_gets_a_palette(tmp_path: Path) -> None:
    import numpy as np

    from genvai.adapters.procedural_image import _palette_for

    assert _palette_for("zzzz qqqq", np.random.default_rng(0)) in PALETTES


# --------------------------------------------------------------------- resolving


def test_resolving_attaches_real_assets(tmp_path: Path) -> None:
    store = FilesystemStore(tmp_path / "projects")
    store.create("idea", "idea")
    timeline = plan("One. Two.", None, target_duration=6.0, canvas=Canvas(width=64, height=64))

    resolved = resolve_visuals(timeline, "idea", ProceduralImageProvider(tmp_path / "cache"), store)
    assert resolved.unresolved_visuals == ()
    for scene in resolved.scenes:
        asset_id = scene.visual.asset_id  # type: ignore[union-attr]
        assert asset_id in resolved.assets
        assert store.asset_path("idea", asset_id).exists()


def test_resolving_records_what_made_the_image(tmp_path: Path) -> None:
    store = FilesystemStore(tmp_path / "projects")
    store.create("idea", "idea")
    timeline = plan("One.", None, target_duration=3.0, canvas=Canvas(width=64, height=64))

    resolved = resolve_visuals(timeline, "idea", ProceduralImageProvider(tmp_path / "cache"), store)
    provenance = next(iter(resolved.assets.values())).provenance
    assert provenance.provider == "procedural"
    assert provenance.prompt


def test_resolving_twice_changes_nothing(tmp_path: Path) -> None:
    store = FilesystemStore(tmp_path / "projects")
    store.create("idea", "idea")
    provider = ProceduralImageProvider(tmp_path / "cache")
    timeline = plan("One. Two.", None, target_duration=6.0, canvas=Canvas(width=64, height=64))

    once = resolve_visuals(timeline, "idea", provider, store)
    twice = resolve_visuals(once, "idea", provider, store)
    assert twice == once


# ------------------------------------------------------------------ choosing one


def test_procedural_is_chosen_when_asked(tmp_path: Path) -> None:
    provider = image_provider(ImageSettings(provider="procedural"), tmp_path)
    assert provider.name == "procedural"


def test_diffusion_degrades_to_procedural_when_absent(tmp_path: Path) -> None:
    """The common case on a real machine, and not an error."""
    notes: list[str] = []
    provider = image_provider(ImageSettings(provider="diffusers"), tmp_path, on_note=notes.append)
    assert provider.name == "procedural"
    assert notes and "uv sync --extra image" in notes[0]


def test_choosing_a_provider_never_fails(tmp_path: Path) -> None:
    for name in ("diffusers", "procedural", "null"):
        assert image_provider(ImageSettings(provider=name), tmp_path) is not None  # type: ignore[arg-type]
