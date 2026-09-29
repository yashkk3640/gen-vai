"""Faces weighted into saliency: backdrop choice, reframing, shot scoring.

The detector itself needs a real face to test and the client artwork is not in the
repository, so these exercise what is done *with* a face - the arithmetic - against
faces placed by hand.
"""

from pathlib import Path

import numpy as np

from genvai.adapters.yunet_faces import YunetFaces
from genvai.media import ClipQuality, Face, MediaItem
from genvai.pipeline.select import FACE_BONUS, item_score
from genvai.reframe import focus
from genvai.regions import photo_regions


def _two_photos() -> np.ndarray:
    """A white page with two equally colourful photos, left and right."""
    rng = np.random.default_rng(3)
    page = np.full((700, 1000, 3), 250, dtype=np.uint8)
    for left in (80, 620):
        page[200:500, left : left + 300] = rng.integers(0, 255, (300, 300, 3), dtype=np.uint8)
    return page


def _centre_x(rect: tuple[float, float, float, float]) -> float:
    return rect[0] + rect[2] / 2


def test_without_faces_either_photo_may_win() -> None:
    regions = photo_regions(_two_photos(), count=2)
    assert len(regions) == 2


def test_a_face_puts_its_photo_first() -> None:
    face = Face(rect=(0.72, 0.36, 0.08, 0.1), confidence=0.9)
    best = photo_regions(_two_photos(), count=2, faces=(face,))[0]
    assert _centre_x(best) > 0.5


def test_a_face_on_the_left_puts_the_left_photo_first() -> None:
    face = Face(rect=(0.18, 0.36, 0.08, 0.1), confidence=0.9)
    best = photo_regions(_two_photos(), count=2, faces=(face,))[0]
    assert _centre_x(best) < 0.5


def test_the_winning_window_holds_the_whole_face() -> None:
    face = Face(rect=(0.72, 0.36, 0.08, 0.1), confidence=0.9)
    x, y, w, h = photo_regions(_two_photos(), count=1, faces=(face,), aspect=9 / 16)[0]
    fx, fy, fw, fh = face.rect
    assert x <= fx and fx + fw <= x + w
    assert y <= fy and fy + fh <= y + h


def test_a_face_on_bare_paper_does_not_make_it_a_backdrop() -> None:
    page = np.full((700, 1000, 3), 250, dtype=np.uint8)
    face = Face(rect=(0.45, 0.45, 0.1, 0.1), confidence=0.95)
    assert photo_regions(page, count=2, faces=(face,)) == ()


def test_focus_is_drawn_toward_a_face() -> None:
    frame = np.full((180, 320), 40.0)
    frame[70:110, 40:80] = np.random.default_rng(1).uniform(0, 255, (40, 40))
    plain = focus(frame)
    pulled = focus(frame, (Face(rect=(0.8, 0.4, 0.1, 0.15), confidence=0.9),))
    assert plain[0] < 0.3
    assert pulled[0] > 0.6


def test_focus_follows_the_largest_face() -> None:
    frame = np.full((180, 320), 40.0)
    small = Face(rect=(0.1, 0.4, 0.05, 0.08), confidence=0.99)
    large = Face(rect=(0.75, 0.3, 0.2, 0.3), confidence=0.85)
    assert focus(frame, (small, large))[0] > 0.6


def _photo(face_area: float) -> MediaItem:
    return MediaItem(
        asset_id=f"a{face_area}",
        kind="image",
        source_name="p.jpg",
        width=1000,
        height=1000,
        quality=ClipQuality(
            sharpness=0.6, exposure=0.5, motion=0.0, shake=0.0, face_area=face_area
        ),
    )


def test_a_visible_face_adds_to_a_shots_score() -> None:
    gain = item_score(_photo(0.05)) - item_score(_photo(0.0))
    assert abs(gain - FACE_BONUS) < 1e-9


def test_a_tiny_face_adds_little() -> None:
    gain = item_score(_photo(0.004)) - item_score(_photo(0.0))
    assert 0 < gain < FACE_BONUS / 5


def test_a_missing_model_is_unavailable_and_finds_nothing(tmp_path: Path) -> None:
    detector = YunetFaces(tmp_path / "absent.onnx")
    assert not detector.is_available()
    assert detector.detect(np.zeros((100, 100, 3), dtype=np.uint8)) == ()


def test_a_face_rect_reports_area_and_centre() -> None:
    face = Face(rect=(0.2, 0.4, 0.2, 0.1), confidence=0.9)
    assert abs(face.area - 0.02) < 1e-9
    assert np.allclose(face.centre, (0.3, 0.45))
