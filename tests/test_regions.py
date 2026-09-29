"""Cutting backdrops out of a poster: erasing its words, and steering clear of the rest."""

import numpy as np

from genvai.regions import ERASE_MAX_HEIGHT, erase_text, photo_regions


def _page_with_word(ground: int = 240) -> np.ndarray:
    page = np.full((400, 400, 3), ground, dtype=np.uint8)
    page[100:112, 100:180] = 30  # a dark word on flat ground
    return page


WORD = (100 / 400, 100 / 400, 80 / 400, 12 / 400)


def test_a_word_on_flat_ground_is_painted_out() -> None:
    clean, stubborn = erase_text(_page_with_word(), (WORD,))
    assert int(clean[100:112, 100:180].min()) >= 235
    assert stubborn == ()


def test_the_rest_of_the_picture_is_untouched() -> None:
    page = _page_with_word()
    page[300:350, 300:350] = (200, 30, 60)
    clean, _ = erase_text(page, (WORD,))
    assert np.array_equal(clean[250:, 250:], page[250:, 250:])


def test_a_word_over_a_picture_is_left_and_reported() -> None:
    page = np.random.default_rng(2).integers(0, 255, (400, 400, 3), dtype=np.uint8)
    clean, stubborn = erase_text(page, (WORD,))
    assert np.array_equal(clean, page)
    assert stubborn == (WORD,)


def test_display_type_is_left_alone() -> None:
    """A huge script title sweeps up whatever it overlaps; erasing it cuts holes."""
    tall = (0.2, 0.2, 0.5, ERASE_MAX_HEIGHT * 2)
    page = _page_with_word()
    clean, stubborn = erase_text(page, (tall,))
    assert np.array_equal(clean, page)
    assert stubborn == ()


def test_nothing_to_erase_returns_the_picture() -> None:
    page = _page_with_word()
    clean, stubborn = erase_text(page, ())
    assert clean is page and stubborn == ()


def _two_photos() -> np.ndarray:
    rng = np.random.default_rng(3)
    page = np.full((700, 1000, 3), 250, dtype=np.uint8)
    for left in (80, 620):
        page[200:500, left : left + 300] = rng.integers(0, 255, (300, 300, 3), dtype=np.uint8)
    return page


def test_words_that_could_not_be_erased_push_a_window_away() -> None:
    banner = (0.62, 0.28, 0.3, 0.06)
    for x, y, w, h in photo_regions(_two_photos(), count=2, text=(banner,)):
        across = min(x + w, 0.92) - max(x, 0.62)
        down = min(y + h, 0.34) - max(y, 0.28)
        assert across <= 0 or down <= 0, "no chosen window holds the banner"
