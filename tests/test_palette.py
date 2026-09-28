"""Reading a brand's colours off its artwork."""

import numpy as np
import pytest

from genvai.palette import Brand, brand, dominant


def _blocks(*colours: tuple[int, int, int], size: int = 120) -> np.ndarray:
    """An image made of equal horizontal bands of the given colours."""
    band = size // len(colours)
    out = np.zeros((size, size, 3), dtype=np.uint8)
    for i, colour in enumerate(colours):
        out[i * band : (i + 1) * band] = colour
    return out


MAROON = (140, 16, 52)
GOLD = (201, 162, 39)
PINK = (249, 220, 226)


def _near(a: tuple[int, int, int], b: tuple[int, int, int], tol: int = 40) -> bool:
    return all(abs(x - y) <= tol for x, y in zip(a, b, strict=True))


# ------------------------------------------------------------------------ dominant


def test_finds_the_colours_that_are_there() -> None:
    found = dominant(_blocks(MAROON, GOLD, PINK))
    assert any(_near(c, MAROON) for c in found)
    assert any(_near(c, GOLD) for c in found)
    assert any(_near(c, PINK) for c in found)


def test_the_biggest_area_comes_first() -> None:
    image = np.zeros((120, 120, 3), dtype=np.uint8)
    image[:] = PINK
    image[:20] = MAROON
    assert _near(dominant(image)[0], PINK)


def test_it_does_not_invent_colours_between_the_real_ones() -> None:
    """Averaging red and white gives pink, which may be nowhere on the poster."""
    found = dominant(_blocks((220, 20, 20), (255, 255, 255)))
    assert not any(120 < c[1] < 200 and 120 < c[2] < 200 for c in found[:2])


def test_a_flat_image_yields_one_colour() -> None:
    found = dominant(np.full((60, 60, 3), 90, dtype=np.uint8))
    assert _near(found[0], (90, 90, 90), tol=6)


def test_an_empty_image_is_survivable() -> None:
    assert dominant(np.zeros((0, 0, 3), dtype=np.uint8)) == []


def test_the_same_picture_always_gives_the_same_palette() -> None:
    """A reel that re-renders in different colours is not reproducible."""
    image = _blocks(MAROON, GOLD, PINK)
    assert dominant(image) == dominant(image)


# --------------------------------------------------------------------------- brand


def test_the_deep_colour_is_dark_enough_to_put_text_on() -> None:
    result = brand(_blocks(MAROON, GOLD, PINK))
    assert sum(result.deep) / 3 < 90


def test_the_deep_colour_keeps_its_hue() -> None:
    """A neutral black would look borrowed from some other brand."""
    result = brand(_blocks(MAROON, GOLD, PINK))
    assert max(result.deep) - min(result.deep) > 8


def test_the_accent_is_the_saturated_one() -> None:
    result = brand(_blocks(MAROON, GOLD, PINK))
    high, low = max(result.accent), min(result.accent)
    assert (high - low) / max(high, 1) > 0.3


def test_the_light_colour_is_pale_enough_for_a_card() -> None:
    result = brand(_blocks(MAROON, GOLD, PINK))
    assert sum(result.light) / 3 > 190


def test_a_colourless_picture_still_yields_a_usable_brand() -> None:
    """Artwork with no hue at all must not produce an unusable palette."""
    result = brand(np.full((80, 80, 3), 128, dtype=np.uint8))
    assert isinstance(result, Brand)
    assert sum(result.deep) / 3 < sum(result.light) / 3


def test_an_empty_picture_falls_back() -> None:
    result = brand(np.zeros((0, 0, 3), dtype=np.uint8))
    assert sum(result.light) / 3 > sum(result.deep) / 3


def test_hex_is_usable_in_a_filtergraph() -> None:
    result = brand(_blocks(MAROON, GOLD, PINK))
    assert result.hex("accent").startswith("#")
    assert len(result.hex("deep")) == 7


def test_a_mid_tone_poster_is_nudged_not_trusted() -> None:
    """A scrim only works if it is actually dark."""
    result = brand(_blocks((150, 90, 110), (160, 100, 120)))
    assert sum(result.deep) / 3 < 80


def test_the_brand_roundtrips() -> None:
    result = brand(_blocks(MAROON, GOLD, PINK))
    assert Brand.model_validate_json(result.model_dump_json()) == result


@pytest.mark.parametrize(
    "artwork",
    [
        _blocks((20, 40, 90), (240, 240, 250)),
        _blocks((10, 80, 40), (200, 220, 190), (240, 200, 60)),
        _blocks((90, 20, 20), (250, 245, 235)),
    ],
)
def test_any_artwork_gives_a_dark_deep_and_a_pale_light(artwork: np.ndarray) -> None:
    result = brand(artwork)
    assert sum(result.deep) / 3 < sum(result.light) / 3 - 60
