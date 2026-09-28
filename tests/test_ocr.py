"""Pairing recognised text into a price list.

Hand-built boxes rather than real OCR, so these describe *layouts* - a two-column price
list, a stacked list of variants, a wrapped label, a grid of small options - and run
without the engine installed.

The layouts here are taken from real posters, including the two that disagreed with each
other and forced the pairing rules to get more precise.
"""

import pytest

from genvai.ocr import Box, find_phone, join_wrapped, pair_offers, read_brief


def _box(text: str, x: float, y: float, *, w: float = 160, h: float = 26, conf: float = 0.98):
    return Box(text=text, confidence=conf, x=x, y=y, width=w, height=h)


def _services(boxes: list[Box]) -> list[tuple[str, str]]:
    return [(o.service, o.price) for o in pair_offers(join_wrapped(boxes))]


# ------------------------------------------------------------- the ordinary layout


def test_a_two_column_price_list() -> None:
    boxes = [
        _box("CLEAN UP", 230, 770),
        _box("250", 594, 772, w=60),
        _box("BLEACH", 230, 855),
        _box("250", 594, 857, w=60),
    ]
    assert _services(boxes) == [("CLEAN UP", "₹250"), ("BLEACH", "₹250")]


def test_a_price_with_nothing_beside_it_is_dropped() -> None:
    """A decorative number is not an offer, and inventing a service for it would put a
    line in the reel that is on no poster."""
    assert _services([_box("2026", 500, 100, w=60)]) == []


def test_a_column_heading_is_not_a_service() -> None:
    boxes = [_box("OTHER SERVICES", 700, 500), _box("250", 594, 505, w=60)]
    assert _services(boxes) == []


def test_the_bigger_of_two_labels_on_a_row_wins() -> None:
    """A row carries the service and its small print; the small print is set smaller
    precisely because it is not the name of the thing."""
    boxes = [
        _box("Sugar", 246, 630, h=26),
        _box("full hand, half leg, underarms", 264, 655, h=16),
        _box("250", 593, 643, w=60),
    ]
    assert _services(boxes)[0][0] == "Sugar"


# ------------------------------------------------------------------ wrapped labels


def test_a_label_wrapped_onto_two_lines_is_rejoined() -> None:
    """From the nail poster: the price pill sits level with both lines, so leaving them
    apart names the offer after whichever line happens to be nearer."""
    boxes = [
        _box("GEL POLISH", 180, 620, h=30),
        _box("HAND TOE", 180, 655, h=30),
        _box("300", 520, 637, w=60),
    ]
    assert _services(boxes) == [("GEL POLISH HAND TOE", "₹300")]


def test_a_stacked_list_is_not_treated_as_one_wrapped_label() -> None:
    """From the skin poster: three facials, same size, left aligned, closely spaced -
    identical to a wrapped label by geometry. Each having its own price is the signal."""
    boxes = [
        _box("Fruit Facial", 259, 1160, h=22),
        _box("500", 605, 1160, w=60),
        _box("Charcoal Facial", 259, 1196, h=22),
        _box("600", 605, 1196, w=60),
        _box("Diamond Facial", 259, 1232, h=22),
        _box("750", 605, 1232, w=60),
    ]
    assert _services(boxes) == [
        ("Fruit Facial", "₹500"),
        ("Charcoal Facial", "₹600"),
        ("Diamond Facial", "₹750"),
    ]


def test_lines_of_different_sizes_are_not_joined() -> None:
    boxes = [_box("WAX", 230, 580, h=30), _box("full hand only", 230, 612, h=14)]
    joined = join_wrapped(boxes)
    assert any(b.cleaned == "WAX" for b in joined)


def test_lines_far_apart_are_not_joined() -> None:
    boxes = [_box("CLEAN UP", 230, 770, h=26), _box("BLEACH", 230, 900, h=26)]
    assert len(join_wrapped(boxes)) == 2


# ----------------------------------------------------------------- grid layouts


def test_a_price_under_its_label_is_still_paired() -> None:
    """Nail art add-ons: a grid where the price sits in a pill beneath its label."""
    boxes = [
        _box("MARBLE ART", 400, 1050, w=120, h=20),
        _box("40", 430, 1110, w=50, h=26),
    ]
    assert _services(boxes) == [("MARBLE ART", "₹40")]


# ------------------------------------------------------------------------ prices


@pytest.mark.parametrize("written", ["40", "₹ 40", "?40", "Rs 40", "40.00"])
def test_a_price_is_recognised_however_the_symbol_came_out(written: str) -> None:
    """OCR renders the rupee mark as whatever it can manage; the digits are what matter."""
    boxes = [_box("EYEBROW", 230, 490), _box(written, 594, 492, w=70)]
    assert _services(boxes) == [("EYEBROW", "₹40")]


def test_a_phone_number_is_never_read_as_a_price() -> None:
    boxes = [_box("CONTACT", 230, 1500), _box("7043641428", 594, 1502, w=200)]
    assert _services(boxes) == []


def test_the_phone_number_is_found() -> None:
    boxes = [_box("Gracy Khatri", 300, 1490), _box("7043641428", 560, 1492, w=200)]
    assert find_phone(boxes) == "7043641428"


def test_a_spaced_phone_number_is_found() -> None:
    assert find_phone([_box("70436 41428", 560, 1492, w=200)]) == "7043641428"


def test_no_phone_is_an_empty_string_not_a_guess() -> None:
    assert find_phone([_box("CLEAN UP", 230, 770)]) == ""


# ------------------------------------------------------------------------- brief


def test_a_brief_is_assembled_from_boxes() -> None:
    boxes = [
        _box("EYEBROW", 230, 490),
        _box("40", 594, 492, w=60),
        _box("Gracy Khatri 7043641428", 300, 1490, w=300),
    ]
    brief = read_brief(boxes, occasion="Raksha Bandhan")
    assert brief.occasion == "Raksha Bandhan"
    assert brief.phone == "7043641428"
    assert brief.offers[0].service == "EYEBROW"


def test_low_confidence_text_is_ignored() -> None:
    """A guessed character in a price is the one thing worth refusing outright."""
    boxes = [_box("EYEBROW", 230, 490), _box("40", 594, 492, w=60, conf=0.2)]
    assert read_brief(boxes).offers == ()


def test_an_empty_reading_yields_an_empty_brief() -> None:
    assert not read_brief([]).is_usable
