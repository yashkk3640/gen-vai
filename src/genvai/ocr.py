"""Turning recognised text boxes into a price list.

OCR returns words and where they sit. Pairing a service with its price is then geometry,
not language: on a price list the service is on the left of a row and the price on the
right, and the row is the thing that binds them.

Doing it by position rather than by asking a model is the point. A model asked to pair
"EYEBROW" with a number will always produce one; this either finds a price on the same
row or reports that it did not.

Pure - takes boxes, returns offers.
"""

import re

from pydantic import Field

from genvai.promo import Brief, Offer
from genvai.timeline import Frozen

MIN_CONFIDENCE = 0.55
"""Below this the characters are a guess, and a guessed price is the thing to avoid."""

ROW_TOLERANCE = 0.55
"""How far apart, as a fraction of the taller box, two boxes may sit and still be one
row. Price lists set the price a little above or below its label."""

NOTE_GAP = 1.9
"""How far below a service its small print may sit, in multiples of the label's height."""

SPANNING = 1.4
"""How much taller than a label a price box must be to count as the group's price.

A pill set across a two-line label is drawn tall enough to span both; a price on its
own row is set at about the height of the line it belongs to.
"""

PRICE = re.compile(r"^[^\d]{0,4}(\d{2,5})(?:\.00?)?[^\d]{0,3}$")
"""A price: two to five digits with at most a little decoration. Bounded at five so a
ten-digit phone number is never mistaken for one."""

PHONE = re.compile(r"(?<!\d)(\d[\d\s-]{8,13}\d)(?!\d)")

CURRENCY_NOISE = str.maketrans({"?": "", "¥": "", "₹": "", "$": ""})
"""OCR renders the rupee mark as whatever it can. The digits are what matter; the symbol
is put back in one consistent form."""

HEADINGS = {
    "other services",
    "otherservices",
    "services",
    "offer",
    "price",
    "rate",
    "rates",
    "menu",
    "contact us",
    "for booking",
    "book now",
}


class Box(Frozen):
    """One recognised run of text and where it sits, in pixels."""

    text: str
    confidence: float
    x: float
    y: float
    width: float
    height: float

    @property
    def right(self) -> float:
        return self.x + self.width

    @property
    def middle(self) -> float:
        return self.y + self.height / 2

    @property
    def cleaned(self) -> str:
        return " ".join(self.text.split()).strip(" .:~-•")


def read_brief(boxes: list[Box], *, occasion: str = "") -> Brief:
    """Assemble a brief from recognised text.

    Everything here is positional. The only judgement is which rows look like prices,
    and that is a regular expression over digits rather than an opinion.
    """
    usable = [b for b in boxes if b.confidence >= MIN_CONFIDENCE and b.cleaned]
    offers = pair_offers(join_wrapped(usable))
    return Brief(
        occasion=occasion,
        phone=find_phone(usable),
        offers=tuple(offers),
    ).cleaned()


def join_wrapped(boxes: list[Box]) -> list[Box]:
    """Rejoin a label that the layout wrapped onto two lines.

    "GEL POLISH" above "HAND TOE" is one service, and OCR reports it as two. Left with
    them apart, a price beside the pair matches whichever line happens to sit nearer its
    middle, so half the list comes out named after its second line.

    Only lines of matching size, left edge and spacing are joined - which is what a
    wrapped label looks like and what a label followed by its small print does not.
    """
    remaining = sorted(boxes, key=lambda b: (b.y, b.x))
    priced = [b for b in remaining if _amount(b)]
    labels = [b for b in remaining if not _amount(b)]
    joined: list[Box] = []
    used: set[int] = set()

    for index, box in enumerate(remaining):
        if index in used or _amount(box):
            continue
        merged = box
        for other_index in range(index + 1, len(remaining)):
            other = remaining[other_index]
            if other_index in used or _amount(other):
                continue
            if not _is_continuation(merged, other) or _has_own_price(other, priced, labels):
                continue
            merged = merged.model_copy(
                update={
                    "text": f"{merged.cleaned} {other.cleaned}",
                    "height": other.y + other.height - merged.y,
                    "width": max(merged.width, other.width),
                }
            )
            used.add(other_index)
        joined.append(merged)

    return joined + [b for b in remaining if _amount(b)]


def _has_own_price(box: Box, priced: list[Box], labels: list[Box]) -> bool:
    """Whether this line carries a price that is *its* price rather than a neighbour's.

    The signal that separates a stacked list from a wrapped label. "Fruit Facial" above
    "Charcoal Facial" looks exactly like one label over two lines; geometry alone cannot
    tell them apart, but each has its own price beside it and a wrapped label's second
    line does not.

    "Beside it" has to mean nearest, not merely level. A price set in a pill spanning a
    two-line label sits level with both lines, so testing for any price on the row would
    call every wrapped label a list.

    And a price noticeably taller than the line is a price *for the group* - a pill
    centred across a wrapped label - not that line's own. Without this the answer turns
    on which of two lines the pill's centre happens to fall nearer, which is a few pixels
    either way and flips between otherwise identical rows.
    """
    for price in priced:
        if price.x <= box.right or not _same_row(price, box):
            continue
        if price.height > box.height * SPANNING:
            continue
        rivals = [
            other
            for other in labels
            if other is not box and other.right < price.x and _same_row(price, other)
        ]
        nearer = min((abs(other.middle - price.middle) for other in rivals), default=float("inf"))
        if abs(box.middle - price.middle) <= nearer:
            return True
    return False


def _is_continuation(first: Box, second: Box) -> bool:
    """Whether `second` is the next line of the same label."""
    gap = second.y - (first.y + first.height)
    return (
        0 <= gap <= first.height * 0.75
        and abs(second.x - first.x) <= first.height * 0.9
        and 0.7 <= second.height / max(first.height, 1e-6) <= 1.4
    )


def pair_offers(boxes: list[Box]) -> list[Offer]:
    """Match each price to the service written on its row.

    A price with no label to its left is dropped rather than guessed at - it is usually
    a decorative number or part of a heading, and inventing a service for it would put a
    line in the reel that is on no poster.
    """
    prices = [(b, _amount(b)) for b in boxes]
    prices = [(b, amount) for b, amount in prices if amount]

    offers: list[Offer] = []
    for box, amount in prices:
        label = _label_for(box, boxes)
        if label is None:
            continue
        offers.append(
            Offer(
                service=label.cleaned,
                price=f"₹{amount}",
                note=_note_below(label, boxes),
            )
        )
    return offers


def find_phone(boxes: list[Box]) -> str:
    """The longest run of digits that looks like a number to ring."""
    for box in sorted(boxes, key=lambda b: -b.y):
        match = PHONE.search(box.cleaned)
        if match:
            digits = re.sub(r"\D", "", match.group(1))
            if 9 <= len(digits) <= 13:
                return digits
    return ""


def _amount(box: Box) -> str:
    """The digits of a price, or empty when the text is not one."""
    text = box.cleaned.translate(CURRENCY_NOISE).strip()
    match = PRICE.match(text)
    return match.group(1) if match else ""


def _label_for(price: Box, boxes: list[Box]) -> Box | None:
    """The service a price belongs to.

    Two layouts, tried in order. Most price lists put the service to the left on the same
    row. A grid of small options - nail art add-ons, say - puts the price in a pill under
    its label instead, so a row with nothing to its left looks directly above.
    """
    return _label_left_of(price, boxes) or _label_above(price, boxes)


def _label_left_of(price: Box, boxes: list[Box]) -> Box | None:
    """The service written to the left of a price, on its row.

    Among several, the tallest wins rather than the nearest. A row often carries both the
    service and its small print - "Sugar" above "full hand, half leg, underarms" - and the
    small print is set smaller precisely because it is not the name of the thing.
    """
    candidates = [
        b
        for b in boxes
        if b is not price
        and b.right <= price.x + price.width * 0.4
        and _same_row(b, price)
        and _is_label(b)
    ]
    if not candidates:
        return None
    tallest = max(b.height for b in candidates)
    return max((b for b in candidates if b.height >= tallest * 0.88), key=lambda b: b.right)


def _label_above(price: Box, boxes: list[Box]) -> Box | None:
    """The service written directly above a price, for grid layouts."""
    candidates = [
        b
        for b in boxes
        if b is not price
        and 0 < price.y - b.y <= price.height * 4.0
        and abs(b.x + b.width / 2 - (price.x + price.width / 2)) < max(b.width, price.width)
        and _is_label(b)
    ]
    return max(candidates, key=lambda b: b.y) if candidates else None


def _is_label(box: Box) -> bool:
    return not _amount(box) and box.cleaned.casefold() not in HEADINGS and len(box.cleaned) > 2


def _note_below(label: Box, boxes: list[Box]) -> str:
    """Small print sitting just under a service, such as what is included.

    Two things disqualify a line. Being the same size or larger means it is the next
    service, not this one's footnote. Carrying a price of its own means the same thing
    even more definitely - and that case is the one that bit: a service whose own note
    says the name of the service below it puts that name on screen under the wrong price.
    """
    priced = [b for b in boxes if _amount(b)]
    labels = [b for b in boxes if not _amount(b)]
    below = [
        b
        for b in boxes
        if b is not label
        and label.y < b.y <= label.y + label.height * NOTE_GAP
        # Roughly the same left edge. Small print sits under its service; a centred
        # section heading further along the row is not small print at all.
        and abs(b.x - label.x) < label.width * 0.55
        and b.height < label.height * 0.95
        and not _amount(b)
        and not _has_own_price(b, priced, labels)
        and not _looks_like_a_heading(b)
    ]
    if not below:
        return ""
    return min(below, key=lambda b: b.y).cleaned


def _looks_like_a_heading(box: Box) -> bool:
    """Whether a line is a section title rather than small print.

    Small print is set in sentence case - "(Uparlips Free)", "full hand, half leg". A
    section heading is set in capitals. Using the casing avoids maintaining a list of
    every heading a salon might print.
    """
    text = box.cleaned
    letters = [c for c in text if c.isalpha()]
    return bool(letters) and all(c.isupper() for c in letters) and len(text) < 26


def _same_row(a: Box, b: Box) -> bool:
    return abs(a.middle - b.middle) <= max(a.height, b.height) * ROW_TOLERANCE


class Recognised(Frozen):
    """What an OCR engine returned for one image."""

    boxes: tuple[Box, ...] = Field(default=())

    @property
    def text(self) -> str:
        return "\n".join(b.cleaned for b in self.boxes if b.cleaned)
