"""What a promotional poster says, once something has read it.

Kept apart from the timeline for the same reason `media.py` is: this is an *observation*
of artwork, not a decision about a cut. Read once, shown to the user for correction, then
turned into beats.

Prices are strings, not numbers. A poster writes "150", "₹ 40" and "from ₹500" in the
same breath, and normalising those into a float would throw away the wording the reel
should actually show - then have to invent it back.
"""

import re

from pydantic import Field

from genvai.timeline import Frozen

MAX_SERVICE = 28
MAX_NOTE = 40
MAX_OFFERS = 12

_DIGIT = re.compile(r"\d")


class Offer(Frozen):
    """One line of a price list."""

    service: str = Field(description="What it is. 'EYEBROW', 'GEL POLISH'.")
    price: str = Field(description="Exactly as written. '150', '₹ 40', 'from ₹500'.")
    note: str = Field(default="", description="The small print. 'upper lips free'.")

    @property
    def looks_like_a_price(self) -> bool:
        """Whether the price field plausibly contains one.

        A model asked to read a poster will occasionally return a heading or a stray
        phrase here. Anything with no digit in it is not a price, whatever else it is.
        """
        return bool(_DIGIT.search(self.price))

    def tidy(self) -> "Offer":
        """Trim to what fits on screen, without changing the wording."""
        return self.model_copy(
            update={
                "service": self.service.strip()[:MAX_SERVICE],
                "price": self.price.strip(),
                "note": self.note.strip()[:MAX_NOTE],
            }
        )


class Brief(Frozen):
    """Everything a reel needs, as read from the artwork.

    The user sees this before anything is built and corrects it if the reading is wrong -
    a wrong price is the one failure a client notices immediately.
    """

    business: str = Field(default="", description="Whose offer it is.")
    occasion: str = Field(default="", description="'Raksha Bandhan offer'.")
    tagline: str = Field(default="", description="One line of the poster's own copy.")
    phone: str = Field(default="", description="How to book.")
    offers: tuple[Offer, ...] = ()

    def cleaned(self) -> "Brief":
        """Drop what cannot be used and tidy the rest.

        Rows without a digit in the price are discarded rather than rendered: a beat
        reading "OTHER SERVICES / OTHER SERVICES" is worse than one beat fewer.
        """
        kept: list[Offer] = []
        seen: set[str] = set()
        for offer in self.offers:
            tidied = offer.tidy()
            key = tidied.service.casefold()
            if not tidied.service or not tidied.looks_like_a_price or key in seen:
                continue
            seen.add(key)
            kept.append(tidied)
        return self.model_copy(update={"offers": tuple(kept[:MAX_OFFERS])})

    def merge(self, other: "Brief") -> "Brief":
        """Fold a second poster's reading into this one.

        Two posters for one business share a phone number and an occasion; whichever
        reading found them first wins, and the offers simply accumulate.
        """
        return Brief(
            business=self.business or other.business,
            occasion=self.occasion or other.occasion,
            tagline=self.tagline or other.tagline,
            phone=self.phone or other.phone,
            offers=(*self.offers, *other.offers),
        ).cleaned()

    @property
    def is_usable(self) -> bool:
        return bool(self.offers)
