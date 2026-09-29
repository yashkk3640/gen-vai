"""Reading a poster: the brief, and the adapter that fills it.

The adapter is tested against a stubbed server. What matters is not that a model reads
correctly - it often will not - but that a bad reading is caught before it reaches a
render, which is entirely this code's job.
"""

import json
from pathlib import Path
from typing import Any

import httpx
import pytest
from PIL import Image

from genvai.adapters.ollama_vision import OllamaVision, _encode, _schema
from genvai.config import LLMSettings
from genvai.errors import ProviderUnavailable
from genvai.promo import Brief, Offer


def _offer(service: str = "EYEBROW", price: str = "₹ 40", note: str = "") -> Offer:
    return Offer(service=service, price=price, note=note)


# -------------------------------------------------------------------------- offers


def test_a_price_needs_a_digit_in_it() -> None:
    """A model asked to read a poster returns headings here surprisingly often."""
    assert _offer(price="₹ 40").looks_like_a_price
    assert _offer(price="150").looks_like_a_price
    assert not _offer(price="OTHER SERVICES").looks_like_a_price
    assert not _offer(price="").looks_like_a_price


def test_the_wording_of_a_price_survives() -> None:
    """'from 500' and '500' say different things; normalising loses that."""
    assert _offer(price="from ₹500").tidy().price == "from ₹500"


def test_tidying_trims_without_rewriting() -> None:
    tidied = Offer(service="  EYEBROW  ", price=" 40 ", note="  upper lips free ").tidy()
    assert (tidied.service, tidied.price, tidied.note) == ("EYEBROW", "40", "upper lips free")


def test_an_overlong_service_is_cut_to_fit() -> None:
    assert len(_offer(service="x" * 80).tidy().service) <= 28


# --------------------------------------------------------------------------- brief


def test_rows_without_a_price_are_dropped() -> None:
    """A beat reading 'OTHER SERVICES / OTHER SERVICES' is worse than one beat fewer."""
    brief = Brief(offers=(_offer(), _offer(service="OTHER SERVICES", price="see below")))
    assert [o.service for o in brief.cleaned().offers] == ["EYEBROW"]


def test_the_same_service_is_not_listed_twice() -> None:
    brief = Brief(offers=(_offer(), _offer(service="eyebrow", price="40")))
    assert len(brief.cleaned().offers) == 1


def test_an_empty_brief_is_not_usable() -> None:
    assert not Brief().is_usable
    assert Brief(offers=(_offer(),)).cleaned().is_usable


def test_two_posters_merge_into_one_brief() -> None:
    """Two posters for one business share a phone and an occasion."""
    skin = Brief(business="GK", phone="7043641428", offers=(_offer(),))
    nails = Brief(occasion="Raksha Bandhan", offers=(_offer(service="GEL POLISH", price="150"),))
    merged = skin.merge(nails)

    assert merged.business == "GK"
    assert merged.phone == "7043641428"
    assert merged.occasion == "Raksha Bandhan"
    assert [o.service for o in merged.offers] == ["EYEBROW", "GEL POLISH"]


def test_merging_keeps_the_first_reading_of_a_field() -> None:
    first = Brief(phone="111")
    second = Brief(phone="222")
    assert first.merge(second).phone == "111"


def test_merging_cleans_the_result() -> None:
    a = Brief(offers=(_offer(),))
    b = Brief(offers=(_offer(), _offer(service="HEADING", price="none")))
    assert len(a.merge(b).offers) == 1


def test_a_brief_roundtrips() -> None:
    brief = Brief(business="GK", offers=(_offer(note="upper lips free"),))
    assert Brief.model_validate_json(brief.model_dump_json()) == brief


# ------------------------------------------------------------------------ adapter


class _Reply:
    def __init__(self, content: str) -> None:
        self._content = content

    def raise_for_status(self) -> None: ...

    def json(self) -> dict:
        return {"message": {"content": self._content}}


@pytest.fixture
def poster(tmp_path: Path) -> Path:
    path = tmp_path / "poster.png"
    Image.new("RGB", (1024, 1536), (250, 235, 235)).save(path)
    return path


@pytest.fixture
def vision() -> OllamaVision:
    return OllamaVision(LLMSettings(), model="test-vision")


def _serve(monkeypatch: pytest.MonkeyPatch, content: str) -> list[dict[str, Any]]:
    sent: list[dict[str, Any]] = []

    def fake_post(url: str, json: dict, timeout: float) -> _Reply:  # noqa: A002
        sent.append(json)
        return _Reply(content)

    monkeypatch.setattr(httpx, "post", fake_post)
    return sent


def test_a_good_reading_becomes_a_brief(
    vision: OllamaVision, poster: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _serve(
        monkeypatch,
        json.dumps(
            {
                "business": "Gracy Khatri",
                "occasion": "Raksha Bandhan",
                "phone": "7043641428",
                "tagline": "",
                "offers": [{"service": "EYEBROW", "price": "40", "note": "upper lips free"}],
            }
        ),
    )
    brief = vision.read(poster)
    assert brief.business == "Gracy Khatri"
    assert brief.offers[0].note == "upper lips free"


def test_the_image_is_actually_sent(
    vision: OllamaVision, poster: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    sent = _serve(monkeypatch, '{"offers": []}')
    vision.read(poster)
    assert sent[0]["messages"][-1]["images"], "the poster has to reach the model"


def test_the_schema_is_sent_so_decoding_is_constrained(
    vision: OllamaVision, poster: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    sent = _serve(monkeypatch, '{"offers": []}')
    vision.read(poster)
    assert "offers" in sent[0]["format"]["properties"]


def test_invented_rows_are_cleaned_out(
    vision: OllamaVision, poster: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The model returning a heading as a service is the common failure."""
    _serve(
        monkeypatch,
        json.dumps(
            {
                "offers": [
                    {"service": "EYEBROW", "price": "40"},
                    {"service": "OTHER SERVICES", "price": "OTHER SERVICES"},
                ]
            }
        ),
    )
    assert [o.service for o in vision.read(poster).offers] == ["EYEBROW"]


def test_an_unreadable_answer_is_an_empty_reading_not_a_crash(
    vision: OllamaVision, poster: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Losing the automation is a smaller failure than losing the command."""
    _serve(monkeypatch, "I am not able to read that, sorry.")
    assert vision.read(poster) == Brief()


def test_a_scratchpad_is_stripped(
    vision: OllamaVision, poster: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _serve(monkeypatch, '<think>hmm</think>{"offers": [{"service": "WAX", "price": "250"}]}')
    assert vision.read(poster).offers[0].service == "WAX"


def test_an_unreachable_server_is_a_typed_error(
    vision: OllamaVision, poster: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def refuse(url: str, json: dict, timeout: float) -> _Reply:  # noqa: A002
        raise httpx.ConnectError("refused")

    monkeypatch.setattr(httpx, "post", refuse)
    with pytest.raises(ProviderUnavailable, match="pulled"):
        vision.read(poster)


def test_availability_is_false_without_the_model(
    vision: OllamaVision, monkeypatch: pytest.MonkeyPatch
) -> None:
    class _Tags:
        def raise_for_status(self) -> None: ...

        def json(self) -> dict:
            return {"models": [{"name": "llama3.2:latest"}]}

    monkeypatch.setattr(httpx, "get", lambda url, timeout: _Tags())
    assert vision.is_available() is False


def test_the_poster_is_downscaled_before_sending(poster: Path) -> None:
    """A 3 MB PNG becomes a 4 MB base64 string the model shrinks again on arrival."""
    encoded = _encode(poster)
    assert len(encoded) < 900_000


def test_the_schema_has_no_unresolved_references() -> None:
    """Ollama does not follow $ref, so a nested model would constrain nothing."""
    assert "$ref" not in json.dumps(_schema())


def test_bullets_come_off_a_label() -> None:
    """OCR keeps the poster's bullet dots; on screen they read as a rendering fault."""
    for raw in ("\u00b7EYEBROW", "\u2022 EYEBROW", "\uff1aEYEBROW", "EYEBROW ~"):
        assert Offer(service=raw, price="40").tidy().service == "EYEBROW"


def test_a_trailing_aside_becomes_the_note() -> None:
    tidied = Offer(service="EYEBROW (Uparlips Free)", price="40").tidy()
    assert tidied.service == "EYEBROW"
    assert tidied.note == "Uparlips Free"


def test_an_unclosed_aside_still_moves() -> None:
    tidied = Offer(service="MANICURE-PEDICURE (Half hand, leg.", price="600").tidy()
    assert tidied.service == "MANICURE-PEDICURE"
    assert tidied.note.startswith("Half hand")


def test_an_existing_note_is_not_overwritten() -> None:
    tidied = Offer(service="EYEBROW (Uparlips Free)", price="40", note="threading").tidy()
    assert tidied.note == "threading"


def test_a_label_that_is_only_an_aside_is_kept() -> None:
    assert Offer(service="(FREE)", price="0").tidy().service == "(FREE)"


def test_a_middle_dot_inside_a_label_stays() -> None:
    assert Offer(service="WAX \u00b7 Sugar", price="250").tidy().service == "WAX \u00b7 Sugar"


def test_a_number_run_into_a_word_is_spaced() -> None:
    assert Offer(service="1FINGER ART", price="20").tidy().service == "1 FINGER ART"
    assert Offer(service="D-TEN", price="200").tidy().service == "D-TEN"


def test_words_run_together_in_small_print_are_split() -> None:
    tidied = Offer(service="WAX", price="250", note="Fullhand,halfleg,underarms").tidy()
    assert tidied.note == "Full hand, half leg, underarms"


def test_a_clients_note_is_attached_by_part_of_the_service_name() -> None:
    brief = Brief(
        offers=(
            Offer(service="DOTING ART", price="₹30"),
            Offer(service="1 FINGER ART", price="₹20"),
            Offer(service="TOE CAT EYE", price="₹500", note="Original Nail"),
        )
    ).with_notes({"art": "per finger"})
    assert [o.note for o in brief.offers] == ["per finger", "per finger", "Original Nail"]
