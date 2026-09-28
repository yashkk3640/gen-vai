"""Reading a poster with a vision model served by Ollama.

The same server the text model runs on, so there is nothing new to install in the
project - only `ollama pull moondream` or similar. A small vision model fits 4 GB where
an 11B one does not; see docs/setup.md for what was measured.

What comes back is never trusted. It is cleaned, then shown to the user for confirmation
before anything is built, because a wrong price on a promo is the one mistake a client
notices immediately.
"""

import base64
import re
from pathlib import Path
from typing import Any

import httpx
from PIL import Image

from genvai.config import LLMSettings
from genvai.errors import ProviderUnavailable
from genvai.promo import Brief

MAX_EDGE = 1280
"""Longest edge sent to the model.

Small vision models downscale their input anyway, and a 4 MB base64 payload is slower to
move than it is worth. Large enough that a price list is still legible.
"""

SYSTEM_PROMPT = (
    "You read promotional posters and return exactly what is printed on them. "
    "Never invent a price, and never convert one - copy the text as written. "
    "Reply with JSON only."
)

PROMPT = (
    "Read this salon offer poster.\n"
    "List every service with its price, copying each price exactly as printed - keep "
    "'from', keep the currency mark or its absence.\n"
    "Put any small print for a service in its note, such as what is included or free.\n"
    "Also give the business name, the occasion, the phone number, and one short line of "
    "the poster's own wording as the tagline.\n"
    "Skip headings, section titles and decoration. Only rows that carry a price."
)

_THINK = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)
_JSON = re.compile(r"\{.*\}", re.DOTALL)


class OllamaVision:
    """Implements `VisionPort`."""

    def __init__(self, settings: LLMSettings, model: str = "moondream") -> None:
        self._settings = settings
        self._model = model

    @property
    def name(self) -> str:
        return f"ollama/{self._model}"

    def is_available(self) -> bool:
        """True when the server answers and the vision model is pulled."""
        try:
            response = httpx.get(f"{self._settings.base_url}/api/tags", timeout=3.0)
            response.raise_for_status()
        except Exception:  # noqa: BLE001 - availability checks never raise
            return False
        names = {m.get("name", "") for m in response.json().get("models", [])}
        return self._model in names or f"{self._model}:latest" in names

    def read(self, image: Path) -> Brief:
        """Read a poster into a brief.

        A model that answers with nothing usable returns an empty brief rather than
        raising. The caller can still build from a brief the user types, and losing the
        automation is a smaller failure than losing the command.
        """
        payload: dict[str, Any] = {
            "model": self._model,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": PROMPT, "images": [_encode(image)]},
            ],
            "stream": False,
            "think": False,
            "format": _schema(),
            "keep_alive": self._settings.keep_alive,
            "options": {"temperature": 0.0},
        }

        try:
            response = httpx.post(
                f"{self._settings.base_url}/api/chat",
                json=payload,
                timeout=max(180.0, self._settings.timeout_s),
            )
            response.raise_for_status()
        except httpx.HTTPError as exc:
            raise ProviderUnavailable(
                self.name,
                f"{type(exc).__name__} talking to {self._settings.base_url}. "
                f"Is 'ollama serve' running, and is '{self._model}' pulled?",
            ) from exc

        raw = str(response.json().get("message", {}).get("content", ""))
        try:
            return Brief.model_validate_json(_extract(raw)).cleaned()
        except Exception:  # noqa: BLE001 - an unreadable answer is an empty reading
            return Brief()

    def unload(self) -> None:
        """Release the vision model. It and the text model cannot share a 4 GB card."""
        try:
            httpx.post(
                f"{self._settings.base_url}/api/generate",
                json={"model": self._model, "keep_alive": 0},
                timeout=10.0,
            )
        except Exception:  # noqa: BLE001
            return


def _encode(image: Path) -> str:
    """Base64 JPEG, downscaled.

    Re-encoded rather than sent as-is: a 3 MB PNG becomes a 4 MB base64 string, and the
    model reduces it on arrival regardless.
    """
    import io

    with Image.open(image) as opened:
        picture = opened.convert("RGB")
        picture.thumbnail((MAX_EDGE, MAX_EDGE), Image.LANCZOS)
        buffer = io.BytesIO()
        picture.save(buffer, format="JPEG", quality=88)
    return base64.b64encode(buffer.getvalue()).decode("ascii")


def _schema() -> dict:
    """The brief's schema, with nested definitions inlined.

    Ollama compiles the schema into a decoding grammar and does not follow `$ref`, so a
    nested model would otherwise constrain nothing at all.
    """
    from genvai.adapters.ollama import _ollama_schema

    return _ollama_schema(Brief)


def _extract(raw: str) -> str:
    cleaned = _THINK.sub("", raw).strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.strip("`")
        cleaned = cleaned.split("\n", 1)[-1] if cleaned.lower().startswith("json") else cleaned
    match = _JSON.search(cleaned)
    if match is None:
        raise ValueError("no JSON object in reply")
    return match.group(0)
