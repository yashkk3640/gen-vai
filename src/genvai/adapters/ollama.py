"""LLMPort over the Ollama HTTP API.

Structured output uses Ollama's `format` parameter with the pydantic model's JSON
schema, which constrains decoding rather than hoping the model complies. Invalid
output is still possible with small models, so `structured` retries with the
validation error fed back.
"""

from typing import TypeVar

from pydantic import BaseModel

from genvai.config import LLMSettings

T = TypeVar("T", bound=BaseModel)


class OllamaLLM:
    """Implements `LLMPort`."""

    def __init__(self, settings: LLMSettings) -> None:
        self._settings = settings

    def is_available(self) -> bool:
        """True if the server responds and has the configured model pulled."""
        raise NotImplementedError

    def complete(self, prompt: str, *, system: str | None = None) -> str:
        raise NotImplementedError

    def structured(
        self,
        prompt: str,
        schema: type[T],
        *,
        system: str | None = None,
        max_retries: int = 3,
    ) -> T:
        raise NotImplementedError

    def unload(self) -> None:
        """Release VRAM by issuing a request with keep_alive=0."""
        raise NotImplementedError
