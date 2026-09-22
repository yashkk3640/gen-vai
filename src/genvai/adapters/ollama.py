"""LLMPort over the Ollama HTTP API.

Structured output goes through Ollama's `format` parameter, which constrains decoding to
a JSON schema rather than asking the model nicely and hoping. Small local models still
produce schema-valid-but-wrong output often enough that `structured` validates and
retries, feeding the error back so the next attempt has something to correct.
"""

import re
from typing import Any, TypeVar

import httpx
from pydantic import BaseModel, ValidationError

from genvai.config import LLMSettings
from genvai.errors import PlanningError, ProviderUnavailable

T = TypeVar("T", bound=BaseModel)

_THINK = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)
"""Reasoning models emit their scratchpad inline. It is not part of the answer."""

_JSON_OBJECT = re.compile(r"\{.*\}", re.DOTALL)


class OllamaLLM:
    """Implements `LLMPort`."""

    def __init__(self, settings: LLMSettings) -> None:
        self._settings = settings

    @property
    def name(self) -> str:
        return f"ollama/{self._settings.model}"

    def is_available(self) -> bool:
        """True if the server answers and has the configured model pulled."""
        try:
            response = httpx.get(f"{self._settings.base_url}/api/tags", timeout=3.0)
            response.raise_for_status()
        except Exception:  # noqa: BLE001 - availability checks never raise
            return False
        names = {m.get("name", "") for m in response.json().get("models", [])}
        return self._settings.model in names or f"{self._settings.model}:latest" in names

    def complete(self, prompt: str, *, system: str | None = None) -> str:
        """Free-form completion. Used for rationales and summaries, not for structure."""
        return self._chat(prompt, system=system)

    def structured(
        self,
        prompt: str,
        schema: type[T],
        *,
        system: str | None = None,
        max_retries: int | None = None,
    ) -> T:
        """Completion constrained to `schema`, validated and retried.

        Raises `PlanningError` once the budget is exhausted. Malformed output is never
        silently repaired - a quietly patched plan is worse than a visible failure,
        because the user would have no way to know the model ignored half the request.
        """
        budget = max_retries if max_retries is not None else self._settings.max_retries
        shape = _ollama_schema(schema)
        attempt_prompt = prompt
        problems: list[str] = []

        for _ in range(max(1, budget)):
            raw = self._chat(attempt_prompt, system=system, fmt=shape)
            try:
                return schema.model_validate_json(_extract_json(raw))
            except (ValidationError, ValueError) as exc:
                problems.append(str(exc)[:600])
                attempt_prompt = (
                    f"{prompt}\n\nYour previous answer was rejected:\n{problems[-1]}\n"
                    "Return only JSON matching the schema. Fix exactly what was wrong."
                )

        raise PlanningError(
            f"{self.name} did not produce valid {schema.__name__} in {budget} attempts. "
            f"Last error: {problems[-1] if problems else 'unknown'}"
        )

    def unload(self) -> None:
        """Release the model from VRAM.

        Called at the plan/resolve phase boundary; at 4 GB the LLM and a diffusion model
        cannot coexist. Failure is ignored - it is a hint to the server, not a promise.
        """
        try:
            httpx.post(
                f"{self._settings.base_url}/api/generate",
                json={"model": self._settings.model, "keep_alive": 0},
                timeout=10.0,
            )
        except Exception:  # noqa: BLE001
            return

    def _chat(self, prompt: str, *, system: str | None, fmt: dict | None = None) -> str:
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        payload: dict[str, Any] = {
            "model": self._settings.model,
            "messages": messages,
            "stream": False,
            "think": False,
            "keep_alive": self._settings.keep_alive,
            "options": {"temperature": self._settings.temperature},
        }
        if fmt is not None:
            payload["format"] = fmt

        try:
            response = httpx.post(
                f"{self._settings.base_url}/api/chat",
                json=payload,
                timeout=self._settings.timeout_s,
            )
            response.raise_for_status()
        except httpx.HTTPError as exc:
            raise ProviderUnavailable(
                self.name,
                f"{type(exc).__name__} talking to {self._settings.base_url}. "
                "Is 'ollama serve' running?",
            ) from exc

        return str(response.json().get("message", {}).get("content", ""))


def _ollama_schema(schema: type[BaseModel]) -> dict:
    """JSON Schema for a pydantic model, with nested definitions inlined.

    Ollama compiles the schema into a decoding grammar and does not follow `$ref` into
    `$defs`, so a model with any nested type would otherwise constrain to nothing.
    Inlining keeps the constraint real.
    """
    return _inline_refs(schema.model_json_schema())


def _inline_refs(node: Any, defs: dict | None = None) -> Any:
    if isinstance(node, dict):
        known = defs if defs is not None else node.get("$defs", {})
        if "$ref" in node:
            name = str(node["$ref"]).rsplit("/", 1)[-1]
            return _inline_refs(known.get(name, {}), known)
        return {k: _inline_refs(v, known) for k, v in node.items() if k != "$defs"}
    if isinstance(node, list):
        return [_inline_refs(item, defs) for item in node]
    return node


def _extract_json(raw: str) -> str:
    """Pull the JSON object out of a reply.

    Constrained decoding usually returns clean JSON, but a reasoning model may still
    wrap it in a scratchpad or prose, and a rejected reply is more useful than a crash.
    """
    cleaned = _THINK.sub("", raw).strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.strip("`")
        cleaned = cleaned.split("\n", 1)[-1] if cleaned.lower().startswith("json") else cleaned
    match = _JSON_OBJECT.search(cleaned)
    if match is None:
        raise ValueError(f"no JSON object in reply: {cleaned[:200]!r}")
    return match.group(0)
