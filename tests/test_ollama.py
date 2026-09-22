"""Ollama adapter, against a stubbed server.

No live model needed: the point is that malformed replies are handled, retries feed the
error back, and the schema sent to Ollama actually constrains anything.
"""

import json
from typing import Any

import httpx
import pytest
from pydantic import Field

from genvai.adapters.ollama import OllamaLLM, _extract_json, _ollama_schema
from genvai.config import LLMSettings
from genvai.errors import PlanningError, ProviderUnavailable
from genvai.timeline import Frozen


class Shot(Frozen):
    asset_id: str
    caption: str = ""


class Plan(Frozen):
    order: tuple[Shot, ...]
    hook_asset_id: str = Field(default="")


class _Reply:
    def __init__(self, content: str, status: int = 200) -> None:
        self._content = content
        self.status_code = status

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            raise httpx.HTTPStatusError("boom", request=None, response=None)  # type: ignore[arg-type]

    def json(self) -> dict:
        return {"message": {"content": self._content}}


@pytest.fixture
def llm() -> OllamaLLM:
    return OllamaLLM(LLMSettings(model="test-model", max_retries=3))


def _serve(monkeypatch: pytest.MonkeyPatch, replies: list[str]) -> list[dict[str, Any]]:
    """Answer each POST with the next reply, recording what was sent."""
    sent: list[dict[str, Any]] = []
    queue = list(replies)

    def fake_post(url: str, json: dict, timeout: float) -> _Reply:  # noqa: A002
        sent.append(json)
        return _Reply(queue.pop(0) if queue else "{}")

    monkeypatch.setattr(httpx, "post", fake_post)
    return sent


# ------------------------------------------------------------------ happy path


def test_returns_a_validated_model(llm: OllamaLLM, monkeypatch: pytest.MonkeyPatch) -> None:
    _serve(monkeypatch, ['{"order": [{"asset_id": "a", "caption": "hi"}], "hook_asset_id": "a"}'])
    plan = llm.structured("go", Plan)
    assert plan.order[0].asset_id == "a"
    assert plan.hook_asset_id == "a"


def test_sends_the_schema_so_decoding_is_constrained(
    llm: OllamaLLM, monkeypatch: pytest.MonkeyPatch
) -> None:
    sent = _serve(monkeypatch, ['{"order": [], "hook_asset_id": ""}'])
    llm.structured("go", Plan)
    assert "format" in sent[0]
    assert sent[0]["format"]["properties"].keys() >= {"order", "hook_asset_id"}


def test_system_prompt_is_passed(llm: OllamaLLM, monkeypatch: pytest.MonkeyPatch) -> None:
    sent = _serve(monkeypatch, ['{"order": [], "hook_asset_id": ""}'])
    llm.structured("go", Plan, system="be an editor")
    assert sent[0]["messages"][0] == {"role": "system", "content": "be an editor"}


def test_free_form_completion_sends_no_schema(
    llm: OllamaLLM, monkeypatch: pytest.MonkeyPatch
) -> None:
    sent = _serve(monkeypatch, ["just words"])
    assert llm.complete("hello") == "just words"
    assert "format" not in sent[0]


# ---------------------------------------------------------------------- retry


def test_retries_and_recovers(llm: OllamaLLM, monkeypatch: pytest.MonkeyPatch) -> None:
    _serve(
        monkeypatch,
        ["not json at all", '{"order": [{"asset_id": "a"}], "hook_asset_id": "a"}'],
    )
    assert llm.structured("go", Plan).order[0].asset_id == "a"


def test_the_error_is_fed_back_so_the_retry_can_fix_it(
    llm: OllamaLLM, monkeypatch: pytest.MonkeyPatch
) -> None:
    sent = _serve(monkeypatch, ["garbage", '{"order": [], "hook_asset_id": ""}'])
    llm.structured("order the shots", Plan)
    retry = sent[1]["messages"][-1]["content"]
    assert "order the shots" in retry, "the original request survives"
    assert "rejected" in retry.lower(), "and the model is told what went wrong"


def test_gives_up_loudly(llm: OllamaLLM, monkeypatch: pytest.MonkeyPatch) -> None:
    """Malformed output is never silently repaired - the user must know."""
    _serve(monkeypatch, ["nope", "still nope", "nope again"])
    with pytest.raises(PlanningError, match="valid Plan"):
        llm.structured("go", Plan)


def test_retry_budget_is_respected(llm: OllamaLLM, monkeypatch: pytest.MonkeyPatch) -> None:
    sent = _serve(monkeypatch, ["a", "b", "c", "d", "e"])
    with pytest.raises(PlanningError):
        llm.structured("go", Plan, max_retries=2)
    assert len(sent) == 2


def test_unreachable_server_is_a_typed_error(
    llm: OllamaLLM, monkeypatch: pytest.MonkeyPatch
) -> None:
    def refuse(url: str, json: dict, timeout: float) -> _Reply:  # noqa: A002
        raise httpx.ConnectError("refused")

    monkeypatch.setattr(httpx, "post", refuse)
    with pytest.raises(ProviderUnavailable, match="ollama serve"):
        llm.structured("go", Plan)


# ------------------------------------------------------------- reply handling


def test_strips_a_reasoning_scratchpad() -> None:
    raw = '<think>let me see...</think>\n{"order": [], "hook_asset_id": ""}'
    assert json.loads(_extract_json(raw)) == {"order": [], "hook_asset_id": ""}


def test_strips_a_fenced_block() -> None:
    raw = '```json\n{"a": 1}\n```'
    assert json.loads(_extract_json(raw)) == {"a": 1}


def test_finds_json_amid_prose() -> None:
    raw = 'Sure! Here you go:\n{"a": 1}\nHope that helps.'
    assert json.loads(_extract_json(raw)) == {"a": 1}


def test_a_reply_with_no_json_is_rejected() -> None:
    with pytest.raises(ValueError, match="no JSON object"):
        _extract_json("I would rather not.")


# --------------------------------------------------------------------- schema


def test_nested_models_are_inlined() -> None:
    """Ollama does not follow $ref into $defs, so a nested model would constrain nothing."""
    shape = _ollama_schema(Plan)
    assert "$defs" not in shape
    assert json.dumps(shape).count("$ref") == 0
    assert shape["properties"]["order"]["items"]["properties"].keys() >= {"asset_id", "caption"}


def test_availability_is_false_when_the_server_is_down(
    llm: OllamaLLM, monkeypatch: pytest.MonkeyPatch
) -> None:
    def refuse(url: str, timeout: float) -> _Reply:
        raise httpx.ConnectError("refused")

    monkeypatch.setattr(httpx, "get", refuse)
    assert llm.is_available() is False
