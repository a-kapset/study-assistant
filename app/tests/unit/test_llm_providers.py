"""The chat model built from configuration: where its calls go and in what form, and how a provider's failure
reaches the rest of the app. The real client library runs; only the network is replaced by a mock transport."""

import json
from collections.abc import Callable

import httpx2
import pytest
from langchain_core.messages import AIMessage, HumanMessage

from study_assistant.composition import build_app
from study_assistant.db import Database
from study_assistant.llm.providers import (
    LlmRequestError,
    LlmUnavailableError,
    UnknownProviderError,
    ainvoke,
    build_chat_model,
)
from study_assistant.settings import LlmSettings, OpenAISettings, Settings

from .conftest import responses_api_reply

pytestmark = pytest.mark.anyio

type Handler = Callable[[httpx2.Request], httpx2.Response]


class FakeProvider:
    """Answers every call with one short text, the way the OpenAI Responses API does, and keeps every request."""

    def __init__(self) -> None:
        """Start with no requests seen."""
        self.requests: list[httpx2.Request] = []

    def body(self, index: int = 0) -> dict[str, object]:
        """Return the JSON body of a request that was sent."""
        body: dict[str, object] = json.loads(self.requests[index].content)
        return body

    def handle(self, request: httpx2.Request) -> httpx2.Response:
        """Keep the request and answer with a completed reply."""
        self.requests.append(request)
        return httpx2.Response(
            200, json=responses_api_reply([{"type": "output_text", "text": "Two.", "annotations": []}])
        )


async def _ask(settings: Settings, handler: Handler) -> AIMessage:
    """Build the answer model from ``settings`` with every request answered by ``handler``, and ask it one question."""
    client = httpx2.AsyncClient(transport=httpx2.MockTransport(handler))
    model = build_chat_model(settings.llm.answer, settings, client)
    return await ainvoke(model, [HumanMessage("How many dice?")])


@pytest.mark.covers("US-009", "AC-2")
def test_unknown_provider_stops_startup_naming_the_known_ones(settings: Settings, fake_db: Database) -> None:
    """Configures the answer model as "mistral/large" and builds the app; expects startup to stop with an error that
    names the unknown provider and the ones that exist.

    Catches a factory that falls back to some provider for a name it does not know: the app would start and send
    every question to a model nobody chose.
    """
    configured = settings.model_copy(update={"llm": LlmSettings(answer="mistral/large")})

    with pytest.raises(UnknownProviderError) as error:
        build_app(configured, fake_db)

    message = str(error.value)
    assert "'mistral'" in message and "openai" in message, f"the error should name both providers: {message}"


async def test_calls_go_to_the_configured_address_with_the_configured_key_and_model(settings: Settings) -> None:
    """Asks one question and expects the request at the configured base address, with the configured key and the
    model name from "provider/model" without the provider.

    Catches a setting not reaching the client: the call would go to OpenAI's default address (past a recording
    proxy), carry another key, or name the model "openai/gpt-5.4-mini", which the provider does not know.
    """
    provider = FakeProvider()

    await _ask(settings, provider.handle)

    request = provider.requests[0]
    assert str(request.url).startswith("http://llm.invalid/v1/"), f"the call went to {request.url}"
    assert request.headers["authorization"] == "Bearer unit-test-openai-key", "the call carried another key"
    assert provider.body()["model"] == "gpt-5.4-mini", f"the call named the model {provider.body()['model']!r}"


async def test_calls_use_the_responses_api_and_ask_the_provider_to_store_nothing(settings: Settings) -> None:
    """Asks one question and expects a request to the Responses API with "store" set to false.

    Catches the chat completions API being used instead, and a call the provider keeps: a kept call could be
    continued from the provider's copy, so a recorded call would no longer be complete on its own.
    """
    provider = FakeProvider()

    await _ask(settings, provider.handle)

    assert provider.requests[0].url.path == "/v1/responses", f"the call went to {provider.requests[0].url.path}"
    assert provider.body().get("store") is False, f"the call asked for store={provider.body().get('store')!r}"


async def test_openai_variables_in_the_environment_do_not_change_the_calls(
    settings: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Sets the OpenAI client's own key and address variables and expects the call to go to the configured address
    with the configured key.

    Catches a key or an address left to the client library: a developer's own OpenAI key or address in the shell
    would then decide where questions go and who pays for them.
    """
    monkeypatch.setenv("OPENAI_API_KEY", "sk-real-key-from-the-shell")
    monkeypatch.setenv("OPENAI_API_BASE", "http://other-host.invalid/v1")
    monkeypatch.setenv("OPENAI_BASE_URL", "http://other-host.invalid/v1")
    provider = FakeProvider()

    await _ask(settings, provider.handle)

    request = provider.requests[0]
    assert request.url.host == "llm.invalid", f"an environment variable sent the call to {request.url.host}"
    assert "sk-real-key-from-the-shell" not in request.headers["authorization"], "the shell's key was sent"


async def test_calls_wait_no_longer_than_the_configured_timeout(settings: Settings) -> None:
    """Configures a 5-second timeout and expects every part of the request to be limited to 5 seconds.

    Catches the setting not reaching the client: the library would wait up to ten minutes for a provider that
    hangs, and the learner's request would hang as long.
    """
    openai_settings = OpenAISettings.model_validate(settings.openai.model_dump() | {"timeout_seconds": 5})
    provider = FakeProvider()

    await _ask(settings.model_copy(update={"openai": openai_settings}), provider.handle)

    timeout = provider.requests[0].extensions["timeout"]
    assert timeout == {"connect": 5.0, "read": 5.0, "write": 5.0, "pool": 5.0}, f"the call waits {timeout}"


@pytest.mark.covers("US-009", "AC-3")
async def test_unreachable_provider_is_reported_as_unavailable_after_two_repeats(settings: Settings) -> None:
    """Lets every connection fail and expects three attempts, then an ``LlmUnavailableError``.

    Catches the library's error leaking out (the answer layer could not tell it from a bug), a changed number of
    repeats (none fails on a short network break; many hold the learner's request for minutes), and the two kinds
    of failure being swapped. Takes about 1.5 s, because the library waits between the attempts.
    """
    attempts: list[httpx2.Request] = []

    def refuse(request: httpx2.Request) -> httpx2.Response:
        """Count the attempt and fail it the way a closed port does."""
        attempts.append(request)
        raise httpx2.ConnectError("connection refused", request=request)

    with pytest.raises(LlmUnavailableError):
        await _ask(settings, refuse)

    assert len(attempts) == 3, f"expected the first attempt and two repeats, got {len(attempts)} attempts"


@pytest.mark.covers("US-009", "AC-3")
async def test_error_status_is_reported_with_its_code(settings: Settings) -> None:
    """Answers with status 401, as for a wrong key, and expects an ``LlmRequestError`` carrying 401.

    Catches the library's error leaking out, and an error status reported as an unreachable provider: the learner
    would be told to try later when the configuration is wrong.
    """

    def refuse(request: httpx2.Request) -> httpx2.Response:
        """Refuse the call the way the provider refuses a wrong key."""
        return httpx2.Response(401, json={"error": {"message": "Incorrect API key", "type": "invalid_request_error"}})

    with pytest.raises(LlmRequestError) as error:
        await _ask(settings, refuse)

    assert error.value.status_code == 401, f"the error carries status {error.value.status_code}"
