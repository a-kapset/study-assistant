import base64
import json
import struct
from collections.abc import Callable

import httpx2
import pytest

from study_assistant.embed.remote import (
    EmbeddingServiceRequestError,
    EmbeddingServiceUnavailableError,
    RemoteEmbedder,
)
from study_assistant.settings import EmbeddingServiceSettings, Settings

# The async tests run on the anyio plugin, which the web framework already brings.
pytestmark = pytest.mark.anyio

# The most texts the embedding server accepts in one request; it refuses a larger request as a whole.
SERVER_BATCH_LIMIT = 32


class FakeEmbeddingServer:
    """Answers embedding requests the way the real server does, and keeps every request it was sent.

    Each text gets the vector [n, 0.0], where n counts the texts across all requests, so a test can see which
    vector came back for which text. Like the real server it refuses more than 32 texts in one request.
    """

    def __init__(self) -> None:
        """Start with no requests seen."""
        self.requests: list[httpx2.Request] = []

    def inputs(self) -> list[list[object]]:
        """Return the list of inputs of every request, in the order they were sent."""
        return [json.loads(request.content)["input"] for request in self.requests]

    def handle(self, request: httpx2.Request) -> httpx2.Response:
        """Keep the request and answer with one vector per text, in the encoding the client asked for."""
        self.requests.append(request)
        body = json.loads(request.content)
        texts = body["input"]
        if len(texts) > SERVER_BATCH_LIMIT:
            return httpx2.Response(422, json={"error": "batch size too large", "error_type": "Validation"})
        first = sum(len(earlier) for earlier in self.inputs()[:-1])
        data = [
            {"object": "embedding", "index": i, "embedding": _encode([float(first + i), 0.0], body)}
            for i in range(len(texts))
        ]
        return httpx2.Response(
            200,
            json={
                "object": "list",
                "data": data,
                "model": body["model"],
                "usage": {"prompt_tokens": 0, "total_tokens": 0},
            },
        )


def _encode(vector: list[float], body: dict[str, object]) -> list[float] | str:
    """Return the vector as a list of numbers, or as base64 of 32-bit floats when the request asks for base64."""
    if body.get("encoding_format") == "base64":
        return base64.b64encode(struct.pack(f"<{len(vector)}f", *vector)).decode("ascii")
    return vector


type Handler = Callable[[httpx2.Request], httpx2.Response]


def _embedder(settings: Settings, handler: Handler, service: EmbeddingServiceSettings | None = None) -> RemoteEmbedder:
    """Build the embedder from the test settings, with every request answered by ``handler`` instead of a server."""
    client = httpx2.AsyncClient(transport=httpx2.MockTransport(handler))
    return RemoteEmbedder.from_settings(settings.embedding, service or settings.embedding_service, http_client=client)


@pytest.mark.covers("US-004", "AC-21")
async def test_passages_are_sent_as_text_with_the_passage_prefix(settings: Settings) -> None:
    """Embeds two passages and expects the server to receive exactly the two texts, each after "passage: ".

    Catches a lost or wrong prefix, and the client library's default of sending OpenAI token ids instead of
    text: the server would read those ids as other tokens and answer with wrong vectors and a success status.
    """
    server = FakeEmbeddingServer()

    await _embedder(settings, server.handle).aembed_documents(["Rule one.", "Rule two."])

    assert server.inputs() == [["passage: Rule one.", "passage: Rule two."]]


@pytest.mark.covers("US-004", "AC-21")
async def test_query_is_sent_as_text_with_the_query_prefix(settings: Settings) -> None:
    """Embeds one question and expects the server to receive only that text, after "query: ".

    Catches a question sent with the passage prefix or with none: the model was trained to match "query: "
    texts with "passage: " texts, and the search would silently find worse passages.
    """
    server = FakeEmbeddingServer()

    await _embedder(settings, server.handle).aembed_query("How many dice?")

    assert server.inputs() == [["query: How many dice?"]]


@pytest.mark.covers("US-004", "AC-17")
async def test_long_lists_are_sent_in_requests_the_server_accepts(settings: Settings) -> None:
    """Embeds 70 passages and expects requests of 32, 32 and 6 texts and 70 vectors back, in the texts' order.

    Catches a batch size above the server's limit, such as the client library's default of 1,000: a course
    with more than 32 chunks would then be refused as a whole and get no vectors.
    """
    server = FakeEmbeddingServer()
    texts = [f"text {n}" for n in range(70)]

    vectors = await _embedder(settings, server.handle).aembed_documents(texts)

    assert [len(batch) for batch in server.inputs()] == [32, 32, 6]
    assert [vector[0] for vector in vectors] == [float(n) for n in range(70)]


async def test_requests_wait_no_longer_than_the_configured_timeout(settings: Settings) -> None:
    """Configures a 5-second timeout and expects every part of the request to be limited to 5 seconds.

    Catches the setting not reaching the client: the library would then wait up to ten minutes for a server
    that hangs, and ingestion would stop for that long.
    """
    server = FakeEmbeddingServer()
    service = EmbeddingServiceSettings.model_validate({"url": "http://embedder.invalid", "timeout_seconds": 5})

    await _embedder(settings, server.handle, service).aembed_documents(["text"])

    assert server.requests[0].extensions["timeout"] == {"connect": 5.0, "read": 5.0, "write": 5.0, "pool": 5.0}


async def test_openai_variables_in_the_environment_do_not_change_the_calls(
    settings: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Sets the OpenAI client's own variables and expects the call to go to the configured server with no real key.

    Catches a parameter left to the client library: it reads OPENAI_API_KEY, OPENAI_API_BASE, OPENAI_PROXY and
    OPENAI_API_TYPE by itself, so a developer's real OpenAI key could be sent to the embedding server, or the
    calls could go through a proxy or fail as an Azure setup.
    """
    monkeypatch.setenv("OPENAI_API_KEY", "sk-real-key-from-the-shell")
    monkeypatch.setenv("OPENAI_API_BASE", "http://other-host.invalid/v1")
    monkeypatch.setenv("OPENAI_BASE_URL", "http://other-host.invalid/v1")
    monkeypatch.setenv("OPENAI_PROXY", "http://proxy.invalid:3128")
    monkeypatch.setenv("OPENAI_API_TYPE", "azure")
    server = FakeEmbeddingServer()

    await _embedder(settings, server.handle).aembed_documents(["text"])

    request = server.requests[0]
    assert str(request.url) == "http://embedder.invalid/v1/embeddings"
    assert "sk-real-key-from-the-shell" not in request.headers.get("authorization", "")


async def test_unreachable_server_is_reported_as_unavailable_after_the_repeats(settings: Settings) -> None:
    """Lets every connection fail and expects three attempts, then an ``EmbeddingServiceUnavailableError``.

    Catches the library's error leaking out (ingestion could not tell it from a bug) and the repeats being
    switched off (a server that restarts for a moment would fail a whole ingestion). Takes about 1.5 s, because
    the library waits between the attempts.
    """
    attempts: list[httpx2.Request] = []

    def refuse(request: httpx2.Request) -> httpx2.Response:
        """Count the attempt and fail it the way a closed port does."""
        attempts.append(request)
        raise httpx2.ConnectError("connection refused", request=request)

    with pytest.raises(EmbeddingServiceUnavailableError):
        await _embedder(settings, refuse).aembed_documents(["text"])

    assert len(attempts) == 3


async def test_error_status_is_reported_with_its_code_and_not_repeated(settings: Settings) -> None:
    """Answers with status 422 and expects one attempt and an ``EmbeddingServiceRequestError`` carrying 422.

    Catches the library's error leaking out, and a refused request being repeated: the same request would be
    refused again, and ingestion would only fail later.
    """
    attempts: list[httpx2.Request] = []

    def refuse(request: httpx2.Request) -> httpx2.Response:
        """Count the attempt and refuse it the way the server refuses an invalid request."""
        attempts.append(request)
        return httpx2.Response(422, json={"error": "invalid input", "error_type": "Validation"})

    with pytest.raises(EmbeddingServiceRequestError) as error:
        await _embedder(settings, refuse).aembed_documents(["text"])

    assert error.value.status_code == 422
    assert len(attempts) == 1


@pytest.mark.covers("US-001", "AC-2")
async def test_ping_asks_the_servers_health_endpoint(settings: Settings) -> None:
    """Answers 200 on the server's health path and expects ping to return after one GET of exactly that path.

    Catches a ping sent to another path (the real server answers 404 there, so the app would never be ready) or one
    that computes a vector, which makes every readiness probe wait for the model.
    """
    requests: list[httpx2.Request] = []

    def healthy(request: httpx2.Request) -> httpx2.Response:
        """Keep the request and answer like a healthy server."""
        requests.append(request)
        return httpx2.Response(200)

    await _embedder(settings, healthy).ping()

    assert [(request.method, str(request.url)) for request in requests] == [("GET", "http://embedder.invalid/health")]


def _refuse_connection(request: httpx2.Request) -> httpx2.Response:
    """Fail the way a server that is not running does."""
    raise httpx2.ConnectError("connection refused", request=request)


@pytest.mark.covers("US-001", "AC-2")
@pytest.mark.parametrize(
    ("handler", "reason"),
    [
        pytest.param(_refuse_connection, "ConnectError", id="unreachable"),
        pytest.param(lambda request: httpx2.Response(503), "health status 503", id="unhealthy"),
    ],
)
async def test_ping_reports_an_unreachable_or_unhealthy_server_as_unavailable(
    settings: Settings, handler: Handler, reason: str
) -> None:
    """Refuses the connection, or answers 503, and expects ``EmbeddingServiceUnavailableError`` naming the reason.

    Catches a ping that lets the transport error escape (readiness would answer 500 instead of 503) or that treats
    any answer as healthy.
    """
    with pytest.raises(EmbeddingServiceUnavailableError, match=reason):
        await _embedder(settings, handler).ping()
