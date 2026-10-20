"""Turns texts into vectors by asking the embedding server, which runs the embedding model.

Ingestion gives it the texts of course chunks and stores the vectors it gets back; search gives it the
learner's question. The model itself runs in the server; this module only prepares the texts, sends them
and returns the server's vectors.
"""

from typing import Self

import httpx2
import openai
from langchain_openai import OpenAIEmbeddings
from pydantic import SecretStr

from study_assistant.embed.base import EmbeddingSpace
from study_assistant.settings import EmbeddingServiceSettings, EmbeddingSettings

# Most texts the server takes in one request; a larger request is refused as a whole.
_BATCH_SIZE = 32
# Repeats of a request that could not connect, timed out or got a server error; a refused request is not repeated.
_MAX_RETRIES = 2
# The server does not check a key, but the client library refuses to start without one.
_UNUSED_KEY = SecretStr("not-used")
# How long the readiness check waits for the server's health answer, which needs no model work.
_PING_TIMEOUT_SECONDS = 5.0


class EmbeddingServiceError(Exception):
    """A call to the embedding server failed; no vectors were returned."""


class EmbeddingServiceUnavailableError(EmbeddingServiceError):
    """The embedding server could not be reached, or did not answer in time, even after the repeats."""


class EmbeddingServiceRequestError(EmbeddingServiceError):
    """The embedding server answered the request with an error status."""

    def __init__(self, status_code: int) -> None:
        """Keep the status the server answered with; it is the only detail the message shows."""
        super().__init__(f"the embedding server answered with status {status_code}")
        self.status_code = status_code


class RemoteEmbedder:
    """Sends texts to the embedding server and returns one vector per text, in the same order.

    Before sending, it puts the model's prefix in front of each text ("passage: " for course text,
    "query: " for a question), since the server cannot tell the two kinds apart. Long lists go out in
    several requests of at most 32 texts. A failed call raises an ``EmbeddingServiceError``.
    """

    def __init__(
        self, space: EmbeddingSpace, client: OpenAIEmbeddings, http_client: httpx2.AsyncClient, health_url: str
    ) -> None:
        """Use ``client`` for the embedding calls and ``http_client`` to ask ``health_url`` whether the server is
        ready; report ``space`` for every vector it returns."""
        self._space = space
        self._client = client
        self._http_client = http_client
        self._health_url = health_url

    @classmethod
    def from_settings(
        cls, embedding: EmbeddingSettings, service: EmbeddingServiceSettings, http_client: httpx2.AsyncClient
    ) -> Self:
        """Build the embedder for the configured model and server; nothing is sent until the first call.

        Every call goes through ``http_client``, which the caller opens and closes; tests give one that answers
        without a server.
        """
        base_url = str(service.url).rstrip("/")
        client = OpenAIEmbeddings(
            # The server ignores the model name; it is sent because the API requires one.
            model=embedding.model,
            openai_api_base=base_url + "/v1",
            # Given explicitly, so OPENAI_* variables in the environment can never redirect or change the calls.
            openai_api_key=_UNUSED_KEY,
            openai_organization=None,
            openai_proxy=None,
            openai_api_type=None,
            # Send the texts themselves: by default the library sends OpenAI token ids, which the server reads as
            # other tokens and answers with wrong vectors and a success status.
            check_embedding_ctx_length=False,
            chunk_size=_BATCH_SIZE,
            request_timeout=service.timeout_seconds,
            max_retries=_MAX_RETRIES,
            http_async_client=http_client,
        )
        return cls(EmbeddingSpace.from_settings(embedding), client, http_client, base_url + "/health")

    @property
    def space(self) -> EmbeddingSpace:
        """The embedding space of every vector this embedder returns."""
        return self._space

    async def aembed_documents(self, texts: list[str]) -> list[list[float]]:
        """Return one vector for each course text, in the same order, with the passage prefix added before sending."""
        return await self._embed([self._space.passage_prefix + text for text in texts])

    async def aembed_query(self, text: str) -> list[float]:
        """Return the vector of one search question, with the query prefix added before sending."""
        vectors = await self._embed([self._space.query_prefix + text])
        return vectors[0]

    async def ping(self) -> None:
        """Ask the server's health endpoint whether the model is loaded and can compute vectors now.

        No answer in time raises ``EmbeddingServiceUnavailableError`` with the error's type; any status but 200
        (the server answers 503 when it is not healthy) raises it with the status.
        """
        try:
            response = await self._http_client.get(self._health_url, timeout=_PING_TIMEOUT_SECONDS)
        except httpx2.TransportError as error:
            raise EmbeddingServiceUnavailableError(type(error).__name__) from error
        if response.status_code != 200:
            raise EmbeddingServiceUnavailableError(f"health status {response.status_code}")

    async def _embed(self, texts: list[str]) -> list[list[float]]:
        """Send texts that already carry their prefix to the server and return its vectors.

        A failed call raises an ``EmbeddingServiceError``; the library's own exception stays attached as the cause.
        """
        try:
            return await self._client.aembed_documents(texts)
        except openai.APIConnectionError as error:
            # Also covers a timeout, which the library reports as a subclass of the connection error.
            raise EmbeddingServiceUnavailableError(type(error).__name__) from error
        except openai.APIStatusError as error:
            raise EmbeddingServiceRequestError(error.status_code) from error
