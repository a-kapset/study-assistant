"""Scores how well each passage answers a question by asking the rerank server, which runs a cross-encoder.

A cross-encoder reads the question and one passage together, so it judges their match more closely than a
comparison of two vectors made apart; it is also much slower, so a search gives it only the passages it already
found. The model runs in the server; this module only sends the texts and returns the server's scores.
"""

from typing import Protocol, Self

import httpx2
from pydantic import BaseModel, TypeAdapter

from study_assistant.settings import RerankSettings

# Most texts the server takes in one request; it refuses a larger request as a whole (with 422).
_BATCH_SIZE = 32
# How long the readiness check waits for the server's health answer, which needs no model work.
_PING_TIMEOUT_SECONDS = 5.0


class RerankServiceError(Exception):
    """A call to the rerank server failed; no scores were returned."""


class RerankServiceUnavailableError(RerankServiceError):
    """The rerank server could not be reached, or did not answer in time."""


class RerankServiceRequestError(RerankServiceError):
    """The rerank server answered the request with an error status."""

    def __init__(self, status_code: int) -> None:
        """Keep the status the server answered with; it is the only detail the message shows."""
        super().__init__(f"the rerank server answered with status {status_code}")
        self.status_code = status_code


class Reranker(Protocol):
    """Scores passages against a question; a higher score means a better answer."""

    async def ascore(self, query: str, texts: list[str]) -> list[float]:
        """Return one score for each text, in the order of ``texts``."""
        ...

    async def ping(self) -> None:
        """Return when the scorer can work now; raise ``RerankServiceUnavailableError`` otherwise."""
        ...


class _Rank(BaseModel):
    """One entry of the server's reply: the place of a text in the request and its score."""

    # Fields the server may add later are ignored: the client reads only these two.
    index: int
    score: float


_RANKS = TypeAdapter(list[_Rank])


class RemoteReranker:
    """Sends a question and passages to the rerank server and returns their scores in the order they were given.

    The server answers best first, each score with the place of its text, so the scores are put back in place.
    Lists longer than 32 go out in several requests; a failed call raises a ``RerankServiceError``.
    """

    def __init__(self, http_client: httpx2.AsyncClient, base_url: str, timeout_seconds: float) -> None:
        """Send every call through ``http_client`` to the server at ``base_url``; wait ``timeout_seconds`` at most."""
        self._http_client = http_client
        self._rerank_url = base_url + "/rerank"
        self._health_url = base_url + "/health"
        self._timeout = timeout_seconds

    @classmethod
    def from_settings(cls, settings: RerankSettings, http_client: httpx2.AsyncClient) -> Self:
        """Build the client for the configured server; nothing is sent until the first call.

        The caller opens and closes ``http_client``; tests give one that answers without a server.
        """
        return cls(http_client, str(settings.url).rstrip("/"), settings.timeout_seconds)

    async def ascore(self, query: str, texts: list[str]) -> list[float]:
        """Return one score for each text, in the order of ``texts``; no request is sent for an empty list."""
        scores: list[float] = []
        for start in range(0, len(texts), _BATCH_SIZE):
            scores += await self._score_batch(query, texts[start : start + _BATCH_SIZE])
        return scores

    async def ping(self) -> None:
        """Ask the server's health endpoint whether the model is loaded and can score now.

        No answer in time raises ``RerankServiceUnavailableError`` with the error's type; any status but 200 raises it
        with the status.
        """
        try:
            response = await self._http_client.get(self._health_url, timeout=_PING_TIMEOUT_SECONDS)
        except httpx2.TransportError as error:
            raise RerankServiceUnavailableError(type(error).__name__) from error
        if response.status_code != 200:
            raise RerankServiceUnavailableError(f"health status {response.status_code}")

    async def _score_batch(self, query: str, texts: list[str]) -> list[float]:
        """Send one request of at most 32 texts and return their scores in the order of ``texts``.

        ``truncate`` is sent so that a long question and passage are cut to what the model reads, whatever the
        server's own default is; without it such a pair is refused.
        """
        try:
            response = await self._http_client.post(
                self._rerank_url,
                json={"query": query, "texts": texts, "truncate": True},
                timeout=self._timeout,
            )
        except httpx2.TransportError as error:
            # Also covers a timeout, which the client reports as a transport error.
            raise RerankServiceUnavailableError(type(error).__name__) from error
        if response.status_code != 200:
            raise RerankServiceRequestError(response.status_code)
        score_at = {rank.index: rank.score for rank in _RANKS.validate_json(response.content)}
        return [score_at[place] for place in range(len(texts))]
