"""The only way the harness talks to the system under test: one HTTP exchange becomes one Observation."""

import time
import uuid
from collections.abc import Mapping
from types import TracebackType
from typing import Self

import httpx2
import structlog
from pydantic import JsonValue

from evalharness.observation import Observation, TransportError
from evalharness.settings import HarnessSettings

# Sent with every request so a harness log line can be matched with the app's own logs.
REQUEST_ID_HEADER = "X-Request-ID"


class SutTransport:
    """Async HTTP client for the system under test that reports every outcome instead of raising.

    An error status or a request that got no response is a result to classify, not an exception,
    so a single failing case never stops a run. Use it as an async context manager so the
    connection pool is closed at the end.
    """

    def __init__(self, settings: HarnessSettings, transport: httpx2.AsyncBaseTransport | None = None) -> None:
        """Prepare the client from settings; tests pass httpx2.MockTransport as the transport."""
        self._client = httpx2.AsyncClient(
            base_url=str(settings.sut_base_url),
            timeout=settings.request_timeout_s,
            transport=transport,
            # Proxy variables from the environment must not silently reroute requests to the system under test.
            trust_env=False,
        )
        self._log: structlog.typing.FilteringBoundLogger = structlog.get_logger(__name__)

    async def __aenter__(self) -> Self:
        """Open the underlying client."""
        await self._client.__aenter__()
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        """Close the underlying client and its connections."""
        await self._client.__aexit__(exc_type, exc, tb)

    async def request(
        self,
        method: str,
        path: str,
        *,
        json: JsonValue | None = None,
        headers: Mapping[str, str] | None = None,
    ) -> Observation:
        """Send one request and record what happened; never raises for an outcome of the exchange."""
        request_id = uuid.uuid4().hex
        request = self._client.build_request(
            method, path, json=json, headers={**(headers or {}), REQUEST_ID_HEADER: request_id}
        )
        log = self._log.bind(request_id=request_id, method=request.method, url=str(request.url))
        log.info("request", json=json)

        started = time.perf_counter()
        try:
            response = await self._client.send(request)
        except httpx2.HTTPError as error:
            elapsed_s = time.perf_counter() - started
            kind = _classify(error)
            log.warning("transport_error", transport_error=kind, error=repr(error), elapsed_s=elapsed_s)
            return Observation(
                request_id=request_id,
                method=request.method,
                url=str(request.url),
                elapsed_s=elapsed_s,
                status_code=None,
                headers={},
                body_text="",
                body_json=None,
                transport_error=kind,
            )
        elapsed_s = time.perf_counter() - started
        log.info("response", status_code=response.status_code, elapsed_s=elapsed_s)
        return Observation(
            request_id=request_id,
            method=request.method,
            url=str(request.url),
            elapsed_s=elapsed_s,
            status_code=response.status_code,
            headers=dict(response.headers),
            body_text=response.text,
            body_json=_parse_json(response),
            transport_error=None,
        )


def _classify(error: httpx2.HTTPError) -> TransportError:
    """Name why no response arrived; a timeout while connecting counts as a timeout."""
    if isinstance(error, httpx2.TimeoutException):
        return "timeout"
    if isinstance(error, httpx2.ConnectError):
        return "connect"
    return "other"


def _parse_json(response: httpx2.Response) -> JsonValue | None:
    """Return the body as JSON, or None when it is empty or not valid JSON."""
    try:
        parsed: JsonValue = response.json()
    except ValueError:
        return None
    return parsed
