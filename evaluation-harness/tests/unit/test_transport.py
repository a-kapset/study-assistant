"""SutTransport turns every exchange into an Observation, with no network involved."""

from collections.abc import Callable

import httpx2
import pytest
from pydantic import HttpUrl

from evalharness.observation import TransportError
from evalharness.settings import HarnessSettings
from evalharness.transport import REQUEST_ID_HEADER, SutTransport

type Handler = Callable[[httpx2.Request], httpx2.Response]


def _transport(handler: Handler, base_url: str = "http://sut.test") -> SutTransport:
    """Build a transport whose requests are answered by the handler instead of the network."""
    return SutTransport(HarnessSettings(sut_base_url=HttpUrl(base_url)), transport=httpx2.MockTransport(handler))


async def test_error_status_is_returned_not_raised() -> None:
    """Guards against raising on non-2xx: an error response must reach the classifier as data."""
    async with _transport(lambda request: httpx2.Response(500, text="internal error")) as sut:
        observation = await sut.request("GET", "/health")

    assert observation.status_code == 500
    assert observation.body_text == "internal error"
    assert observation.body_json is None
    assert observation.transport_error is None


async def test_json_body_is_parsed() -> None:
    """Guards the JSON parsing: a valid JSON body must be kept as data, not dropped to None."""
    async with _transport(lambda request: httpx2.Response(200, json={"status": "ok"})) as sut:
        observation = await sut.request("GET", "/health")

    assert observation.body_json == {"status": "ok"}


async def test_request_id_is_sent_and_overrides_caller_header() -> None:
    """Guards the header merge order: the id in the observation must be the one the app received."""
    received: list[str] = []

    def handler(request: httpx2.Request) -> httpx2.Response:
        """Record the request id header the app would see."""
        received.append(request.headers[REQUEST_ID_HEADER])
        return httpx2.Response(200)

    async with _transport(handler) as sut:
        observation = await sut.request("GET", "/health", headers={REQUEST_ID_HEADER: "from-caller"})

    assert received == [observation.request_id]


@pytest.mark.parametrize(
    ("error_type", "expected"),
    [
        (httpx2.ConnectError, "connect"),
        (httpx2.ReadTimeout, "timeout"),
        # Failing to connect in time is still a timeout, not a connect error.
        (httpx2.ConnectTimeout, "timeout"),
        (httpx2.RemoteProtocolError, "other"),
    ],
)
async def test_exchange_error_becomes_classified_observation(
    error_type: type[httpx2.TransportError], expected: TransportError
) -> None:
    """Guards the error classification, and that a failed exchange is recorded instead of raised."""

    def handler(request: httpx2.Request) -> httpx2.Response:
        """Fail the exchange the way the network would."""
        raise error_type("simulated", request=request)

    async with _transport(handler) as sut:
        observation = await sut.request("GET", "/health")

    assert observation.status_code is None
    assert observation.transport_error == expected
    assert observation.url == "http://sut.test/health"


async def test_base_url_path_prefix_is_kept() -> None:
    """Guards how the configured URL is passed on: an app mounted under a path must stay reachable."""
    async with _transport(lambda request: httpx2.Response(200), base_url="http://sut.test/api") as sut:
        observation = await sut.request("GET", "/health")

    assert observation.url == "http://sut.test/api/health"
