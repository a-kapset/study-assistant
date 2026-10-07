"""The error format and request ids of the running system under test, as any client sees them."""

from evalharness.observation import Observation
from evalharness.transport import SutTransport


def _assert_error_envelope(observation: Observation, code: str, message: str) -> None:
    """Check the whole error body: its fields, the error code, and that it names the request the harness sent."""
    assert observation.body_json == {
        "status": "error",
        "error": {"code": code, "message": message},
        "request_id": observation.request_id,
    }


async def test_unknown_path_answers_404_in_the_envelope(sut: SutTransport) -> None:
    """Guards the error contract seen from outside: an unknown path is 404 in the envelope, never 200 or plain text."""
    observation = await sut.request("GET", "/no-such-path")

    assert observation.transport_error is None, f"system under test unreachable at {observation.url}"
    assert observation.status_code == 404
    _assert_error_envelope(observation, "not_found", "Not Found")


async def test_wrong_method_answers_405_with_allow(sut: SutTransport) -> None:
    """Guards the 405 contract: the exact status, the envelope and the Allow header that tells a client what works."""
    observation = await sut.request("POST", "/health")

    assert observation.transport_error is None, f"system under test unreachable at {observation.url}"
    assert observation.status_code == 405
    assert observation.headers["allow"] == "GET"
    _assert_error_envelope(observation, "method_not_allowed", "Method Not Allowed")


async def test_successful_response_echoes_the_harness_request_id(sut: SutTransport) -> None:
    """Guards correlation: every observation must be findable in the app's logs by the id the harness sent."""
    observation = await sut.request("GET", "/health")

    assert observation.transport_error is None, f"system under test unreachable at {observation.url}"
    assert observation.status_code == 200
    assert observation.headers["x-request-id"] == observation.request_id
