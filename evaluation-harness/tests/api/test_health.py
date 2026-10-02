"""The health endpoint of the running system under test."""

from evalharness.transport import SutTransport


async def test_health_reports_ok(sut: SutTransport) -> None:
    """Guards the health contract the harness relies on before any other check: 200 and status ok."""
    observation = await sut.request("GET", "/health")

    assert observation.transport_error is None, f"system under test unreachable at {observation.url}"
    assert observation.status_code == 200
    assert observation.body_json == {"status": "ok"}
