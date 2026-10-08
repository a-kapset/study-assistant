"""The readiness endpoint of the running system under test, with its real database."""

from evalharness.transport import SutTransport


async def test_ready_reports_the_database_ok(sut: SutTransport) -> None:
    """Guards the deployed wiring: a wrong host, password or missing driver turns readiness into a 503 here."""
    observation = await sut.request("GET", "/ready")

    assert observation.transport_error is None, f"system under test unreachable at {observation.url}"
    assert observation.status_code == 200, f"not ready: {observation.body_text}"
    assert observation.body_json == {"status": "ok", "checks": {"db": "ok"}}
    assert observation.headers["x-request-id"] == observation.request_id
