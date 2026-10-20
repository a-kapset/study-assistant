"""The readiness endpoint of the running system under test, with its real database and embedding server."""

import pytest

from evalharness.transport import SutTransport


@pytest.mark.covers("US-001", "AC-2")
async def test_ready_reports_the_database_and_the_embedder_ok(sut: SutTransport) -> None:
    """Guards the deployed wiring: a wrong database host or password, a missing driver, or a wrong embedding server
    address turns readiness into a 503 here."""
    observation = await sut.request("GET", "/ready")

    assert observation.transport_error is None, f"system under test unreachable at {observation.url}"
    assert observation.status_code == 200, f"not ready: {observation.body_text}"
    assert observation.body_json == {"status": "ok", "checks": {"db": "ok", "embedder": "ok"}}
    assert observation.headers["x-request-id"] == observation.request_id
