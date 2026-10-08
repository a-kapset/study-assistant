from typing import TYPE_CHECKING

import pytest
from fastapi.testclient import TestClient

from study_assistant.composition import build_app
from study_assistant.settings import Settings

if TYPE_CHECKING:
    from conftest import FakeDatabase


def test_ready_answers_200_with_each_check_when_the_database_answers(settings: Settings, fake_db: FakeDatabase) -> None:
    """Guards the success body: a renamed field or a missing check would break every readiness probe."""
    with TestClient(build_app(settings, fake_db)) as client:
        response = client.get("/ready")

    assert response.status_code == 200
    assert response.json() == {"status": "ok", "checks": {"db": "ok"}}


def test_ready_answers_503_in_the_envelope_naming_the_database(settings: Settings, fake_db: FakeDatabase) -> None:
    """Guards the failure path: a 200, another status or an envelope without the failed check would hide the cause."""
    fake_db.up = False

    with TestClient(build_app(settings, fake_db)) as client:
        response = client.get("/ready", headers={"X-Request-ID": "ready-1"})

    assert response.status_code == 503
    assert response.json() == {
        "status": "error",
        "error": {
            "code": "service_unavailable",
            "message": "The app is not ready",
            "details": [{"loc": ["checks", "db"], "msg": "unavailable"}],
        },
        "request_id": "ready-1",
    }
    assert response.headers["X-Request-ID"] == "ready-1"


def test_health_stays_200_while_the_database_is_down(settings: Settings, fake_db: FakeDatabase) -> None:
    """Guards liveness: if /health checked the database, an orchestrator would restart a healthy process."""
    fake_db.up = False

    with TestClient(build_app(settings, fake_db)) as client:
        response = client.get("/health")

    assert response.status_code == 200


def test_a_bug_in_the_check_is_a_500_not_a_503(
    settings: Settings, fake_db: FakeDatabase, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Guards the narrow catch: catching every exception would report a bug in the app as an unavailable database."""

    async def broken_ping() -> None:
        """Fail the way a programming error would, not the way the database does."""
        raise RuntimeError("bug")

    monkeypatch.setattr(fake_db, "ping", broken_ping)

    with TestClient(build_app(settings, fake_db), raise_server_exceptions=False) as client:
        response = client.get("/ready")

    assert response.status_code == 500


def test_the_database_is_opened_at_startup_and_closed_at_shutdown(settings: Settings, fake_db: FakeDatabase) -> None:
    """Guards the lifespan: without it the pool would never open, without the close it would leak connections."""
    app = build_app(settings, fake_db)
    assert fake_db.events == []

    with TestClient(app):
        assert fake_db.events == ["open"]

    assert fake_db.events == ["open", "close"]
