from fastapi.testclient import TestClient

from study_assistant.composition import build_app
from study_assistant.db import Database
from study_assistant.settings import Settings


def test_health_returns_ok(settings: Settings, fake_db: Database) -> None:
    """Guards the liveness contract: a broken route or response model would fail every health probe."""
    with TestClient(build_app(settings, fake_db)) as client:
        response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}
