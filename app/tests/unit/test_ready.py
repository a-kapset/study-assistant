from typing import TYPE_CHECKING

import pytest
from fastapi.testclient import TestClient

from study_assistant.composition import build_app
from study_assistant.settings import SearchSettings, Settings

if TYPE_CHECKING:
    from .conftest import FakeDatabase, FakeEmbedder, FakeReranker


@pytest.mark.covers("US-001", "AC-2")
def test_ready_answers_200_with_each_check_when_the_database_and_the_embedder_answer(
    settings: Settings, fake_db: FakeDatabase, fake_embedder: FakeEmbedder
) -> None:
    """Guards the success body: a renamed field or a missing check would break every readiness probe."""
    with TestClient(build_app(settings, fake_db, embedder=fake_embedder)) as client:
        response = client.get("/ready")

    assert response.status_code == 200
    assert response.json() == {"status": "ok", "checks": {"db": "ok", "embedder": "ok"}}


@pytest.mark.covers("US-001", "AC-2")
def test_ready_answers_503_in_the_envelope_naming_the_database(
    settings: Settings, fake_db: FakeDatabase, fake_embedder: FakeEmbedder
) -> None:
    """Guards the failure path: a 200, another status or an envelope without the failed check would hide the cause."""
    fake_db.up = False

    with TestClient(build_app(settings, fake_db, embedder=fake_embedder)) as client:
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


@pytest.mark.covers("US-001", "AC-2")
def test_ready_answers_200_degraded_naming_the_embedder_when_only_it_is_down(
    settings: Settings, fake_db: FakeDatabase, fake_embedder: FakeEmbedder
) -> None:
    """Keeps the database up, makes the embedding server unreachable and expects 200 with the status "degraded" and
    the embedder check "unavailable".

    Catches readiness that ignores the embedder (a client about to measure search quality would measure the weaker
    path by words without knowing it) and readiness that fails on it (a load balancer would stop sending requests the
    app can still serve on that path).
    """
    fake_embedder.up = False

    with TestClient(build_app(settings, fake_db, embedder=fake_embedder)) as client:
        response = client.get("/ready")

    seen = (response.status_code, response.json())
    expected = (200, {"status": "degraded", "checks": {"db": "ok", "embedder": "unavailable"}})
    assert seen == expected, f"expected {expected}, got {seen}"


@pytest.mark.covers("US-001", "AC-1")
def test_health_stays_200_while_the_database_is_down(settings: Settings, fake_db: FakeDatabase) -> None:
    """Guards liveness: if /health checked the database, an orchestrator would restart a healthy process."""
    fake_db.up = False

    with TestClient(build_app(settings, fake_db)) as client:
        response = client.get("/health")

    assert response.status_code == 200


@pytest.mark.covers("US-001", "AC-2")
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


@pytest.mark.covers("US-001", "AC-2")
@pytest.mark.parametrize(
    ("rerank", "body"),
    [
        (False, {"status": "ok", "checks": {"db": "ok", "embedder": "ok"}}),
        (True, {"status": "degraded", "checks": {"db": "ok", "embedder": "ok", "reranker": "unavailable"}}),
    ],
    ids=["reranking off", "reranking on"],
)
def test_ready_checks_the_reranker_only_when_searches_use_it(
    settings: Settings,
    fake_db: FakeDatabase,
    fake_embedder: FakeEmbedder,
    fake_reranker: FakeReranker,
    rerank: bool,
    body: dict[str, object],
) -> None:
    """Makes the rerank server unreachable, with reranking off and then on.

    Expects 200 "ok" without a reranker check while reranking is off, and 200 "degraded" with the reranker
    "unavailable" once it is on. Catches readiness that ignores the reranker (a run that compares results with and
    without it would measure the order without it twice) and readiness that always asks it (an app that never
    reranks would report degraded because of a server it does not use).
    """
    fake_reranker.up = False
    configured = settings.model_copy(update={"search": SearchSettings(rerank=rerank)})

    with TestClient(build_app(configured, fake_db, embedder=fake_embedder, reranker=fake_reranker)) as client:
        response = client.get("/ready")

    seen = (response.status_code, response.json())
    assert seen == (200, body), f"expected {(200, body)} with reranking {rerank}, got {seen}"
