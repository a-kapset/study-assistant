from pathlib import Path
from typing import TYPE_CHECKING

import pytest
from fastapi.testclient import TestClient

from study_assistant.composition import build_app
from study_assistant.settings import CoursesSettings, Settings

if TYPE_CHECKING:
    from conftest import FakeCorpusStore, FakeDatabase

# The repository's own course folder: app/tests/unit/<this file> -> repository root.
REPO_COURSES = Path(__file__).resolve().parents[3] / "courses"
INGEST_DEMO = "/v1/admin/courses/demo/ingest"


def _client(
    settings: Settings, db: FakeDatabase, store: FakeCorpusStore, source_dir: Path = REPO_COURSES
) -> TestClient:
    """A client for an app serving the demo course, with sources read from ``source_dir``."""
    courses = CoursesSettings(manifest_dir=REPO_COURSES, source_dir=source_dir, enabled=("demo",))
    return TestClient(build_app(settings.model_copy(update={"courses": courses}), db, store))


def _key(settings: Settings) -> dict[str, str]:
    """The header that carries the configured admin key."""
    return {"X-Admin-Key": settings.admin.api_key.get_secret_value()}


@pytest.mark.parametrize("headers", [{}, {"X-Admin-Key": "not-the-admin-key"}, {"X-Admin-Key": ""}])
def test_ingest_without_the_right_key_is_refused_and_writes_nothing(
    settings: Settings, fake_db: FakeDatabase, fake_store: FakeCorpusStore, headers: dict[str, str]
) -> None:
    """Guards the key check: without it anyone who reaches the port could rewrite a course's corpus."""
    with _client(settings, fake_db, fake_store) as client:
        response = client.post(INGEST_DEMO, headers=headers)

    assert response.status_code == 401
    assert response.json()["error"]["code"] == "unauthorized"
    assert fake_store.writers_opened == 0


def test_key_is_checked_before_the_course_is_looked_up(
    settings: Settings, fake_db: FakeDatabase, fake_store: FakeCorpusStore
) -> None:
    """Guards the order: a 404 without a key would tell anyone which course ids exist."""
    with _client(settings, fake_db, fake_store) as client:
        response = client.post("/v1/admin/courses/no_such_course/ingest")

    assert response.status_code == 401


def test_ingest_of_a_course_not_served_answers_404(
    settings: Settings, fake_db: FakeDatabase, fake_store: FakeCorpusStore
) -> None:
    """Guards the course lookup: an unknown id must not reach the ingestor."""
    with _client(settings, fake_db, fake_store) as client:
        response = client.post("/v1/admin/courses/no_such_course/ingest", headers=_key(settings))

    assert response.status_code == 404
    assert response.json()["error"]["code"] == "not_found"
    assert fake_store.writers_opened == 0


def test_ingest_answers_with_the_report(settings: Settings, fake_db: FakeDatabase, fake_store: FakeCorpusStore) -> None:
    """Guards the contract: per-file outcomes, removed files and the stored counts, nothing else."""
    with _client(settings, fake_db, fake_store) as client:
        response = client.post(INGEST_DEMO, headers=_key(settings))

    assert response.status_code == 200
    assert response.json() == {
        "course_id": "demo",
        "documents": [{"source_file": "vardolin_rules.pdf", "outcome": "added"}],
        "removed": [],
        "corpus": {"documents": 1, "sections": 4, "learning_objectives": 11, "chunks": 12},
    }


def test_source_that_cannot_be_read_answers_422_naming_the_file(
    settings: Settings, fake_db: FakeDatabase, fake_store: FakeCorpusStore, tmp_path: Path
) -> None:
    """Guards the error path: a missing source must be an error the operator can act on, never a 200."""
    with _client(settings, fake_db, fake_store, source_dir=tmp_path) as client:
        response = client.post(INGEST_DEMO, headers=_key(settings))

    assert response.status_code == 422
    error = response.json()["error"]
    assert error["code"] == "ingest_failed"
    assert "vardolin_rules.pdf" in error["message"]


def test_course_detail_shows_the_stored_counts(
    settings: Settings, fake_db: FakeDatabase, fake_store: FakeCorpusStore
) -> None:
    """Guards the detail endpoint: counts must come from the store, zero before ingestion and filled after it."""
    with _client(settings, fake_db, fake_store) as client:
        before = client.get("/v1/courses/demo").json()["corpus"]
        client.post(INGEST_DEMO, headers=_key(settings))
        after = client.get("/v1/courses/demo")

    assert before == {"documents": 0, "sections": 0, "learning_objectives": 0, "chunks": 0}
    assert after.status_code == 200
    assert after.json()["id"] == "demo"
    assert after.json()["corpus"] == {"documents": 1, "sections": 4, "learning_objectives": 11, "chunks": 12}


def test_course_detail_of_a_course_not_served_answers_404(
    settings: Settings, fake_db: FakeDatabase, fake_store: FakeCorpusStore
) -> None:
    """Guards the lookup: an unknown course must be a 404 envelope, not zero counts that look like an empty course."""
    with _client(settings, fake_db, fake_store) as client:
        response = client.get("/v1/courses/no_such_course")

    assert response.status_code == 404
    assert response.json()["error"]["code"] == "not_found"
