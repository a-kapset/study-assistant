from pathlib import Path
from typing import TYPE_CHECKING

import pytest
from fastapi.testclient import TestClient

from study_assistant.composition import build_app
from study_assistant.courses.models import CourseManifest
from study_assistant.embed.remote import EmbeddingServiceRequestError, EmbeddingServiceUnavailableError
from study_assistant.ingest.service import CourseIngestor, IngestConflictError, IngestReport
from study_assistant.settings import CoursesSettings, Settings

if TYPE_CHECKING:
    from conftest import FakeCorpusStore, FakeDatabase, FakeEmbedder

# The repository's own course folder: app/tests/unit/<this file> -> repository root.
REPO_COURSES = Path(__file__).resolve().parents[3] / "courses"
INGEST_DEMO = "/v1/admin/courses/demo/ingest"


def _client(
    settings: Settings,
    db: FakeDatabase,
    store: FakeCorpusStore,
    embedder: FakeEmbedder,
    source_dir: Path = REPO_COURSES,
) -> TestClient:
    """A client for an app serving the demo course, with sources read from ``source_dir``."""
    courses = CoursesSettings(manifest_dir=REPO_COURSES, source_dir=source_dir, enabled=("demo",))
    return TestClient(build_app(settings.model_copy(update={"courses": courses}), db, store, embedder))


def _key(settings: Settings) -> dict[str, str]:
    """The header that carries the configured admin key."""
    return {"X-Admin-Key": settings.admin.api_key.get_secret_value()}


@pytest.mark.covers("US-004", "AC-1")
@pytest.mark.parametrize("headers", [{}, {"X-Admin-Key": "not-the-admin-key"}, {"X-Admin-Key": ""}])
def test_ingest_without_the_right_key_is_refused_and_writes_nothing(
    settings: Settings,
    fake_db: FakeDatabase,
    fake_store: FakeCorpusStore,
    fake_embedder: FakeEmbedder,
    headers: dict[str, str],
) -> None:
    """Guards the key check: without it anyone who reaches the port could rewrite a course's corpus."""
    with _client(settings, fake_db, fake_store, fake_embedder) as client:
        response = client.post(INGEST_DEMO, headers=headers)

    assert response.status_code == 401
    assert response.json()["error"]["code"] == "unauthorized"
    assert fake_store.writers_opened == 0


@pytest.mark.covers("US-004", "AC-1")
def test_key_is_checked_before_the_course_is_looked_up(
    settings: Settings, fake_db: FakeDatabase, fake_store: FakeCorpusStore, fake_embedder: FakeEmbedder
) -> None:
    """Guards the order: a 404 without a key would tell anyone which course ids exist."""
    with _client(settings, fake_db, fake_store, fake_embedder) as client:
        response = client.post("/v1/admin/courses/no_such_course/ingest")

    assert response.status_code == 401


def test_ingest_of_a_course_not_served_answers_404(
    settings: Settings, fake_db: FakeDatabase, fake_store: FakeCorpusStore, fake_embedder: FakeEmbedder
) -> None:
    """Guards the course lookup: an unknown id must not reach the ingestor."""
    with _client(settings, fake_db, fake_store, fake_embedder) as client:
        response = client.post("/v1/admin/courses/no_such_course/ingest", headers=_key(settings))

    assert response.status_code == 404
    assert response.json()["error"]["code"] == "not_found"
    assert fake_store.writers_opened == 0


@pytest.mark.covers("US-004", "AC-2")
def test_ingest_answers_with_the_report(
    settings: Settings, fake_db: FakeDatabase, fake_store: FakeCorpusStore, fake_embedder: FakeEmbedder
) -> None:
    """Guards the contract: per-file outcomes, removed files, the stored counts and vectors, nothing else."""
    with _client(settings, fake_db, fake_store, fake_embedder) as client:
        response = client.post(INGEST_DEMO, headers=_key(settings))

    assert response.status_code == 200
    assert response.json() == {
        "course_id": "demo",
        "documents": [{"source_file": "vardolin_rules.pdf", "outcome": "added"}],
        "removed": [],
        "corpus": {"documents": 1, "sections": 4, "learning_objectives": 11, "chunks": 12},
        "embedding_space": fake_embedder.space.id,
        "vectors": 12,
    }


@pytest.mark.covers("US-004", "AC-8")
def test_source_that_cannot_be_read_answers_422_naming_the_file(
    settings: Settings, fake_db: FakeDatabase, fake_store: FakeCorpusStore, fake_embedder: FakeEmbedder, tmp_path: Path
) -> None:
    """Guards the error path: a missing source must be an error the operator can act on, never a 200."""
    with _client(settings, fake_db, fake_store, fake_embedder, source_dir=tmp_path) as client:
        response = client.post(INGEST_DEMO, headers=_key(settings))

    assert response.status_code == 422
    error = response.json()["error"]
    assert error["code"] == "ingest_failed"
    assert "vardolin_rules.pdf" in error["message"]


@pytest.mark.covers("US-004", "AC-2")
def test_course_detail_shows_the_stored_counts(
    settings: Settings, fake_db: FakeDatabase, fake_store: FakeCorpusStore, fake_embedder: FakeEmbedder
) -> None:
    """Guards the detail endpoint: counts must come from the store, zero before ingestion and filled after it."""
    with _client(settings, fake_db, fake_store, fake_embedder) as client:
        before = client.get("/v1/courses/demo").json()["corpus"]
        client.post(INGEST_DEMO, headers=_key(settings))
        after = client.get("/v1/courses/demo")

    assert before == {"documents": 0, "sections": 0, "learning_objectives": 0, "chunks": 0}
    assert after.status_code == 200
    assert after.json()["id"] == "demo"
    assert after.json()["corpus"] == {"documents": 1, "sections": 4, "learning_objectives": 11, "chunks": 12}


@pytest.mark.covers("US-004", "AC-17")
def test_course_detail_shows_the_vectors_in_the_configured_space(
    settings: Settings, fake_db: FakeDatabase, fake_store: FakeCorpusStore, fake_embedder: FakeEmbedder
) -> None:
    """Reads the course detail before an ingestion and after it with one stored vector removed, and expects the
    embedder's space with 0, then 11 vectors (the demo has 12 chunks).

    Catches a count taken from the chunks or from the ingest plan instead of the stored vectors: after a change of
    model the detail would show vectors the configured space does not have.
    """
    with _client(settings, fake_db, fake_store, fake_embedder) as client:
        before = client.get("/v1/courses/demo").json()
        client.post(INGEST_DEMO, headers=_key(settings))
        fake_store.vectors["demo"].popitem()
        after = client.get("/v1/courses/demo").json()

    space = fake_embedder.space.id
    assert (before["embedding_space"], before["vectors"]) == (space, 0)
    assert (after["embedding_space"], after["vectors"]) == (space, 11)


@pytest.mark.covers("US-003", "AC-3")
def test_course_detail_of_a_course_not_served_answers_404(
    settings: Settings, fake_db: FakeDatabase, fake_store: FakeCorpusStore, fake_embedder: FakeEmbedder
) -> None:
    """Guards the lookup: an unknown course must be a 404 envelope, not zero counts that look like an empty course."""
    with _client(settings, fake_db, fake_store, fake_embedder) as client:
        response = client.get("/v1/courses/no_such_course")

    assert response.status_code == 404
    assert response.json()["error"]["code"] == "not_found"


@pytest.mark.covers("US-004", "AC-17")
@pytest.mark.parametrize(
    ("error", "status", "code"),
    [
        pytest.param(IngestConflictError("changed meanwhile"), 409, "ingest_conflict", id="conflict"),
        pytest.param(EmbeddingServiceRequestError(413), 502, "embedding_failed", id="server-error"),
        pytest.param(EmbeddingServiceUnavailableError("ConnectError"), 503, "embedding_unavailable", id="unreachable"),
    ],
)
def test_ingest_failure_answers_with_its_own_status_and_code(
    settings: Settings,
    fake_db: FakeDatabase,
    fake_store: FakeCorpusStore,
    fake_embedder: FakeEmbedder,
    monkeypatch: pytest.MonkeyPatch,
    error: Exception,
    status: int,
    code: str,
) -> None:
    """Makes the ingestion fail with each error it can raise besides an unreadable source and expects its own status
    and code in the envelope.

    Catches a failure answered as a 500 or a 200, or one mapped to another failure's status: the operator could not
    tell a retry (409, 503) from a fault to fix (502).
    """

    async def failing_ingest(_self: CourseIngestor, _manifest: CourseManifest) -> IngestReport:
        """Fail the way the ingestion does for this case."""
        raise error

    monkeypatch.setattr(CourseIngestor, "ingest", failing_ingest)

    with _client(settings, fake_db, fake_store, fake_embedder) as client:
        response = client.post(INGEST_DEMO, headers=_key(settings))

    assert response.status_code == status
    assert response.json()["error"]["code"] == code
