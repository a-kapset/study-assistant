"""The rules of a search, on an in-memory store: the lowest score, the status, which courses can be searched, the
values a request may leave out, and how each failure is answered."""

from pathlib import Path
from typing import TYPE_CHECKING

import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError

from study_assistant.composition import build_app
from study_assistant.embed.remote import EmbeddingServiceRequestError, EmbeddingServiceUnavailableError
from study_assistant.retrieval.repository import Coverage
from study_assistant.retrieval.service import CourseNotSearchableError, VectorSearch
from study_assistant.settings import CoursesSettings, SearchSettings, Settings

from .conftest import FakeSearchStore, scored_passages

if TYPE_CHECKING:
    from .conftest import FakeCorpusStore, FakeDatabase, FakeEmbedder

pytestmark = pytest.mark.anyio

# The repository's own course folder: app/tests/unit/<this file> -> repository root.
REPO_COURSES = Path(__file__).resolve().parents[3] / "courses"
SEARCH = "/v1/search"


def _client(
    settings: Settings, db: FakeDatabase, corpus: FakeCorpusStore, embedder: FakeEmbedder, store: FakeSearchStore
) -> TestClient:
    """A client for an app serving the demo course, with the lowest score 0.85 and 5 hits by default."""
    courses = CoursesSettings(manifest_dir=REPO_COURSES, source_dir=REPO_COURSES, enabled=("demo",))
    search = SearchSettings(default_k=5, max_k=10, min_score=0.85)
    configured = settings.model_copy(update={"courses": courses, "search": search})
    return TestClient(build_app(configured, db, corpus, embedder, store))


@pytest.mark.covers("US-006", "AC-1")
async def test_passages_below_the_lowest_score_are_left_out_and_the_kept_ones_are_ranked_from_one(
    fake_embedder: FakeEmbedder,
) -> None:
    """Gives the search four passages scored 0.9, 0.85, 0.85 and 0.8 with the lowest score 0.85.

    Expects the first three, a score equal to the lowest one included, ranked 1, 2 and 3. Catches a search that
    keeps every passage, one that drops a passage exactly at the lowest score, and ranks that start at 0.
    """
    store = FakeSearchStore(scored_passages(0.9, 0.85, 0.85, 0.8))

    result = await VectorSearch(store, fake_embedder).search("demo", "How many dice?", k=4, min_score=0.85)

    kept = [(hit.chunk_id, hit.rank) for hit in result.hits]
    assert kept == [("c1", 1), ("c2", 2), ("c3", 3)], f"hits at or above 0.85 should be ranked 1..3, got {kept}"


@pytest.mark.covers("US-006", "AC-2")
async def test_search_where_no_passage_reaches_the_lowest_score_is_empty(fake_embedder: FakeEmbedder) -> None:
    """Gives the search passages that all score below the lowest score.

    Expects the status "empty" and no hits. Catches a result that says "ok" with an empty list, which a caller
    could not tell from a search that found the answer missing for another reason.
    """
    store = FakeSearchStore(scored_passages(0.8, 0.7))

    result = await VectorSearch(store, fake_embedder).search("demo", "Bake bread", k=5, min_score=0.85)

    assert (result.status, result.hits) == ("empty", []), f"expected an empty result, got {result.status}"


@pytest.mark.covers("US-006", "AC-5")
@pytest.mark.parametrize(
    "coverage",
    [Coverage(chunks=0, vectors=0), Coverage(chunks=12, vectors=0), Coverage(chunks=12, vectors=11)],
    ids=["not ingested", "new space without vectors", "one chunk without a vector"],
)
async def test_course_not_fully_embedded_in_the_space_is_refused(
    fake_embedder: FakeEmbedder, coverage: Coverage
) -> None:
    """Plays the states a course is in before its first ingestion, after the configuration names a new embedding
    space, and while one chunk still lacks its vector.

    Expects ``CourseNotSearchableError`` in each. Catches a search that answers from part of a course, or from
    nothing, as if it had searched all of it.
    """
    store = FakeSearchStore(scored_passages(0.9), coverage=coverage)

    with pytest.raises(CourseNotSearchableError):
        await VectorSearch(store, fake_embedder).search("demo", "How many dice?", k=5, min_score=0.85)


@pytest.mark.covers("US-006", "AC-3")
@pytest.mark.parametrize(
    ("body", "asked_k", "applied_min_score"),
    [({}, 5, 0.85), ({"k": 2, "min_score": 0.5}, 2, 0.5)],
    ids=["left out", "given"],
)
def test_values_left_out_come_from_the_settings_and_given_values_win(
    settings: Settings,
    fake_db: FakeDatabase,
    fake_store: FakeCorpusStore,
    fake_embedder: FakeEmbedder,
    body: dict[str, float],
    asked_k: int,
    applied_min_score: float,
) -> None:
    """Sends a search without the number of hits and the lowest score, then with both.

    Expects the store to be asked for the configured 5 hits or the requested 2, and the result to report the lowest
    score really applied. Catches a request value that is ignored, and a result that reports the configured value
    when another one was used, which would make a client record the wrong setting with its numbers.
    """
    store = FakeSearchStore(scored_passages(0.9))
    with _client(settings, fake_db, fake_store, fake_embedder, store) as client:
        response = client.post(SEARCH, json={"course_id": "demo", "query": "How many dice?", **body})

    assert response.status_code == 200, f"the search failed: {response.json()}"
    assert store.asked_k == [asked_k], f"the store was asked for {store.asked_k} hits, expected {asked_k}"
    reported = response.json()["min_score"]
    assert reported == applied_min_score, f"the result reports min_score {reported}, applied {applied_min_score}"


@pytest.mark.covers("US-006", "AC-4")
def test_more_hits_than_the_configured_maximum_are_refused(
    settings: Settings, fake_db: FakeDatabase, fake_store: FakeCorpusStore, fake_embedder: FakeEmbedder
) -> None:
    """Asks for 11 hits where the configured maximum is 10.

    Expects a 422 in the error envelope that points at ``k``, and no search. Catches a missing check, which would
    let one request read a whole course.
    """
    store = FakeSearchStore(scored_passages(0.9))
    with _client(settings, fake_db, fake_store, fake_embedder, store) as client:
        response = client.post(SEARCH, json={"course_id": "demo", "query": "How many dice?", "k": 11})

    assert response.status_code == 422, f"expected 422 for k above the maximum, got {response.status_code}"
    assert response.json()["error"]["details"][0]["loc"] == ["body", "k"], f"wrong field: {response.json()}"
    assert store.asked_k == [], "the search ran although the request was refused"


@pytest.mark.covers("US-006", "AC-5")
@pytest.mark.covers("US-006", "AC-6")
@pytest.mark.parametrize(
    ("course_id", "coverage", "error", "status", "code"),
    [
        ("vardolin", Coverage(12, 12), None, 404, "not_found"),
        ("demo", Coverage(12, 0), None, 409, "course_not_searchable"),
        ("demo", Coverage(12, 12), EmbeddingServiceRequestError(500), 502, "embedding_failed"),
        ("demo", Coverage(12, 12), EmbeddingServiceUnavailableError("ConnectError"), 503, "embedding_unavailable"),
    ],
    ids=["course not served", "course not embedded", "server error", "server unreachable"],
)
def test_each_failure_is_answered_with_its_status_and_code(
    settings: Settings,
    fake_db: FakeDatabase,
    fake_store: FakeCorpusStore,
    fake_embedder: FakeEmbedder,
    course_id: str,
    coverage: Coverage,
    error: Exception | None,
    status: int,
    code: str,
) -> None:
    """Searches a course the app does not serve, a course without vectors in the configured space, and a served
    course while the embedding server answers with an error or cannot be reached.

    Expects 404, 409, 502 and 503 with their codes in the error envelope. Catches a failure answered with the wrong
    status, and an unknown course that reaches the store, where it would look like a course without vectors.
    """
    fake_embedder.error = error
    store = FakeSearchStore(scored_passages(0.9), coverage=coverage)
    with _client(settings, fake_db, fake_store, fake_embedder, store) as client:
        response = client.post(SEARCH, json={"course_id": course_id, "query": "How many dice?"})

    answered = (response.status_code, response.json()["error"]["code"])
    assert answered == (status, code), f"expected {(status, code)}, got {answered}"


def test_default_number_of_hits_above_the_maximum_is_refused() -> None:
    """Configures 20 hits by default with a maximum of 10.

    Expects the settings to be refused. Catches a configuration where every request without ``k`` gets more hits
    than any request may ask for, because the maximum is checked only for values that come with a request.
    """
    with pytest.raises(ValidationError, match="default_k"):
        SearchSettings(default_k=20, max_k=10)
