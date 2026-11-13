"""The rules of a search, on an in-memory store: the lowest score, the status, which courses can be searched, the
values a request may leave out, the reranker's order, and how each failure is answered."""

from dataclasses import replace
from pathlib import Path
from typing import TYPE_CHECKING

import pytest
from fastapi.testclient import TestClient
from opentelemetry.trace import NoOpTracer
from pydantic import ValidationError

from study_assistant.composition import build_app
from study_assistant.embed.remote import EmbeddingServiceRequestError, EmbeddingServiceUnavailableError
from study_assistant.retrieval.models import SearchStrategy
from study_assistant.retrieval.repository import Coverage, Passage
from study_assistant.retrieval.rerank import RerankServiceRequestError, RerankServiceUnavailableError
from study_assistant.retrieval.service import CourseNotSearchableError, CourseSearch
from study_assistant.settings import AskSettings, CoursesSettings, SearchSettings, Settings

from .conftest import FakeReranker, FakeSearchStore, scored_passages

if TYPE_CHECKING:
    from .conftest import FakeCorpusStore, FakeDatabase, FakeEmbedder

pytestmark = pytest.mark.anyio

# The repository's own course folder: app/tests/unit/<this file> -> repository root.
REPO_COURSES = Path(__file__).resolve().parents[3] / "courses"
SEARCH = "/v1/search"


def _client(
    settings: Settings,
    db: FakeDatabase,
    corpus: FakeCorpusStore,
    embedder: FakeEmbedder,
    store: FakeSearchStore,
    search: SearchSettings | None = None,
    reranker: FakeReranker | None = None,
) -> TestClient:
    """A client for an app serving the demo course, answering grounded questions; by default with the lowest score
    0.85 and 5 hits. ``reranker`` is used only when ``search`` turns reranking on."""
    courses = CoursesSettings(manifest_dir=REPO_COURSES, source_dir=REPO_COURSES, enabled=("demo",))
    search = search or SearchSettings(default_k=5, max_k=10, min_score=0.85)
    configured = settings.model_copy(update={"courses": courses, "search": search, "ask": AskSettings(mode="grounded")})
    return TestClient(build_app(configured, db, corpus, embedder, store, reranker=reranker))


def _named_passages(*scores: float) -> list[Passage]:
    """Passages with the given cosine scores, closest first, each with its chunk id as its text, so a fake reranker
    can score each one apart."""
    return [replace(passage, text=passage.chunk_id) for passage in scored_passages(*scores)]


@pytest.mark.covers("US-006", "AC-1")
async def test_passages_below_the_lowest_score_are_left_out_and_the_kept_ones_are_ranked_from_one(
    fake_embedder: FakeEmbedder,
) -> None:
    """Gives the search four passages scored 0.9, 0.85, 0.85 and 0.8 with the lowest score 0.85.

    Expects the first three, a score equal to the lowest one included, ranked 1, 2 and 3. Catches a search that
    keeps every passage, one that drops a passage exactly at the lowest score, and ranks that start at 0.
    """
    store = FakeSearchStore(scored_passages(0.9, 0.85, 0.85, 0.8))

    result = await CourseSearch(store, fake_embedder, NoOpTracer(), strategy="vector", window=10).search(
        "demo", "How many dice?", k=4, min_score=0.85
    )

    kept = [(hit.chunk_id, hit.rank) for hit in result.hits]
    assert kept == [("c1", 1), ("c2", 2), ("c3", 3)], f"hits at or above 0.85 should be ranked 1..3, got {kept}"


@pytest.mark.covers("US-006", "AC-2")
async def test_search_where_no_passage_reaches_the_lowest_score_is_empty(fake_embedder: FakeEmbedder) -> None:
    """Gives the search passages that all score below the lowest score.

    Expects the status "empty" and no hits. Catches a result that says "ok" with an empty list, which a caller
    could not tell from a search that found the answer missing for another reason.
    """
    store = FakeSearchStore(scored_passages(0.8, 0.7))

    result = await CourseSearch(store, fake_embedder, NoOpTracer(), strategy="vector", window=10).search(
        "demo", "Bake bread", k=5, min_score=0.85
    )

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
        await CourseSearch(store, fake_embedder, NoOpTracer(), strategy="vector", window=10).search(
            "demo", "How many dice?", k=5, min_score=0.85
        )


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


@pytest.mark.covers("US-018", "AC-1", "AC-2")
@pytest.mark.parametrize(
    ("configured", "served_by"),
    [(SearchSettings(min_score=0.85), "vector"), (SearchSettings(min_score=0.85, strategy="hybrid"), "hybrid")],
    ids=["not configured", "hybrid configured"],
)
def test_search_uses_the_configured_strategy_and_the_result_names_it(
    settings: Settings,
    fake_db: FakeDatabase,
    fake_store: FakeCorpusStore,
    fake_embedder: FakeEmbedder,
    configured: SearchSettings,
    served_by: str,
) -> None:
    """Builds the app once without a strategy in its settings and once with the hybrid strategy, and searches.

    Expects the search by meaning when nothing is configured and the hybrid search when it is, each named in the
    result. Catches an app that ignores the setting, a default that is not the search by meaning, and a result that
    names one strategy while the other one ran, which would make a comparison of the two compare one with itself.
    """
    store = FakeSearchStore(scored_passages(0.9))
    with _client(settings, fake_db, fake_store, fake_embedder, store, configured) as client:
        response = client.post(SEARCH, json={"course_id": "demo", "query": "How many dice?"})

    assert response.status_code == 200, f"the search failed: {response.json()}"
    ran = "hybrid" if store.hybrid_queries else "vector"
    named = response.json()["strategy"]
    assert (ran, named) == (served_by, served_by), f"expected {served_by} to run and be named, ran {ran}, named {named}"


@pytest.mark.covers("US-018", "AC-5")
async def test_hybrid_rankings_keep_the_same_number_of_passages_whatever_number_of_hits_is_asked(
    fake_embedder: FakeEmbedder,
) -> None:
    """Runs a hybrid search for 2 hits and one for 7, with the most hits a request may ask for set to 10.

    Expects both to rank 10 passages by meaning and by words before fusion. Catches rankings cut to the number of
    hits asked for: fusion would then see fewer passages for a small request, and its first hits could differ from
    the first hits of a larger request for the same question.
    """
    store = FakeSearchStore(scored_passages(0.9))
    search = CourseSearch(store, fake_embedder, NoOpTracer(), strategy="hybrid", window=10)

    for k in (2, 7):
        await search.search("demo", "How many dice?", k=k, min_score=0.85)

    windows = [query.window for query in store.hybrid_queries]
    assert windows == [10, 10], f"each ranking should keep 10 passages for any number of hits, kept {windows}"


@pytest.mark.covers("US-018", "AC-6")
@pytest.mark.parametrize("strategy", ["vector", "hybrid"])
async def test_the_reranker_orders_the_strategys_passages_and_the_hits_are_taken_after_it(
    strategy: SearchStrategy, fake_embedder: FakeEmbedder
) -> None:
    """Gives each strategy six passages above the lowest score, c1 closest by meaning and c6 furthest; the reranker
    scores c6 best and c3 second. Asks for 2 hits.

    Expects the hits c6 and c3, each still with its cosine score, and the result marked as reranked. Catches the
    hits cut to 2 before reranking (the reranker would only swap c1 and c2, and a passage it rates best but the
    strategy places low could never be found), the reranker's score written over the cosine (a hit could then not
    be checked against the lowest score), and reranking that runs for one strategy only.
    """
    store = FakeSearchStore(_named_passages(0.95, 0.93, 0.91, 0.89, 0.87, 0.86))
    reranker = FakeReranker({"c6": 0.9, "c3": 0.8})
    search = CourseSearch(store, fake_embedder, NoOpTracer(), strategy=strategy, window=10, reranker=reranker)

    result = await search.search("demo", "How many dice?", k=2, min_score=0.85)

    hits = [(hit.chunk_id, hit.score) for hit in result.hits]
    assert (hits, result.reranked) == ([("c6", 0.86), ("c3", 0.91)], True), f"reranked hits: {hits}, {result}"


@pytest.mark.covers("US-018", "AC-6")
@pytest.mark.parametrize("strategy", ["vector", "hybrid"])
async def test_the_reranker_gets_the_whole_window_above_the_lowest_score_whatever_number_of_hits_is_asked(
    strategy: SearchStrategy, fake_embedder: FakeEmbedder
) -> None:
    """Runs a reranked search for 1 hit and one for 3 over passages scored 0.9, 0.84 and 0.88, with the lowest score
    0.85 and the most hits a request may ask for set to 10.

    Expects the reranker to get c1 and c3 both times: every passage of the window that reaches the lowest score,
    and no other. Catches a reranker fed only the number of hits asked for (its first hit for 1 could then differ
    from its first hit for 3, so a smaller result would not be the beginning of a larger one) and a passage below
    the lowest score reaching the reranker, which could then place it among the hits.
    """
    store = FakeSearchStore(_named_passages(0.9, 0.84, 0.88))
    reranker = FakeReranker()
    search = CourseSearch(store, fake_embedder, NoOpTracer(), strategy=strategy, window=10, reranker=reranker)

    for k in (1, 3):
        await search.search("demo", "How many dice?", k=k, min_score=0.85)

    given = [texts for _query, texts in reranker.calls]
    assert given == [["c1", "c3"], ["c1", "c3"]], f"the reranker should get c1 and c3 for any k, got {given}"


@pytest.mark.covers("US-018", "AC-6")
@pytest.mark.parametrize(("rerank", "calls"), [(False, 0), (True, 1)], ids=["not configured", "configured"])
def test_reranking_is_switched_by_configuration_and_the_result_says_whether_it_ran(
    settings: Settings,
    fake_db: FakeDatabase,
    fake_store: FakeCorpusStore,
    fake_embedder: FakeEmbedder,
    fake_reranker: FakeReranker,
    rerank: bool,
    calls: int,
) -> None:
    """Builds the app with a reranker once with reranking left at its default and once with it switched on, and
    searches.

    Expects no call to the reranker and ``reranked`` false by default, one call and ``reranked`` true when it is on.
    Catches an app that ignores the setting, a default that reranks (every search would pay up to a second of CPU
    and depend on one more server), and a result that says the opposite of what ran, which would make a comparison
    with and without the reranker compare one variant with itself.
    """
    configured = SearchSettings(min_score=0.85, rerank=True) if rerank else SearchSettings(min_score=0.85)
    store = FakeSearchStore(_named_passages(0.9))
    with _client(settings, fake_db, fake_store, fake_embedder, store, configured, fake_reranker) as client:
        response = client.post(SEARCH, json={"course_id": "demo", "query": "How many dice?"})

    assert response.status_code == 200, f"the search failed: {response.json()}"
    seen = (len(fake_reranker.calls), response.json()["reranked"])
    assert seen == (calls, rerank), f"expected {calls} reranker calls and reranked={rerank}, got {seen}"


@pytest.mark.covers("US-018", "AC-7")
@pytest.mark.parametrize(
    ("path", "body"),
    [(SEARCH, {"query": "How many dice?"}), ("/v1/ask", {"question": "How many dice?"})],
    ids=["search", "ask"],
)
@pytest.mark.parametrize(
    ("error", "status", "code"),
    [
        (RerankServiceRequestError(500), 502, "rerank_failed"),
        (RerankServiceUnavailableError("ConnectError"), 503, "rerank_unavailable"),
    ],
    ids=["server error", "server unreachable"],
)
def test_a_failed_reranker_is_answered_with_its_status_and_code(
    settings: Settings,
    fake_db: FakeDatabase,
    fake_store: FakeCorpusStore,
    fake_embedder: FakeEmbedder,
    fake_reranker: FakeReranker,
    path: str,
    body: dict[str, str],
    error: Exception,
    status: int,
    code: str,
) -> None:
    """Switches reranking on and lets the rerank server answer with an error, then be unreachable, for a search and
    for a grounded question.

    Expects 502 ``rerank_failed`` and 503 ``rerank_unavailable`` in the error envelope from both endpoints. Catches a
    rerank failure escaping as a 500, which blames the app and hides that the rerank server is down, and one of the
    two endpoints answering it differently.
    """
    fake_reranker.error = error
    configured = SearchSettings(min_score=0.85, rerank=True)
    store = FakeSearchStore(_named_passages(0.9))
    with _client(settings, fake_db, fake_store, fake_embedder, store, configured, fake_reranker) as client:
        response = client.post(path, json={"course_id": "demo", **body})

    answered = (response.status_code, response.json()["error"]["code"])
    assert answered == (status, code), f"expected {(status, code)}, got {answered}"
