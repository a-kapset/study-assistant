"""The ask endpoint on a fake answer model and a fake search store: an answer for a served course, and how each
failure of the model or of the search is answered."""

from pathlib import Path
from typing import TYPE_CHECKING

import httpx2
import openai
import pytest
from fastapi.testclient import TestClient

from study_assistant.composition import build_app
from study_assistant.embed.remote import EmbeddingServiceRequestError, EmbeddingServiceUnavailableError
from study_assistant.retrieval.repository import Coverage
from study_assistant.settings import AskSettings, CoursesSettings, Settings

from .conftest import FakeSearchStore, scored_passages

if TYPE_CHECKING:
    from .conftest import FakeChatModel, FakeCorpusStore, FakeDatabase, FakeEmbedder

# The repository's own course folder: app/tests/unit/<this file> -> repository root.
REPO_COURSES = Path(__file__).resolve().parents[3] / "courses"
ASK = "/v1/ask"
PROVIDER_REQUEST = httpx2.Request("POST", "http://llm.invalid/v1/responses")


def _client(settings: Settings, db: FakeDatabase, model: FakeChatModel) -> TestClient:
    """A client for an app serving the demo course in the ungrounded mode, with ``model`` writing the answers."""
    courses = CoursesSettings(manifest_dir=REPO_COURSES, source_dir=REPO_COURSES, enabled=("demo",))
    configured = settings.model_copy(update={"courses": courses, "ask": AskSettings(mode="ungrounded")})
    return TestClient(build_app(configured, db, chat_model=model))


@pytest.mark.covers("US-009", "AC-1")
def test_ask_returns_the_models_answer(
    settings: Settings, fake_db: FakeDatabase, fake_chat_model: FakeChatModel
) -> None:
    """Asks about the demo course with a model that replies "Roll two dice." and expects 200 with that answer, not
    declined and without citations.

    Catches the endpoint answering with something other than the model's text, e.g. an empty or a fixed answer,
    and a response that leaves out a field a client reads.
    """
    response = _client(settings, fake_db, fake_chat_model).post(ASK, json={"course_id": "demo", "question": "Dice?"})

    assert response.status_code == 200, f"expected 200, got {response.status_code}: {response.text}"
    expected = {"answer": "Roll two dice.", "abstained": False, "citations": [], "invalid_citations": 0}
    assert response.json() == expected, f"the answer is not the model's: {response.json()}"


@pytest.mark.covers("US-009", "AC-1")
def test_unknown_course_is_404_and_the_model_is_not_called(
    settings: Settings, fake_db: FakeDatabase, fake_chat_model: FakeChatModel
) -> None:
    """Asks about a course the app does not serve and expects 404 with no call to the model.

    Catches the course being checked after the model call, or not at all: every such question would cost a paid
    call, and the model would answer about a course nobody serves.
    """
    response = _client(settings, fake_db, fake_chat_model).post(ASK, json={"course_id": "other", "question": "Dice?"})

    assert response.status_code == 404, f"expected 404, got {response.status_code}: {response.text}"
    assert fake_chat_model.calls == [], f"the model was called {len(fake_chat_model.calls)} time(s)"


@pytest.mark.covers("US-009", "AC-3")
def test_unreachable_provider_is_503_without_an_answer(
    settings: Settings, fake_db: FakeDatabase, fake_chat_model: FakeChatModel
) -> None:
    """Lets the model fail as the client library does when the provider cannot be reached, and expects 503 with
    the code "llm_unavailable" and no answer.

    Catches the failure answered as a provider error (502), which tells a client not to retry, and the library's
    error escaping as a 500.
    """
    fake_chat_model.error = openai.APIConnectionError(request=PROVIDER_REQUEST)

    response = _client(settings, fake_db, fake_chat_model).post(ASK, json={"course_id": "demo", "question": "Dice?"})

    assert response.status_code == 503, f"expected 503, got {response.status_code}: {response.text}"
    assert response.json()["error"]["code"] == "llm_unavailable", f"wrong code: {response.json()}"


@pytest.mark.covers("US-009", "AC-3")
@pytest.mark.parametrize(
    ("error", "reply"),
    [
        (
            openai.RateLimitError("Rate limit", response=httpx2.Response(429, request=PROVIDER_REQUEST), body=None),
            "",
        ),
        (None, "   "),
    ],
    ids=["provider error status", "reply without text"],
)
def test_provider_failure_is_502_without_an_answer(
    error: Exception | None, reply: str, settings: Settings, fake_db: FakeDatabase, fake_chat_model: FakeChatModel
) -> None:
    """Lets the provider answer with an error status, or reply with spaces only, and expects 502 with the code
    "llm_failed" and no answer.

    Catches a failure answered with 200: an error status or a reply without text would reach the learner as an
    empty answer, and nothing would count it as a failure.
    """
    fake_chat_model.error = error
    fake_chat_model.reply = reply

    response = _client(settings, fake_db, fake_chat_model).post(ASK, json={"course_id": "demo", "question": "Dice?"})

    assert response.status_code == 502, f"expected 502, got {response.status_code}: {response.text}"
    assert response.json()["error"]["code"] == "llm_failed", f"wrong code: {response.json()}"


@pytest.mark.covers("US-009", "AC-10")
@pytest.mark.parametrize(
    ("coverage", "error", "status", "code"),
    [
        (Coverage(12, 0), None, 409, "course_not_searchable"),
        (Coverage(12, 12), EmbeddingServiceRequestError(500), 502, "embedding_failed"),
        (Coverage(12, 12), EmbeddingServiceUnavailableError("ConnectError"), 503, "embedding_unavailable"),
    ],
    ids=["course not embedded", "embedding server error", "embedding server unreachable"],
)
def test_search_failure_is_answered_as_search_does(
    coverage: Coverage,
    error: Exception | None,
    status: int,
    code: str,
    settings: Settings,
    fake_db: FakeDatabase,
    fake_store: FakeCorpusStore,
    fake_embedder: FakeEmbedder,
    fake_chat_model: FakeChatModel,
) -> None:
    """Asks a grounded question about a course without vectors in the configured space, and while the embedding
    server answers with an error or cannot be reached, and expects 409, 502 and 503 with the codes /v1/search uses.

    Catches a search failure escaping as a 500, which blames the app and hides that the course needs ingesting or
    the embedding server is down, and codes that differ from the search endpoint's for the same failure.
    """
    fake_embedder.error = error
    courses = CoursesSettings(manifest_dir=REPO_COURSES, source_dir=REPO_COURSES, enabled=("demo",))
    configured = settings.model_copy(update={"courses": courses, "ask": AskSettings(mode="grounded")})
    store = FakeSearchStore(scored_passages(0.9), coverage=coverage)
    app = build_app(configured, fake_db, fake_store, fake_embedder, store, chat_model=fake_chat_model)

    response = TestClient(app).post(ASK, json={"course_id": "demo", "question": "Dice?"})

    answered = (response.status_code, response.json()["error"]["code"])
    assert answered == (status, code), f"expected {(status, code)}, got {answered}"
