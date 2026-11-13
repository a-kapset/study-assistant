"""The request trace: a span per stage of our own code whose status says whether the stage failed, no error text in
it, and a response that names its trace. Spans go to an in-memory exporter instead of a trace backend."""

from collections.abc import Iterator
from dataclasses import replace
from pathlib import Path
from typing import Literal

import httpx2
import pytest
from fastapi.testclient import TestClient
from opentelemetry.sdk.trace import ReadableSpan, TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from opentelemetry.trace import StatusCode

from study_assistant.answer.service import GENERATE_SPAN, Answerer
from study_assistant.composition import build_app
from study_assistant.courses.registry import load_course
from study_assistant.llm.providers import LlmError, LlmUnavailableError, build_chat_model
from study_assistant.retrieval.repository import Coverage
from study_assistant.retrieval.service import RERANK_SPAN, CourseNotSearchableError, CourseSearch
from study_assistant.settings import AskSettings, SearchSettings, Settings
from study_assistant.tracing import TRACER_NAME, stage_span

from .conftest import FakeChatModel, FakeDatabase, FakeEmbedder, FakeReranker, FakeSearchStore, scored_passages

# The repository's own course folder: app/tests/unit/<this file> -> repository root.
REPO_COURSES = Path(__file__).resolve().parents[3] / "courses"
type AskMode = Literal["grounded", "ungrounded"]
SEARCH = SearchSettings(default_k=3, max_k=10, min_score=0.85)
# Text that must never be written into a span: stands in for a learner's question carried by an exception.
CANARY = "canary-question-not-for-traces"


class Traces:
    """A tracer provider whose finished spans are kept in memory, read back by name."""

    def __init__(self) -> None:
        """Send every finished span at once to an in-memory exporter."""
        self.exporter = InMemorySpanExporter()
        self.provider = TracerProvider()
        self.provider.add_span_processor(SimpleSpanProcessor(self.exporter))

    def spans(self) -> list[ReadableSpan]:
        """Return every finished span, in the order the spans ended."""
        return list(self.exporter.get_finished_spans())

    def named(self, name: str) -> ReadableSpan:
        """Return the one finished span called ``name``; fail, listing the names seen, when there is not exactly one."""
        found = [span for span in self.spans() if span.name == name]
        assert len(found) == 1, f"expected one span {name!r}, got {len(found)} in {[s.name for s in self.spans()]}"
        return found[0]


@pytest.fixture
def traces() -> Iterator[Traces]:
    """A fresh in-memory trace store for one test, shut down after it."""
    store = Traces()
    yield store
    store.provider.shutdown()


# What the search span says about a search besides its status.
SEARCH_SPAN_ATTRIBUTES = ("search.strategy", "search.hybrid.by_meaning", "search.hybrid.by_words", "search.hit_ids")


def _failing_provider(_request: httpx2.Request) -> httpx2.Response:
    """Answer every model call the way a provider rejects a request, which the client does not repeat."""
    return httpx2.Response(400, json={"error": {"message": "bad request", "type": "invalid_request_error"}})


def _answerer(mode: AskMode, settings: Settings, embedder: FakeEmbedder, traces: Traces) -> Answerer:
    """An answerer in ``mode`` whose search finds a passage and whose every model call fails, traced into ``traces``.

    The grounded mode uses the real client library, because only it gives the structured reply that mode asks for;
    the ungrounded mode uses the fake model, set to fail as an unreachable provider does.
    """
    tracer = traces.provider.get_tracer(TRACER_NAME)
    search = CourseSearch(FakeSearchStore(scored_passages(0.9)), embedder, tracer, strategy="vector", window=10)
    if mode == "grounded":
        client = httpx2.AsyncClient(transport=httpx2.MockTransport(_failing_provider))
        model = build_chat_model(settings.llm.answer, settings, client)
    else:
        model = FakeChatModel(error=LlmUnavailableError("ConnectError"))
    return Answerer(model, search, AskSettings(mode=mode), SEARCH, tracer)


@pytest.mark.covers("US-017", "AC-2")
@pytest.mark.anyio
async def test_a_failed_search_marks_its_span_as_failed(fake_embedder: FakeEmbedder, traces: Traces) -> None:
    """Searches a course that has no stored chunks, which the search refuses, and expects the search span to end with
    the status ERROR and the error's class.

    Catches a stage span that ends as if the stage worked while the stage raised: the trace would then show a
    healthy search for a request that failed in it, and a reader of the trace would look for the fault elsewhere.
    """
    store = FakeSearchStore(scored_passages(0.9), coverage=Coverage(chunks=0, vectors=0))
    search = CourseSearch(store, fake_embedder, traces.provider.get_tracer(TRACER_NAME), strategy="vector", window=10)

    with pytest.raises(CourseNotSearchableError):
        await search.search("demo", "How many dice?", k=3, min_score=0.85)

    span = traces.named("search")
    seen = (span.status.status_code, span.attributes.get("error.type") if span.attributes else None)
    assert seen == (StatusCode.ERROR, "CourseNotSearchableError"), f"the failed search span ended as {seen}"


@pytest.mark.covers("US-018", "AC-2")
@pytest.mark.anyio
async def test_the_search_span_names_the_strategy_what_each_ranking_found_and_the_hits(
    fake_embedder: FakeEmbedder, traces: Traces
) -> None:
    """Runs a hybrid search over three passages, two of which reach the lowest score, and reads its span.

    Expects the strategy "hybrid", two passages in each ranking and the two hit ids in rank order. Catches a span
    that does not say which strategy ran, so a trace of a hybrid search could not be told from one by meaning; and
    missing ranking counts, which are the only sign of a ranking by words that finds nothing.
    """
    store = FakeSearchStore(scored_passages(0.9, 0.87, 0.5))
    search = CourseSearch(store, fake_embedder, traces.provider.get_tracer(TRACER_NAME), strategy="hybrid", window=10)

    await search.search("demo", "How many dice?", k=5, min_score=0.85)

    attributes = dict(traces.named("search").attributes or {})
    seen = {name: attributes.get(name) for name in SEARCH_SPAN_ATTRIBUTES}
    expected = {
        "search.strategy": "hybrid",
        "search.hybrid.by_meaning": 2,
        "search.hybrid.by_words": 2,
        "search.hit_ids": ("c1", "c2"),
    }
    assert seen == expected, f"the search span should carry {expected}, carries {seen}"


@pytest.mark.covers("US-017", "AC-2")
@pytest.mark.anyio
@pytest.mark.parametrize("mode", ["grounded", "ungrounded"])
async def test_a_failed_model_call_marks_the_answer_span_as_failed(
    mode: AskMode, settings: Settings, fake_embedder: FakeEmbedder, traces: Traces
) -> None:
    """Asks a question whose model call fails, in each mode, and expects the answer-generation span to end with the
    status ERROR, while a search span, when the mode searches, does not.

    Catches the model call of either mode falling outside its span (each mode calls the model in its own place),
    which would leave a failed request without the failed stage, and a failure blamed on the search.
    """
    with pytest.raises(LlmError) as raised:
        await _answerer(mode, settings, fake_embedder, traces).answer(load_course(REPO_COURSES, "demo"), "Dice?")

    statuses = {span.name: span.status.status_code for span in traces.spans()}
    expected = {GENERATE_SPAN: StatusCode.ERROR}
    if mode == "grounded":
        expected["search"] = StatusCode.UNSET
    stage_statuses = {name: status for name, status in statuses.items() if name in ("search", GENERATE_SPAN)}
    assert stage_statuses == expected, f"stage spans after a failed model call ({raised.value!r}): {stage_statuses}"


@pytest.mark.covers("US-017", "AC-3")
def test_a_failed_stage_keeps_the_error_text_out_of_its_span(traces: Traces) -> None:
    """Runs a stage that raises an error whose message holds the learner's question, and expects the question
    nowhere in the span: not in its status text, its attributes or its events.

    Catches recording the exception the default way, which copies its message and every chained cause, such as a
    rejected model reply, into a trace that is kept and shown outside the request.
    """
    tracer = traces.provider.get_tracer(TRACER_NAME)

    with pytest.raises(ValueError, match=CANARY), stage_span(tracer, "stage"):
        raise ValueError(CANARY)

    span = traces.named("stage")
    written = [span.status.description or "", *map(str, (span.attributes or {}).values())]
    written += [f"{event.name} {dict(event.attributes or {})}" for event in span.events]
    assert not [text for text in written if CANARY in text], f"the error text reached the span: {written}"


@pytest.mark.covers("US-017", "AC-4")
def test_a_response_names_the_trace_of_its_request(settings: Settings, fake_db: FakeDatabase, traces: Traces) -> None:
    """Calls the app and expects the standard traceresponse header to carry the trace id of that request's span.

    Catches the app answering without the header, or with an id of another trace: a client could then not fetch
    the trace of the answer it got.
    """
    with TestClient(build_app(settings, fake_db, tracer_provider=traces.provider)) as client:
        response = client.get("/health")

    request_span = traces.named("GET /health")
    trace_id = f"{request_span.context.trace_id:032x}"
    header = response.headers.get("traceresponse", "")
    assert header.split("-")[1:2] == [trace_id], f"traceresponse {header!r} does not name trace {trace_id}"


@pytest.mark.covers("US-017", "AC-4")
def test_the_request_span_keeps_the_request_id(settings: Settings, fake_db: FakeDatabase, traces: Traces) -> None:
    """Calls the app with a request id and expects the request's span to hold that id.

    Catches losing the join between a log line, which carries the request id, and the trace of the same request.
    """
    with TestClient(build_app(settings, fake_db, tracer_provider=traces.provider)) as client:
        client.get("/health", headers={"X-Request-ID": "case-007"})

    attributes = dict(traces.named("GET /health").attributes or {})
    kept = attributes.get("http.response.header.x_request_id")
    assert kept == ("case-007",), f"the request span holds request id {kept!r}; attributes: {attributes}"


@pytest.mark.covers("US-018", "AC-8")
@pytest.mark.anyio
async def test_the_search_span_keeps_the_order_before_reranking_and_the_reranker_has_its_own_span(
    fake_embedder: FakeEmbedder, traces: Traces
) -> None:
    """Runs a reranked search over three passages above the lowest score; the reranker puts the last one first.

    Expects the search span to say that rerank ran, to list the candidates c1, c2, c3 in the strategy's order and
    the hits c3, c1, c2 in the reranker's order, and a rerank span inside the search span. Catches candidates written
    after the reordering (the trace could then not show that a passage was found and placed too low by the reranker,
    rather than never found), a span that does not say rerank ran, and a reranker call outside a span of its own,
    whose time could then not be told apart from the rest of the search.
    """
    passages = [replace(passage, text=passage.chunk_id) for passage in scored_passages(0.9, 0.88, 0.87)]
    tracer = traces.provider.get_tracer(TRACER_NAME)
    reranker = FakeReranker({"c3": 0.9, "c1": 0.5, "c2": 0.1})
    search = CourseSearch(
        FakeSearchStore(passages), fake_embedder, tracer, strategy="vector", window=10, reranker=reranker
    )

    await search.search("demo", "How many dice?", k=5, min_score=0.85)

    search_span = traces.named("search")
    attributes = dict(search_span.attributes or {})
    seen = {name: attributes.get(name) for name in ("search.rerank", "search.rerank.candidate_ids", "search.hit_ids")}
    expected = {
        "search.rerank": True,
        "search.rerank.candidate_ids": ("c1", "c2", "c3"),
        "search.hit_ids": ("c3", "c1", "c2"),
    }
    assert seen == expected, f"the search span should carry {expected}, carries {seen}"
    parent = traces.named(RERANK_SPAN).parent
    assert parent is not None and parent.span_id == search_span.context.span_id, "the rerank span is not in the search"
