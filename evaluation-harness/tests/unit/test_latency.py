"""Unit tests of a latency run: traces written by hand, a fake app and a fake trace store, never the running stack."""

import json
from collections.abc import Mapping

import httpx2
import pytest
from pydantic import HttpUrl, JsonValue

from evalharness.datasets.retrieval_gold import GoldCase, GoldDeclaration, GoldSet
from evalharness.evaluation.latency import (
    LatencyRecord,
    StageSummary,
    TraceShapeError,
    budget_verdict,
    percentile,
    run,
    stage_times,
)
from evalharness.settings import HarnessSettings
from evalharness.traces import Span, Trace, TraceReader
from evalharness.transport import SutTransport

COURSE = "evalharness_course"


def _span(name: str, span_id: str, parent_id: str | None, seconds: float, **attributes: JsonValue) -> Span:
    """Build one span that lasted ``seconds`` and carries ``attributes`` (written with "_" for the dots)."""
    named = {key.replace("_", "."): value for key, value in attributes.items()}
    return Span(name=name, span_id=span_id, parent_id=parent_id, status="unset", duration_s=seconds, attributes=named)


def _search_trace(*calls: Span) -> Trace:
    """Build the trace of one search: a request of 0.050 s, a search stage of 0.040 s inside it, and ``calls``."""
    request = _span("POST /v1/search", "r1", None, 0.050)
    search = _span("search", "s1", "r1", 0.040)
    return Trace(spans=(request, search, *calls), raw="")


def test_the_times_of_a_search_are_read_per_stage_and_calls_of_one_kind_add_up() -> None:
    """Reads a trace in which the search makes two database queries (2 ms and 6 ms), one call to the embedding
    server (11 ms) and one call to the reranker (20 ms, with its own HTTP call inside), and expects the request,
    the search and the three kinds of call with the two queries added.

    Catches a reader that keeps only the last call of a kind, counts the reranker's inner HTTP call as an
    embedding call, or swaps the request with the search: the table would then name the wrong stage as the slow one.
    """
    trace = _search_trace(
        _span("SELECT", "d1", "s1", 0.002, db_system="postgresql"),
        _span("SELECT", "d2", "s1", 0.006, db_system="postgresql"),
        _span("POST", "h1", "s1", 0.011, http_method="POST"),
        _span("rerank", "k1", "s1", 0.020),
        _span("POST", "h2", "k1", 0.019, http_method="POST"),
    )

    times = stage_times(trace)

    expected = {"request": 0.050, "search": 0.040, "database": 0.008, "embedder": 0.011, "rerank": 0.020}
    assert times == pytest.approx(expected), f"the stage times were read as {times}"


def test_a_call_of_an_unknown_kind_inside_the_search_is_an_error_not_a_missing_time() -> None:
    """Reads a trace in which the search makes a call that is neither a database query, the reranker nor an HTTP
    call, and expects an error that names the call.

    Catches a reader that leaves such a call out: after an update of a tracing library that names its attributes
    differently, the embedding stage would read as zero and the table would still look complete.
    """
    trace = _search_trace(_span("POST", "h1", "s1", 0.011, http_request_method="POST"))

    with pytest.raises(TraceShapeError) as raised:
        stage_times(trace)

    assert "'POST'" in str(raised.value), f"the error does not name the call: {raised.value}"


def test_a_percentile_is_the_value_at_its_place_in_the_sorted_times() -> None:
    """Takes the times 1, 2, …, 21 s in mixed order and expects the 50th percentile 11 s and the 95th 20 s: the
    places (21 - 1) * 0.50 = 10 and (21 - 1) * 0.95 = 19, counted from 0.

    Catches another definition slipping in (for example one that returns 20.9 s here), an unsorted input being read
    by place, and the two percentiles being swapped: the reference describes this definition, and a budget is
    judged by the number it gives.
    """
    times = [float(second) for second in (5, 21, 1, 13, 9, 17, 3, 20, 7, 11, 15, 19, 2, 6, 10, 14, 18, 4, 8, 12, 16)]

    found = (percentile(times, 50), percentile(times, 95))

    assert found == pytest.approx((11.0, 20.0)), f"expected 11 s and 20 s, got {found}"


@pytest.mark.parametrize(
    ("search_p95", "verdict"), [(0.999, "within budget"), (1.0, "within budget"), (1.001, "over budget")]
)
def test_the_budget_is_judged_by_the_95th_percentile_of_the_search_stage(search_p95: float, verdict: str) -> None:
    """Judges a summary whose search stage has the given 95th percentile, a median of 0.1 s and a slowest search
    of 5 s, next to a request stage that is over the budget everywhere, against a budget of 1 s.

    Catches a verdict read from the median or the slowest request, or from the request stage instead of the search
    stage: the budget is set for the search, and a learner's wait for it is judged by its tail.
    """
    summary = {
        "request": StageSummary(n=20, p50=2.0, p95=3.0, max=6.0),
        "search": StageSummary(n=20, p50=0.1, p95=search_p95, max=5.0),
    }

    found = budget_verdict(summary, 1.0)

    assert found == verdict, f"a search p95 of {search_p95} s against 1 s should be {verdict!r}, got {found!r}"


def _case(case_id: str) -> GoldCase:
    """Build one gold case of set A whose question holds its id."""
    return GoldCase(
        case_id=case_id,
        query=f"Question {case_id}?",
        gold_section="1.1",
        exam_set="A",
        chapter="1",
        knowledge_level="K2",
        origin="sample_exam",
        held_out=False,
    )


def _gold(*cases: GoldCase) -> GoldSet:
    """Build a gold set of ``cases`` with recall at 5 on set A as its main result."""
    declaration = GoldDeclaration.model_validate(
        {
            "course_id": COURSE,
            "bank_file": "bank.jsonl",
            "bank_sha256": "0" * 64,
            "objective_section": r"FL-(?P<section>(?P<chapter>\d+)\.\d+)\.\d+",
            "held_out_sets": ["D"],
            "primary": {"metric": "recall", "k": 5, "sets": ["A"], "target": 0.8},
        }
    )
    return GoldSet(declaration=declaration, cases=cases, sha256="1" * 64)


class _Stack:
    """A fake app and a fake trace store: every search gets a trace of its own, and the search stage of the
    n-th search lasts n hundredths of a second, so each request's time says which request it was."""

    def __init__(self, failing_question: str) -> None:
        """Keep the question the app answers with an error; nothing was searched yet."""
        self._failing_question = failing_question
        self.searches = 0
        self._seconds_by_trace: dict[str, float] = {}

    def app(self, request: httpx2.Request) -> httpx2.Response:
        """Answer the app's settings, readiness and search requests."""
        if request.url.path == "/v1/config":
            return httpx2.Response(200, json={"settings": {"git_commit": "evalharness_app_commit"}})
        if request.url.path == "/ready":
            return httpx2.Response(200, json={"status": "ok", "checks": {"db": "ok", "embedder": "ok"}})
        self.searches += 1
        if json.loads(request.content)["query"] == self._failing_question:
            return httpx2.Response(502, json={"error": {"code": "embedding_failed", "message": "broken"}})
        trace_id = f"{self.searches:032x}"
        self._seconds_by_trace[trace_id] = self.searches / 100
        body = {"embedding_space": "s", "k": 5, "min_score": 0.83, "strategy": "vector", "reranked": False, "hits": []}
        return httpx2.Response(200, json=body, headers={"traceresponse": f"00-{trace_id}-{'1' * 16}-01"})

    def trace_store(self, request: httpx2.Request) -> httpx2.Response:
        """Answer the store's trace request with the request span and the search span of that trace."""
        seconds = self._seconds_by_trace[request.url.path.rsplit("/", 1)[1]]
        spans = [_otlp("POST /v1/search", "r1", None, seconds + 0.001), _otlp("search", "s1", "r1", seconds)]
        return httpx2.Response(200, json={"result": {"resourceSpans": [{"scopeSpans": [{"spans": spans}]}]}})


def _otlp(name: str, span_id: str, parent_id: str | None, seconds: float) -> Mapping[str, JsonValue]:
    """One span as the store's OTLP JSON gives it, lasting ``seconds``."""
    span: dict[str, JsonValue] = {
        "spanId": span_id,
        "name": name,
        "status": {},
        "startTimeUnixNano": "1000000000",
        "endTimeUnixNano": str(1_000_000_000 + round(seconds * 1e9)),
    }
    if parent_id is not None:
        span["parentSpanId"] = parent_id
    return span


async def _run(stack: _Stack, gold: GoldSet, warm_up: int) -> LatencyRecord:
    """Make a latency run of ``gold`` against the fake app and trace store."""
    settings = HarnessSettings(sut_base_url=HttpUrl("http://sut.test"))
    async with (
        SutTransport(settings, transport=httpx2.MockTransport(stack.app)) as sut,
        TraceReader("http://jaeger.invalid", wait_s=1.0, transport=httpx2.MockTransport(stack.trace_store)) as traces,
    ):
        return await run(
            gold,
            gold.cases,
            sut,
            traces,
            declaration_file="datasets/evalharness_course/gold.toml",
            harness_commit="evalharness_commit",
            harness_dirty=False,
            client_machine="evalharness_machine",
            warm_up=warm_up,
        )


async def test_warm_up_requests_and_failed_searches_stay_out_of_the_summary() -> None:
    """Runs four questions with a warm-up of two against a fake app that fails the third question and whose n-th
    search lasts n hundredths of a second. Six searches are sent: 1 and 2 are the warm-up, 3 to 6 are measured, and
    search 5 (the third question) fails. Expects the search stage summed over searches 3, 4 and 6 only.

    Catches a summary that counts the warm-up requests (the cold first requests would stretch the tail), one that
    counts a failed search as a time of zero, and a run that takes a trace other than the request's own: the
    slowest measured search must be the sixth, 0.06 s.
    """
    stack = _Stack(failing_question="Question A-3?")
    gold = _gold(_case("A-1"), _case("A-2"), _case("A-3"), _case("A-4"))

    record = await _run(stack, gold, warm_up=2)

    search = record.summary["search"]
    found = (stack.searches, search.n, search.p50, search.max)
    assert found == pytest.approx((6, 3, 0.04, 0.06)), (
        f"expected 6 searches sent and the search stage over searches 3, 4 and 6 (n 3, p50 0.04, max 0.06), "
        f"got searches, n, p50, max = {found}"
    )
