"""Unit tests of reading what each stage of an ask did from its trace: traces written by hand, never the running stack.

The attribute names and the nesting of the spans are those of the app's tracing as read in its code; the api test
of the stages confirms them on a real trace.
"""

import pytest
from pydantic import JsonValue

from evalharness.stages import StageName, StageShapeError, stage_observations
from evalharness.traces import Span, SpanStatus, Trace


def _span(
    name: str, span_id: str, parent_id: str | None, status: SpanStatus = "unset", **attributes: JsonValue
) -> Span:
    """Build one span with ``attributes``, given by their names with "__" for each dot ("search__status")."""
    named = {key.replace("__", "."): value for key, value in attributes.items()}
    return Span(name=name, span_id=span_id, parent_id=parent_id, status=status, duration_s=0.01, attributes=named)


def _http(
    span_id: str, parent_id: str, code: int | None, status: SpanStatus = "unset", **attributes: JsonValue
) -> Span:
    """Build one HTTP call made inside ``parent_id``; without ``code`` no response came."""
    if code is not None:
        attributes["http__status_code"] = code
    return _span("POST", span_id, parent_id, status, http__method="POST", **attributes)


_REQUEST = _span("POST /v1/ask", "r1", None)
_HITS: list[JsonValue] = ["evalharness_c1", "evalharness_c2"]

# A grounded ask with the reranker on: the embedding call and a database query inside the search, the rerank stage
# with its own HTTP call, and the answer step whose model call failed twice before it got a reply.
_GROUNDED = Trace(
    spans=(
        _REQUEST,
        _span("search", "s1", "r1", search__status="ok", search__hit_ids=_HITS),
        _span("SELECT", "d1", "s1", db__system="postgresql"),
        _http("h1", "s1", 200),
        _span("rerank", "k1", "s1"),
        _http("h2", "k1", 200),
        _span("generate_answer", "g1", "r1"),
        _http("p1", "g1", 500, "error"),
        _http("p2", "g1", 500, "error"),
        _http("p3", "g1", 200),
    ),
    raw="",
)
# The embedding server could not be reached: its call got no response, and the search answered by words only.
_NO_EMBEDDER = Trace(
    spans=(
        _REQUEST,
        _span("search", "s1", "r1", search__status="degraded", search__degraded_reason="embedder_unavailable"),
        _http("h1", "s1", None, "error", error__type="ConnectError"),
    ),
    raw="",
)
# The answer step without a search (the app answers from what the model knows), and its model call failed.
_UNGROUNDED = Trace(
    spans=(
        _REQUEST,
        _span("generate_answer", "g1", "r1", "error", error__type="EmptyAnswerError"),
        _http("p1", "g1", 200),
    ),
    raw="",
)


@pytest.mark.covers("US-019", "AC-9")
@pytest.mark.parametrize(
    ("trace", "expected"),
    [
        pytest.param(
            _GROUNDED,
            {
                "embedding": "200",
                "search": "ok hits=evalharness_c1,evalharness_c2",
                "rerank": "unset",
                "provider": "500 500 200",
                "generate": "unset",
            },
            id="grounded-with-rerank-and-retries",
        ),
        pytest.param(
            _NO_EMBEDDER,
            {
                "embedding": "error:ConnectError",
                "search": "degraded embedder_unavailable hits=",
                "rerank": "absent",
                "provider": "absent",
                "generate": "absent",
            },
            id="embedder-unreachable",
        ),
        pytest.param(
            _UNGROUNDED,
            {
                "embedding": "absent",
                "search": "absent",
                "rerank": "absent",
                "provider": "200",
                "generate": "error EmptyAnswerError",
            },
            id="no-search-and-a-failed-answer-step",
        ),
    ],
)
def test_a_trace_becomes_what_each_stage_of_the_ask_did(trace: Trace, expected: dict[StageName, str]) -> None:
    """What it checks: three traces written by hand, as the app's tracing shapes them, are each turned into the five
    stage descriptions: the HTTP calls are given to the stage whose span encloses them, a call without a response
    shows its error, the search shows its result and its hits, and a stage with no span is "absent".

    What it tells: a pass means the record of an ask says what each stage did, so two asks can be compared stage by
    stage after the trace store has forgotten them. A failure names the stage read wrongly.

    Why it is worth having: the attribution of every mutant rests on these descriptions. A retry of the model call
    read as a call to the embedding server would blame the wrong part; a missing search read as a healthy one would
    hide that the app skipped it; hits left out would make a search that found other passages look unchanged.
    """
    seen = stage_observations(trace)

    assert seen == expected, f"the stages were read as {seen}"


def test_a_trace_with_two_searches_is_not_read_as_one_ask() -> None:
    """What it checks: a trace holding two search spans is refused with StageShapeError.

    What it tells: a pass means a trace that is not of one ask never gives stage descriptions; a failure means one
    of the two searches would be compared silently.

    Why it is worth having: the response names its trace by id, and a wrong id or a reused one would give the
    trace of other requests; comparing one of their searches would point at a stage the ask never passed.
    """
    trace = Trace(spans=(_REQUEST, _span("search", "s1", "r1"), _span("search", "s2", "r1")), raw="")

    with pytest.raises(StageShapeError):
        stage_observations(trace)
