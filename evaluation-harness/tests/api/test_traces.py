"""API tests of the app's request traces on the running stack: the trace of an ask, read back from the trace store."""

import pytest
from pydantic import JsonValue

from evalharness.proxy.faults import fault_suffix
from evalharness.traces import Span, Trace, TraceReader, trace_id_from
from evalharness.transport import SutTransport

# A demo question with a recorded answer: its best passage scores above the stack's lowest score, so the app
# searches and then calls the model.
QUESTION = "How many barges does each player receive?"

# What an outgoing call of a stage is, read from the attributes its instrumentation writes.
type CallKind = str


def _call_kind(span: Span) -> CallKind:
    """Name an outgoing call by what it reached: a database query or an HTTP call, else the span's own name."""
    if "db.system" in span.attributes:
        return "database"
    if "http.method" in span.attributes:
        return "http"
    return span.name


def _stages(trace: Trace) -> dict[str, JsonValue]:
    """Describe the trace by its request span and the app's two stages: each one's status and the kinds of the
    calls made inside it."""
    request = next(span for span in trace.spans if span.parent_id is None)
    shape: dict[str, JsonValue] = {"request": request.status}
    for name in ("search", "generate_answer"):
        for stage in trace.named(name):
            calls = sorted({_call_kind(child) for child in trace.children(stage)})
            parent = "request" if stage.parent_id == request.span_id else "elsewhere"
            shape[name] = {"status": stage.status, "inside": parent, "calls": list(calls)}
    return shape


async def _ask_and_read_the_trace(sut: SutTransport, traces: TraceReader, suffix: str) -> tuple[int | None, Trace]:
    """Ask the demo question with ``suffix`` on the request id; return the status and the request's whole trace."""
    observation = await sut.request(
        "POST", "/v1/ask", json={"course_id": "demo", "question": QUESTION}, request_id_suffix=suffix
    )
    trace_id = trace_id_from(observation.headers)
    assert trace_id is not None, f"the response names no trace: headers {observation.headers}"
    return observation.status_code, await traces.fetch(trace_id)


@pytest.mark.covers("US-017", "AC-1", "AC-2")
@pytest.mark.usefixtures("proxy_in_front")
@pytest.mark.parametrize(
    ("suffix", "status", "outcome"),
    [("", 200, "unset"), (fault_suffix("server_error"), 502, "error")],
    ids=["answered", "provider-fails"],
)
async def test_an_ask_trace_shows_each_stage_with_the_outcome_of_its_work(
    suffix: str, status: int, outcome: str, sut: SutTransport, traces: TraceReader
) -> None:
    """What it checks: the trace of an ask holds the request span and, inside it, the search stage with its
    database queries and its call to the embedding server, then the answer stage with its call to the model; the
    answer stage and the request are marked as failed exactly when the model call failed. The failure is a real
    provider error (500 on every call), injected by the LLM proxy through a mark in the request id; the answered
    case is replayed from a recording.

    What it tells: a pass means a failed answer can be traced to the stage and the call that failed, and a
    healthy stage is not blamed. A failure shows the stages as they came back, so a missing stage, a stage
    outside its request, a call that is not traced or a status that does not match the outcome is named.

    Why it is worth having: the app's unit tests check the stage spans on fakes, which make no real database
    queries or HTTP calls; only the running stack shows that the real clients are traced, that the spans reach
    the store, and that a failed provider call ends as a failed stage instead of a healthy-looking trace.
    """
    answered, trace = await _ask_and_read_the_trace(sut, traces, suffix)

    assert answered == status, f"the ask should answer {status} here, got {answered}"
    expected: dict[str, JsonValue] = {
        "request": outcome,
        "search": {"status": "unset", "inside": "request", "calls": ["database", "http"]},
        "generate_answer": {"status": outcome, "inside": "request", "calls": ["http"]},
    }
    assert _stages(trace) == expected, f"the trace does not show the ask's stages as they ran: {_stages(trace)}"


@pytest.mark.covers("US-017", "AC-3")
@pytest.mark.usefixtures("proxy_in_front")
async def test_no_span_of_an_ask_holds_the_question(sut: SutTransport, traces: TraceReader) -> None:
    """What it checks: the whole trace of an answered ask, every span of the app and of the libraries it uses,
    with their attributes, events and status texts, does not contain the learner's question.

    What it tells: a pass means traces, which are kept and shown outside the request, carry no learner text from
    this path. A failure means some span records the question: a new attribute in our code, an instrumentation
    that records request bodies or query values, or an error text copied into a span.

    Why it is worth having: trace tools invite recording prompts and completions as span attributes, and the
    instrumentation libraries can record bodies and query parameters by a setting; the app's unit tests check only
    the stage spans it writes itself, while this test reads everything the store received for one real request.
    """
    answered, trace = await _ask_and_read_the_trace(sut, traces, "")

    assert answered == 200 and trace.named("generate_answer"), (
        "the ask did not reach the model, so its trace proves little"
    )
    assert QUESTION not in trace.raw, "a span of the ask's trace holds the learner's question"
