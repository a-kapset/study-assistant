"""Reading a trace from the trace store: the reader waits until the trace is whole."""

import httpx2
from pydantic import JsonValue

from evalharness.traces import TraceReader

TRACE_ID = "8e472d7435ca66fac8ade84c55cc0b3b"


def _span(
    name: str, span_id: str, parent_id: str | None, start_ns: int = 1_000_000_000, end_ns: int = 1_000_000_000
) -> dict[str, JsonValue]:
    """One span as the store's OTLP JSON gives it: the request span has no parent id, and the start and end
    times are nanoseconds written as text."""
    span: dict[str, JsonValue] = {
        "traceId": TRACE_ID,
        "spanId": span_id,
        "name": name,
        "kind": 1,
        "status": {},
        "startTimeUnixNano": str(start_ns),
        "endTimeUnixNano": str(end_ns),
    }
    if parent_id is not None:
        span["parentSpanId"] = parent_id
    return span


def _answer(*spans: dict[str, JsonValue]) -> httpx2.Response:
    """The store's answer holding ``spans`` of one service."""
    body = {"result": {"resourceSpans": [{"scopeSpans": [{"spans": list(spans)}]}]}}
    return httpx2.Response(200, json=body)


async def test_the_reader_waits_until_the_request_span_has_arrived() -> None:
    """Serves the trace the way the store fills it: first nothing (404), then only the search span, then the whole
    trace with the request span, and expects the reader to return the whole trace.

    Catches a reader that returns the first trace it finds: the app sends spans in batches, so an early read lacks
    the spans that ended last, and a check would see a failed request without its failed stage.
    """
    answers = iter(
        [
            httpx2.Response(404, json={"error": {"httpCode": 404, "message": "No traces found"}}),
            _answer(_span("search", "s1", "r1")),
            _answer(
                _span("search", "s1", "r1"), _span("generate_answer", "g1", "r1"), _span("POST /v1/ask", "r1", None)
            ),
        ]
    )
    transport = httpx2.MockTransport(lambda _request: next(answers))

    async with TraceReader("http://jaeger.invalid", wait_s=5.0, transport=transport) as reader:
        trace = await reader.fetch(TRACE_ID, poll_s=0.0)

    names = sorted(span.name for span in trace.spans)
    assert names == ["POST /v1/ask", "generate_answer", "search"], f"the reader returned a partial trace: {names}"


async def test_a_span_says_how_long_it_lasted_in_seconds() -> None:
    """Serves a trace whose search span starts 1,794,738,997.977880313 s after 1970 and ends 27.32 ms later, the
    times written as the store writes them, and expects the span to report 0.02732 s.

    Catches a reader that takes the times in another unit (milliseconds or microseconds instead of nanoseconds) or
    reads them as floating-point numbers, which cannot hold nineteen digits: every latency read from the traces
    would then be wrong by a factor or by rounding, and nothing else would show it.
    """
    start_ns = 1_794_738_997_977_880_313
    answer = _answer(
        _span("search", "s1", "r1", start_ns, start_ns + 27_320_000),
        _span("POST /v1/search", "r1", None, start_ns - 1_000_000, start_ns + 28_000_000),
    )
    transport = httpx2.MockTransport(lambda _request: answer)

    async with TraceReader("http://jaeger.invalid", wait_s=5.0, transport=transport) as reader:
        trace = await reader.fetch(TRACE_ID, poll_s=0.0)

    (search,) = trace.named("search")
    assert search.duration_s == 0.02732, f"the search span lasted 27.32 ms, the reader says {search.duration_s} s"
