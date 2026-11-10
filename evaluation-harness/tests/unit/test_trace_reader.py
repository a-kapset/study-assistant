"""Reading a trace from the trace store: the reader waits until the trace is whole."""

import httpx2
from pydantic import JsonValue

from evalharness.traces import TraceReader

TRACE_ID = "8e472d7435ca66fac8ade84c55cc0b3b"


def _span(name: str, span_id: str, parent_id: str | None) -> dict[str, JsonValue]:
    """One span as the store's OTLP JSON gives it; the request span has no parent id."""
    span: dict[str, JsonValue] = {"traceId": TRACE_ID, "spanId": span_id, "name": name, "kind": 1, "status": {}}
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
