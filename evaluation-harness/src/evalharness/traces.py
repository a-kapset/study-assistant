"""Request traces of the system under test, read back from its trace store (Jaeger) by trace id.

The app names the trace of every response in the standard ``traceresponse`` header. The store's query API
``GET /api/v3/traces/<trace id>`` answers in the OpenTelemetry (OTLP) JSON format; Jaeger marks it stable, unlike
its older ``/api/traces/<id>``, which it calls internal and subject to change.
"""

import asyncio
import re
import time
from collections.abc import Mapping
from dataclasses import dataclass
from types import TracebackType
from typing import Literal, Self, cast

import httpx2
from pydantic import JsonValue

# version-traceid-spanid-flags, e.g. 00-8e472d7435ca66fac8ade84c55cc0b3b-b1a1658393577b26-03
_TRACERESPONSE = re.compile(r"00-([0-9a-f]{32})-[0-9a-f]{16}-[0-9a-f]{2}")

type SpanStatus = Literal["unset", "ok", "error"]

# OTLP numbers the status codes; the names are those of the OpenTelemetry specification.
_STATUSES: tuple[SpanStatus, ...] = ("unset", "ok", "error")


class TraceNotFoundError(Exception):
    """The store did not hold the whole trace within the wait: no trace, or one without its request span yet."""


@dataclass(frozen=True)
class Span:
    """One span of a trace, with the fields a check reads."""

    name: str
    span_id: str
    # None for the span that has no parent in the trace: the request span of the app.
    parent_id: str | None
    status: SpanStatus
    attributes: Mapping[str, JsonValue]


@dataclass(frozen=True)
class Trace:
    """All spans of one request, and the store's answer as text for checks that look at everything it holds."""

    spans: tuple[Span, ...]
    raw: str

    def named(self, name: str) -> list[Span]:
        """Return the spans called ``name``, in the store's order."""
        return [span for span in self.spans if span.name == name]

    def children(self, parent: Span) -> list[Span]:
        """Return the spans whose parent is ``parent``."""
        return [span for span in self.spans if span.parent_id == parent.span_id]


def trace_id_from(headers: Mapping[str, str]) -> str | None:
    """Return the trace id the ``traceresponse`` header names, or None when the header is missing or malformed."""
    match = _TRACERESPONSE.fullmatch(headers.get("traceresponse", ""))
    return match.group(1) if match else None


class TraceReader:
    """Reads traces from the store's query API; use it as an async context manager."""

    def __init__(self, base_url: str, wait_s: float, transport: httpx2.AsyncBaseTransport | None = None) -> None:
        """Point the reader at the store; ``wait_s`` is the longest wait for a whole trace. A test passes a fake
        ``transport`` instead of a running store."""
        self._client = httpx2.AsyncClient(base_url=base_url, timeout=5.0, transport=transport)
        self._wait_s = wait_s

    async def __aenter__(self) -> Self:
        """Open the connection to the store."""
        await self._client.__aenter__()
        return self

    async def __aexit__(
        self, exc_type: type[BaseException] | None, exc: BaseException | None, tb: TracebackType | None
    ) -> None:
        """Close the connection to the store."""
        await self._client.__aexit__(exc_type, exc, tb)

    async def fetch(self, trace_id: str, poll_s: float = 0.5) -> Trace:
        """Return the trace once the store holds its request span; raise ``TraceNotFoundError`` after the wait.

        The app sends spans in batches as they end, so the spans of one request can reach the store in several
        batches, and a trace read too early lacks the spans that ended last. The request span ends after all the
        others, so once it is there the trace is whole.
        """
        deadline = time.monotonic() + self._wait_s
        seen = "no trace"
        while True:
            response = await self._client.get(f"/api/v3/traces/{trace_id}")
            if response.status_code == 200:
                trace = _parse(response.text, response.json())
                if any(span.parent_id is None for span in trace.spans):
                    return trace
                seen = f"{len(trace.spans)} spans without the request span"
            elif response.status_code != 404:
                response.raise_for_status()
            if time.monotonic() >= deadline:
                raise TraceNotFoundError(f"trace {trace_id} not whole after {self._wait_s} s: {seen}")
            await asyncio.sleep(poll_s)


def _parse(raw: str, body: JsonValue) -> Trace:
    """Turn the store's OTLP JSON answer into a ``Trace``: spans are grouped by service and instrumentation scope."""
    result = cast(dict[str, dict[str, list[dict[str, JsonValue]]]], body)["result"]
    spans: list[Span] = []
    for resource in result["resourceSpans"]:
        for scope in cast(list[dict[str, JsonValue]], resource["scopeSpans"]):
            spans.extend(_span(span) for span in cast(list[dict[str, JsonValue]], scope["spans"]))
    return Trace(spans=tuple(spans), raw=raw)


def _span(span: dict[str, JsonValue]) -> Span:
    """Read one OTLP span: a missing parent marks the request span, a missing status code means "unset"."""
    status = cast(dict[str, JsonValue], span.get("status") or {})
    attributes = cast(list[dict[str, JsonValue]], span.get("attributes") or [])
    return Span(
        name=cast(str, span["name"]),
        span_id=cast(str, span["spanId"]),
        parent_id=cast(str | None, span.get("parentSpanId") or None),
        status=_STATUSES[cast(int, status.get("code", 0))],
        attributes={cast(str, item["key"]): _value(cast(dict[str, JsonValue], item["value"])) for item in attributes},
    )


def _value(value: dict[str, JsonValue]) -> JsonValue:
    """Unwrap an OTLP attribute value: a list keeps its items, any other value is returned as given."""
    if "arrayValue" in value:
        items = cast(dict[str, list[dict[str, JsonValue]]], value["arrayValue"]).get("values", [])
        return [_value(item) for item in items]
    return next(iter(value.values()), None)
