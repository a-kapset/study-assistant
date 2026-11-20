"""What each stage of one ask did, read from the ask's trace and written down as short text, so two asks can be
compared stage by stage long after the trace store has forgotten their traces.

An ask passes five stages in this order: the call to the embedding server, the search over the stored passages, the
call to the rerank server, the call to the model provider, and the writing of the answer from its reply. Each stage
is described by what its spans show: their status, the type of an error, the status codes of the HTTP calls, and for
the search its result and the ids of the passages it found. A stage that has no span in the trace is "absent", which
is a value too: an ask that skipped a stage differs from one that ran it.

The calls are told apart by where they are made, as the app's own stages enclose them: an HTTP call inside the
search goes to the embedding server, an HTTP call inside the answer step goes to the model provider, and the rerank
call is a stage of its own inside the search.
"""

from collections.abc import Sequence
from typing import Literal

from evalharness.traces import Span, Trace

type StageName = Literal["embedding", "search", "rerank", "provider", "generate"]
# The order in which an ask passes the stages: a fault of an earlier stage changes what the later ones see.
STAGES: tuple[StageName, ...] = ("embedding", "search", "rerank", "provider", "generate")

ABSENT = "absent"

# The names of the app's own stage spans.
_SEARCH_SPAN = "search"
_RERANK_SPAN = "rerank"
_GENERATE_SPAN = "generate_answer"
# Attributes the HTTP client instrumentation writes on a call, under the names the app's version of it uses.
_HTTP_METHOD = "http.method"
_HTTP_STATUS_CODE = "http.status_code"


class StageShapeError(ValueError):
    """The trace does not look like the trace of one ask, so its stages cannot be read."""


def stage_observations(trace: Trace) -> dict[StageName, str]:
    """Describe every stage of one ask's trace in a few words.

    Raises ``StageShapeError`` when the trace holds more than one search or answer step: an ask makes each at most
    once, so a second one means the trace is not of one ask, and a comparison would read the wrong span.
    """
    search = _at_most_one(trace, _SEARCH_SPAN)
    generate = _at_most_one(trace, _GENERATE_SPAN)
    rerank = _at_most_one(trace, _RERANK_SPAN)
    return {
        "embedding": _calls(_http_calls(trace, search)) if search is not None else ABSENT,
        "search": _search(search) if search is not None else ABSENT,
        "rerank": _status(rerank) if rerank is not None else ABSENT,
        "provider": _calls(_http_calls(trace, generate)) if generate is not None else ABSENT,
        "generate": _status(generate) if generate is not None else ABSENT,
    }


def _at_most_one(trace: Trace, name: str) -> Span | None:
    """Return the one span called ``name``, or None when the trace has none; more than one is an error."""
    spans = trace.named(name)
    if len(spans) > 1:
        raise StageShapeError(f"{len(spans)} spans called {name!r} in one trace, expected at most one")
    return spans[0] if spans else None


def _http_calls(trace: Trace, stage: Span | None) -> list[Span]:
    """Return the HTTP calls made directly inside ``stage``, in the store's order; database queries and the rerank
    stage are not HTTP calls of it."""
    if stage is None:
        return []
    return [span for span in trace.children(stage) if _HTTP_METHOD in span.attributes]


def _calls(calls: Sequence[Span]) -> str:
    """Describe the HTTP calls of a stage by their outcomes in order, e.g. "200" or "500 500 500"; a call that got no
    response is "error" with the type of its error; a stage that made no call is "absent"."""
    if not calls:
        return ABSENT
    return " ".join(_call(span) for span in calls)


def _call(span: Span) -> str:
    """Describe one HTTP call by its status code, or by its error when no response came."""
    code = span.attributes.get(_HTTP_STATUS_CODE)
    if code is not None:
        return str(code)
    error_type = span.attributes.get("error.type")
    return f"error:{error_type}" if error_type is not None else "error"


def _status(span: Span) -> str:
    """Describe a stage span by its status and, when it failed, the type of the error the app wrote on it."""
    error_type = span.attributes.get("error.type")
    return f"{span.status} {error_type}" if error_type is not None else span.status


def _search(span: Span) -> str:
    """Describe the search by its span status, the status of its result, the reason of a weaker result and the ids of
    the passages it found in their order; a search that raised has no result, only its error."""
    if span.status == "error":
        return _status(span)
    result = span.attributes.get("search.status")
    reason = span.attributes.get("search.degraded_reason")
    hit_ids = span.attributes.get("search.hit_ids") or []
    hits = ",".join(str(hit) for hit in hit_ids) if isinstance(hit_ids, list) else str(hit_ids)
    parts = [str(result)] + ([str(reason)] if reason is not None else []) + [f"hits={hits}"]
    return " ".join(parts)
