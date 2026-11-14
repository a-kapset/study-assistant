"""API tests of the app without its embedding server, on the running stack with that server's container stopped.

Every test of this module runs during one outage: the container is stopped before the first test and started
again after the last one. A request then takes much longer than usual, because the app tries the server several
times before it answers on its weaker path.
"""

from typing import cast

import pytest
from pydantic import JsonValue

from evalharness.failures import classify
from evalharness.proxy.ledger import read_ledger
from evalharness.settings import HarnessSettings
from evalharness.traces import TraceReader, trace_id_from
from evalharness.transport import SutTransport

pytestmark = pytest.mark.usefixtures("embedder_stopped")

# A question whose words are in the demo course (courses/demo/vardolin_rules.md), so a search by words finds
# passages for it, and an ask would have passages to answer from.
QUESTION = "How many barges does each player receive?"


@pytest.mark.covers("US-018", "AC-9", "AC-12")
async def test_a_search_without_the_embedding_server_says_degraded_in_its_response_and_in_its_trace(
    patient_sut: SutTransport, traces: TraceReader
) -> None:
    """What it checks: with the embedding server's container stopped, a search of the demo course answers 200
    with the status "degraded", the reason "embedder_unavailable" and passages that have no score; the search span
    of the same request carries the same status and reason and is not marked as failed.

    What it tells: a pass means the app really falls back to the search by words when the server cannot be
    reached, and says so in both places a reader looks at. A failure shows which place is wrong: an error status
    (the outage is not recognised as "cannot be reached"), the status "ok" or "empty" in the response, or a span
    that says something else than the response, so a trace of a weaker answer would look healthy or broken.

    Why it is worth having: the app's unit tests reach this path with a fake embedder that raises "unavailable"
    by itself; only a really stopped server shows that the real client reports the outage that way. It is also
    the real response the harness's own run tests imitate with a hit that has no score.
    """
    observation = await patient_sut.request("POST", "/v1/search", json={"course_id": "demo", "query": QUESTION})

    assert observation.status_code == 200, (
        f"the search did not answer on a weaker path: {observation.status_code or observation.transport_error} "
        f"{observation.body_text[:300]}"
    )
    result = cast(dict[str, JsonValue], observation.body_json)
    hits = cast(list[dict[str, JsonValue]], result["hits"])
    answered = (result["status"], result["degraded_reason"], bool(hits), all(hit["score"] is None for hit in hits))
    assert answered == ("degraded", "embedder_unavailable", True, True), (
        f"expected a degraded result for the embedder with passages and no scores, got (status, reason, passages "
        f"found, all without a score) {answered}"
    )
    trace_id = trace_id_from(observation.headers)
    assert trace_id is not None, f"the response names no trace: headers {observation.headers}"
    trace = await traces.fetch(trace_id)
    spans = [
        (span.attributes.get("search.status"), span.attributes.get("search.degraded_reason"), span.status)
        for span in trace.named("search")
    ]
    assert spans == [("degraded", "embedder_unavailable", "unset")], (
        f"the search span does not say what the response says: (status, reason, span status) {spans}"
    )


@pytest.mark.covers("US-018", "AC-11")
@pytest.mark.covers("US-016", "AC-9")
@pytest.mark.usefixtures("proxy_in_front")
async def test_an_ask_without_the_embedding_server_is_declined_without_a_model_call_and_labelled_as_an_outage(
    patient_sut: SutTransport, settings: HarnessSettings
) -> None:
    """What it checks: with the embedding server's container stopped, an ask of the demo course answers 200 and
    declines with the reason "search_degraded", the proxy's ledger has no model call for this request, and the
    harness labels the response "degraded_retrieval".

    What it tells: a pass means the app writes no answer from passages found by words only, and an answer run
    that meets this outage counts the question as a failure of the stack. A ledger line or a failed answer means
    the model was called on those passages; another reason or label means the app's response and the harness's
    rule for it no longer fit together.

    Why it is worth having: the app's unit test proves the decline with a fake search, and the harness's unit
    test labels a response built by hand; only the running stack shows that the real response is the one the
    label is read from, and only the proxy, which the app does not control, proves that no model was called.
    """
    observation = await patient_sut.request("POST", "/v1/ask", json={"course_id": "demo", "question": QUESTION})

    lines = [line for line in read_ledger(settings.proxy_ledger_file) if line.request_id == observation.request_id]
    assert observation.status_code == 200, (
        f"the ask was not declined: {observation.status_code or observation.transport_error} "
        f"{observation.body_text[:300]}; model calls {lines}"
    )
    reason = cast(dict[str, JsonValue], observation.body_json)["abstain_reason"]
    assert (reason, lines) == ("search_degraded", []), (
        f"expected the decline for a degraded search and no model call, got reason {reason!r} and calls {lines}"
    )
    label = classify(observation, lines)
    assert label == "degraded_retrieval", f"the declined ask was labelled {label!r}"
