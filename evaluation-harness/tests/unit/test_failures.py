"""Unit tests of the failure labels: one prepared response of the app and the ledger lines of its model calls per
rule, never the running stack."""

from datetime import UTC, datetime

import pytest
from pydantic import JsonValue

from evalharness.failures import CATEGORY, FailureLabel, classify
from evalharness.observation import Observation, TransportError
from evalharness.proxy.ledger import LedgerLine, Source

REQUEST_ID = "evalharness_request"


def _response(status: int, code: str | None = None) -> Observation:
    """Build a response of the app with ``status``: its error envelope with ``code``, or a successful answer."""
    body: JsonValue = (
        {"status": "error", "error": {"code": code, "message": "evalharness"}, "request_id": REQUEST_ID}
        if code is not None
        else {"answer": "evalharness"}
    )
    return Observation(
        request_id=REQUEST_ID,
        method="POST",
        url="http://sut.test/v1/ask",
        elapsed_s=0.1,
        status_code=status,
        headers={},
        body_text="evalharness",
        body_json=body,
        transport_error=None,
    )


def _no_response(error: TransportError) -> Observation:
    """Build a request that got no response from the app."""
    return Observation(
        request_id=REQUEST_ID,
        method="POST",
        url="http://sut.test/v1/ask",
        elapsed_s=0.1,
        status_code=None,
        headers={},
        body_text="",
        body_json=None,
        transport_error=error,
    )


def _call(status: int, error: str | None = None, source: Source = "upstream") -> LedgerLine:
    """Build the ledger line of one model call of the request, answered with ``status`` and the provider's code."""
    return LedgerLine(
        at=datetime.now(UTC),
        request_id=REQUEST_ID,
        key="evalharness_key",
        source=source,
        status=status,
        model_requested="openai/evalharness_model",
        model_returned=None,
        input_tokens=None,
        cached_tokens=None,
        output_tokens=None,
        reasoning_tokens=None,
        latency_ms=1.0,
        cost_usd=None,
        error=error,
    )


@pytest.mark.covers("US-016", "AC-1", "AC-2", "AC-3", "AC-4")
@pytest.mark.parametrize(
    ("response", "calls", "label"),
    [
        pytest.param(_response(200), [_call(200)], None, id="answered"),
        pytest.param(_response(200), [], None, id="declined-without-a-model-call"),
        pytest.param(_response(200), [_call(429), _call(200)], None, id="answered-after-a-repeat"),
        pytest.param(_no_response("timeout"), [], "no_response", id="no-response"),
        pytest.param(_response(503, "embedding_unavailable"), [], "search_failed", id="embedder-down"),
        pytest.param(_response(409, "course_not_searchable"), [], "search_failed", id="course-not-ready"),
        pytest.param(_response(503, "llm_unavailable"), [], "provider_unreachable", id="provider-unreachable"),
        pytest.param(
            _response(502, "llm_failed"),
            [_call(429, "rate_limit_exceeded")] * 3,
            "rate_limited",
            id="rate-limit",
        ),
        pytest.param(
            _response(502, "llm_failed"),
            [_call(429, "project_spend_limit_exceeded")],
            "budget_exceeded",
            id="spend-limit",
        ),
        pytest.param(_response(502, "llm_failed"), [_call(429, "insufficient_quota")], "budget_exceeded", id="quota"),
        pytest.param(
            _response(502, "llm_failed"), [_call(429, "credit_balance_exhausted")], "budget_exceeded", id="no-credit"
        ),
        pytest.param(
            _response(502, "llm_failed"),
            [_call(429, "organization_usage_limit_exceeded")],
            "budget_exceeded",
            id="usage-limit",
        ),
        pytest.param(_response(502, "llm_failed"), [_call(429, "slow_down")], "rate_limited", id="slow-down"),
        pytest.param(_response(502, "llm_failed"), [_call(500)], "provider_error", id="provider-5xx"),
        pytest.param(_response(502, "llm_failed"), [_call(401)], "provider_refused_access", id="key-refused"),
        pytest.param(
            _response(502, "llm_failed"), [_call(404, "replay_miss", source="miss")], "replay_miss", id="replay-miss"
        ),
        pytest.param(_response(502, "llm_failed"), [_call(200)], "unusable_reply", id="reply-not-usable"),
        pytest.param(_response(502, "llm_failed"), [_call(400)], "app_error", id="request-built-wrong"),
        pytest.param(_response(502, "llm_failed"), [], "app_error", id="model-failed-without-a-call"),
        pytest.param(_response(500, "internal_server_error"), [_call(200)], "app_error", id="app-crashed"),
        pytest.param(_response(200), [_call(500)], "error_as_answer", id="error-shown-as-an-answer"),
    ],
)
def test_each_failure_gets_the_label_of_its_cause(
    response: Observation, calls: list[LedgerLine], label: FailureLabel | None
) -> None:
    """Labels one response of the app with the model calls behind it, one rule per row, from a successful answer
    to a provider failure hidden behind the app's single code "llm_failed".

    A pass means a run can say whether a failed question was the stack, the provider, the model or the app. A
    failure points to the rule of that row: a rate limit read as a spending limit sends someone to the billing page
    for nothing; an unusable reply read as an outage hides a model problem; an error shown as an answer read as a
    success hides the worst kind of app defect.
    """
    assert classify(response, calls) == label, f"expected {label!r} for {response.status_code} with {calls}"


@pytest.mark.covers("US-016", "AC-1", "AC-2", "AC-3")
def test_only_an_unusable_reply_is_a_model_failure() -> None:
    """Reads the category of every label and expects "model" for the unusable reply only.

    A pass means only a reply the model itself spoiled counts against answer quality. A failure means an outage
    label is counted as a wrong answer (lower numbers for a broken stack), or the model failure is dropped from them.
    """
    models = [label for label, category in CATEGORY.items() if category == "model"]

    assert models == ["unusable_reply"], f"labels counted as the model's answers: {models}"
