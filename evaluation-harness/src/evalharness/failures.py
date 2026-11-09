"""Why a request to the app failed: one label per failed request, from the app's response and the proxy's ledger.

The app answers one code for several causes (a rate limit, a provider error and an unusable reply are all
"llm_failed"), so only the model calls behind the response tell them apart. Each label belongs to one category,
and the category decides how a run counts the case: an infrastructure or app failure measured nothing about
quality and leaves the quality numbers, while a model failure is the model's answer and counts as a wrong one.
"""

from collections.abc import Sequence
from typing import Literal

from pydantic import BaseModel, ConfigDict, ValidationError

from evalharness.observation import Observation
from evalharness.proxy.ledger import LedgerLine

type FailureLabel = Literal[
    "no_response",
    "search_failed",
    "provider_unreachable",
    "rate_limited",
    "budget_exceeded",
    "provider_error",
    "provider_refused_access",
    "replay_miss",
    "unusable_reply",
    "app_error",
    "error_as_answer",
]
type FailureCategory = Literal["infrastructure", "model", "app"]

CATEGORY: dict[FailureLabel, FailureCategory] = {
    "no_response": "infrastructure",
    "search_failed": "infrastructure",
    "provider_unreachable": "infrastructure",
    "rate_limited": "infrastructure",
    "budget_exceeded": "infrastructure",
    "provider_error": "infrastructure",
    "provider_refused_access": "infrastructure",
    "replay_miss": "infrastructure",
    "unusable_reply": "model",
    "app_error": "app",
    "error_as_answer": "app",
}

# The app's codes for a search that could not run; an ask answers them the same way.
_SEARCH_CODES = frozenset({"course_not_searchable", "embedding_failed", "embedding_unavailable"})
# The provider's 429 codes for money, from its error-code guide: a spending or usage limit of the project or the
# organisation, or no credit left. "insufficient_quota" is the older form of the last one. Any other 429 (the guide
# names "slow_down"; replies also carry "rate_limit_exceeded") is a limit on the pace of calls.
_BUDGET_CODES = frozenset(
    {
        "project_spend_limit_exceeded",
        "organization_spend_limit_exceeded",
        "organization_usage_limit_exceeded",
        "credit_balance_exhausted",
        "insufficient_quota",
    }
)


class _ErrorBody(BaseModel):
    """The code of the app's error envelope."""

    model_config = ConfigDict(extra="ignore")
    code: str


class _ErrorEnvelope(BaseModel):
    """The app's error envelope, read only for its code."""

    model_config = ConfigDict(extra="ignore")
    error: _ErrorBody


def classify(observation: Observation, calls: Sequence[LedgerLine]) -> FailureLabel | None:
    """Label one request to the app from its response and the ledger lines of its model calls, oldest first.

    Returns None for a successful response whose model calls (if any) ended in a reply. A successful response whose
    last model call failed is "error_as_answer": the app has shown something that is not the model's answer.
    """
    if observation.transport_error is not None:
        return "no_response"
    last = calls[-1] if calls else None
    if observation.status_code == 200:
        return "error_as_answer" if last is not None and last.status != 200 else None
    code = _app_code(observation)
    if code in _SEARCH_CODES:
        return "search_failed"
    if code == "llm_unavailable":
        return "provider_unreachable"
    if code == "llm_failed" and last is not None:
        return _model_call_label(last)
    return "app_error"


def _app_code(observation: Observation) -> str | None:
    """Return the code of the app's error envelope; None when the body is not one."""
    try:
        return _ErrorEnvelope.model_validate(observation.body_json).error.code
    except ValidationError:
        return None


def _model_call_label(last: LedgerLine) -> FailureLabel:
    """Label a failed model step by its last call: the app stops repeating a call when that one fails too."""
    if last.source == "miss":
        return "replay_miss"
    if last.status == 200:
        return "unusable_reply"
    if last.status == 429:
        return "budget_exceeded" if last.error in _BUDGET_CODES else "rate_limited"
    if last.status in (401, 403):
        return "provider_refused_access"
    if last.status >= 500:
        return "provider_error"
    # Any other refusal of the provider (400, 404, 422) is a request the app built wrong.
    return "app_error"
