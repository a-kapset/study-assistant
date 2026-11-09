"""The OpenAI Responses API as the proxy sees it; the only proxy file that knows OpenAI's field names."""

import json
from dataclasses import dataclass

from pydantic import BaseModel, ValidationError

from evalharness.proxy.faults import FaultKind
from evalharness.proxy.prices import TokenUsage

SURFACE = "openai_responses"
PROVIDER = "openai"
ROUTE = "/v1/responses"
# Joined to the upstream address, which already ends in /v1.
UPSTREAM_PATH = "/responses"


@dataclass(frozen=True)
class ReplyFacts:
    """What the ledger needs from a successful reply."""

    model_returned: str
    usage: TokenUsage | None


class _InputDetails(BaseModel):
    """The part of the reply's input count that came from the provider's cache."""

    cached_tokens: int


class _OutputDetails(BaseModel):
    """The part of the reply's output count spent on reasoning."""

    reasoning_tokens: int


class _Usage(BaseModel):
    """The token counts of a reply."""

    input_tokens: int
    input_tokens_details: _InputDetails
    output_tokens: int
    output_tokens_details: _OutputDetails


class _Reply(BaseModel):
    """The fields of a reply the ledger reads; the rest of the body is kept as it came."""

    model: str
    usage: _Usage | None = None


class _Request(BaseModel):
    """The field of a request the ledger reads."""

    model: str


def requested_model(body: bytes) -> str:
    """Return the model a request asks for, as "provider/model", the form the price table uses."""
    return f"{PROVIDER}/{_Request.model_validate_json(body).model}"


def read_reply(body: bytes) -> ReplyFacts:
    """Read the returned model and the token counts from the body of a successful reply.

    A body of another shape raises: a changed reply format must stop the proxy's count, not turn it into zeros.
    """
    reply = _Reply.model_validate_json(body)
    if reply.usage is None:
        return ReplyFacts(model_returned=reply.model, usage=None)
    usage = reply.usage
    return ReplyFacts(
        model_returned=reply.model,
        usage=TokenUsage(
            input=usage.input_tokens,
            cached=usage.input_tokens_details.cached_tokens,
            output=usage.output_tokens,
            reasoning=usage.output_tokens_details.reasoning_tokens,
        ),
    )


class _ErrorDetail(BaseModel):
    """The part of an error reply that names the error for programs."""

    code: str | None = None


class _ErrorReply(BaseModel):
    """An error reply of the provider."""

    error: _ErrorDetail


def error_code(body: bytes) -> str | None:
    """Return the code of an error reply, such as "rate_limit_exceeded"; None when the body has none.

    The code tells a rate limit from a spending limit, which share the status 429. An error body of another shape
    gives None instead of raising: the call has failed either way, and its status is still recorded.
    """
    try:
        return _ErrorReply.model_validate_json(body).error.code
    except ValidationError:
        return None


def _error_body(message: str, error_type: str | None, code: str | None) -> bytes:
    """Build an error reply in the provider's format."""
    return json.dumps({"error": {"message": message, "type": error_type, "param": None, "code": code}}).encode()


def _reply_body(model: str, text: str) -> bytes:
    """Build a successful reply with one message of ``text``, in the shape of a recorded reply."""
    reply = {
        "id": "resp_fault",
        "object": "response",
        "created_at": 0,
        "status": "completed",
        "model": model,
        "output": [
            {
                "id": "msg_fault",
                "type": "message",
                "status": "completed",
                "role": "assistant",
                "content": [{"type": "output_text", "annotations": [], "text": text}],
            }
        ],
        "usage": {
            "input_tokens": 0,
            "input_tokens_details": {"cached_tokens": 0},
            "output_tokens": 0,
            "output_tokens_details": {"reasoning_tokens": 0},
            "total_tokens": 0,
        },
    }
    return json.dumps(reply).encode()


def fault_reply(kind: FaultKind, body: bytes) -> tuple[int, bytes]:
    """Return the status and body the provider would send for ``kind``, for the request ``body``.

    The error codes are the ones the provider documents: "rate_limit_exceeded" for too many requests and
    "project_spend_limit_exceeded" for a project that reached its spending limit. A successful fault names the model
    the request asked for, as a real reply would.
    """
    match kind:
        case "rate_limited":
            return 429, _error_body("Rate limit reached for requests", "requests", "rate_limit_exceeded")
        case "spend_limit":
            return 429, _error_body("The project reached its spending limit", None, "project_spend_limit_exceeded")
        case "server_error":
            return 500, _error_body("The server had an error while processing your request", "server_error", None)
        case "empty_output":
            return 200, _reply_body(_Request.model_validate_json(body).model, "")
        case "truncated_body":
            whole = _reply_body(_Request.model_validate_json(body).model, '{"answer": "cut off"}')
            return 200, whole[: len(whole) // 2]
