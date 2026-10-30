"""The OpenAI Responses API as the proxy sees it; the only proxy file that knows OpenAI's field names."""

from dataclasses import dataclass

from pydantic import BaseModel

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
