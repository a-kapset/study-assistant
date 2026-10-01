"""One raw HTTP exchange with the system under test, recorded as data."""

from typing import Literal, Self

from pydantic import BaseModel, ConfigDict, Field, JsonValue, model_validator

# Why no response arrived: the server was unreachable, did not answer in time, or anything else.
type TransportError = Literal["connect", "timeout", "other"]


class Observation(BaseModel):
    """What the harness sent and what came back, without judging it.

    Every outcome is an observation, including error statuses and requests that got no
    response at all, so a failure is reported and classified instead of stopping a run.
    All fields are required: the transport must state each value explicitly.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    request_id: str
    method: str
    url: str
    elapsed_s: float = Field(ge=0)
    # Set when a response arrived (any status); None when there was no response.
    status_code: int | None
    headers: dict[str, str]
    body_text: str
    # The body parsed as JSON, or None when it is not valid JSON.
    body_json: JsonValue | None
    # Set only when no response arrived.
    transport_error: TransportError | None

    @model_validator(mode="after")
    def _response_xor_transport_error(self) -> Self:
        """An observation is either a response or a transport failure, never both or neither.

        Whatever classifies the outcome relies on exactly one of the two signals, and a failure
        must never look like a response.

        The function passes through only those observations that unambiguously signify either
        "the server responded with this" or "the server did not respond for such-and-such reason."
        """

        got_response = self.status_code is not None
        if got_response == (self.transport_error is not None):
            raise ValueError("exactly one of status_code and transport_error must be set")
        if not got_response and (self.headers or self.body_text or self.body_json is not None):
            raise ValueError("an observation without a response cannot carry headers or a body")
        return self
