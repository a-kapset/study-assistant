"""Provider failures on request: a test marks the request id it sends to the app, the app passes the id on, and the
proxy answers that call with the failure instead of calling the provider.

So a test can show how the app and the harness handle a rate limit or a broken reply, at no cost and without
waiting for the provider to fail. Ordinary runs send plain ids and never meet a fault.
"""

import re
from typing import Literal, get_args

# "rate_limited": too many requests. "spend_limit": the provider's spending limit was reached. "server_error": the
# provider failed. "empty_output": a successful reply without any text. "truncated_body": a successful reply cut off
# in the middle of its JSON.
type FaultKind = Literal["rate_limited", "spend_limit", "server_error", "empty_output", "truncated_body"]

# "<id>.fault-rate_limited": answer every call of this request with a rate limit.
_FAULT = re.compile(r"\.fault-([a-z_]+)$")
_KINDS: frozenset[str] = frozenset(get_args(FaultKind.__value__))


def fault_suffix(kind: FaultKind) -> str:
    """Return what a test appends to a request id so that the proxy answers its calls with ``kind``."""
    return f".fault-{kind}"


def fault_of(request_id: str | None) -> FaultKind | None:
    """Return the fault a request id is marked with; None for a plain id or no id.

    An unknown kind raises: a typo in a test must not turn into a real call that passes for the fault.
    """
    if request_id is None:
        return None
    match = _FAULT.search(request_id)
    if match is None:
        return None
    kind = match.group(1)
    if kind not in _KINDS:
        raise ValueError(f"unknown fault {kind!r} in request id {request_id!r}; known: {sorted(_KINDS)}")
    return kind  # type: ignore[return-value]
