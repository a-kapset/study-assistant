"""Give every HTTP request an id, and answer any exception the app did not catch with a 500 error envelope."""

import re
import uuid

import structlog
from starlette.datastructures import Headers, MutableHeaders
from starlette.types import ASGIApp, Message, Receive, Scope, Send

from study_assistant.errors import ErrorBody, code_for, error_response

REQUEST_ID_HEADER = "X-Request-ID"

# Letters, digits, dot, underscore and hyphen, up to 128 characters: enough for UUIDs and readable ids such as
# "case-042-rep3", and nothing that could break a header or a log line.
_VALID_REQUEST_ID = re.compile(r"[A-Za-z0-9._-]{1,128}")


def accept_or_generate(incoming: str | None) -> str:
    """Keep a well-formed request id sent by the client; otherwise make a new UUID4."""
    if incoming is not None and _VALID_REQUEST_ID.fullmatch(incoming):
        return incoming
    return str(uuid.uuid4())


class RequestIdMiddleware:
    """Give every HTTP request an id: in the logs, in error bodies and in the X-Request-ID response header.

    It also catches any exception the app did not handle, logs it with the traceback and answers 500 in the
    common error envelope. Starlette's own 500 handler sits outside this middleware, so its response would
    not carry the request id, and the server would print a second, plain-text traceback.
    """

    def __init__(self, app: ASGIApp) -> None:
        """Wrap the next ASGI app in the chain."""
        self._app = app
        self._log: structlog.typing.FilteringBoundLogger = structlog.get_logger(__name__)

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        """Run one request with its id bound; only HTTP requests get an id, other ASGI events pass through."""
        if scope["type"] != "http":
            await self._app(scope, receive, send)
            return

        request_id = accept_or_generate(Headers(scope=scope).get(REQUEST_ID_HEADER))
        # Request.state reads this dict, so the error handlers put the same id into the body.
        scope.setdefault("state", {})["request_id"] = request_id
        response_started = False

        async def send_with_request_id(message: Message) -> None:
            """Add the request id header to the response before passing each message on."""
            nonlocal response_started
            if message["type"] == "http.response.start":
                response_started = True
                MutableHeaders(scope=message)[REQUEST_ID_HEADER] = request_id
            await send(message)

        tokens = structlog.contextvars.bind_contextvars(request_id=request_id)
        try:
            await self._app(scope, receive, send_with_request_id)
        except Exception:
            self._log.exception("unhandled_error", method=scope["method"], path=scope["path"])
            if response_started:
                # Part of the response is already sent and a second one cannot follow; let the server close it.
                raise
            error = ErrorBody(code=code_for(500), message="Internal server error")
            response = error_response(500, error, request_id)
            await response(scope, receive, send_with_request_id)
        finally:
            structlog.contextvars.reset_contextvars(**tokens)
