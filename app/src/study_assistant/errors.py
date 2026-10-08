"""One error format for every failed request: the status code says what kind, the body says what and which request."""

from collections.abc import Mapping
from http import HTTPStatus
from typing import Literal

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict
from starlette.exceptions import HTTPException as StarletteHTTPException


class ErrorDetail(BaseModel):
    """One thing that went wrong: where it is (a request field, or a readiness check) and what is wrong with it."""

    model_config = ConfigDict(extra="forbid")
    loc: list[str | int]
    msg: str


class ErrorBody(BaseModel):
    """What went wrong: a stable code for programs and a message for people."""

    model_config = ConfigDict(extra="forbid")
    code: str
    message: str
    # Validation errors list the invalid parts and readiness errors the failing checks; others leave it out.
    details: list[ErrorDetail] | None = None


class ErrorEnvelope(BaseModel):
    """The body of every error response, so a client never has to guess whether a call failed."""

    model_config = ConfigDict(extra="forbid")
    status: Literal["error"] = "error"
    error: ErrorBody
    request_id: str


def error_response(
    status_code: int, error: ErrorBody, request_id: str, headers: Mapping[str, str] | None = None
) -> JSONResponse:
    """Build an error response in the common envelope; fields that are not set are left out of the body."""
    envelope = ErrorEnvelope(error=error, request_id=request_id)
    return JSONResponse(envelope.model_dump(mode="json", exclude_none=True), status_code=status_code, headers=headers)


def code_for(status_code: int) -> str:
    """Name a status as a code: 404 -> "not_found", 405 -> "method_not_allowed"; an unknown status -> "http_error"."""
    try:
        phrase = HTTPStatus(status_code).phrase
    except ValueError:
        return "http_error"
    return phrase.lower().replace(" ", "_").replace("-", "_")


def install_error_handlers(app: FastAPI) -> None:
    """Make HTTP errors and invalid requests answer in the common envelope instead of FastAPI's default bodies.

    Unhandled exceptions are not handled here: the request-id middleware turns them into a 500.
    """

    @app.exception_handler(StarletteHTTPException)
    async def handle_http_error(request: Request, exc: StarletteHTTPException) -> JSONResponse:
        """Answer an HTTP error, including an unknown path or a wrong method, with its status and headers."""
        error = ErrorBody(code=code_for(exc.status_code), message=str(exc.detail))
        return error_response(exc.status_code, error, request.state.request_id, exc.headers)

    @app.exception_handler(RequestValidationError)
    async def handle_invalid_request(request: Request, exc: RequestValidationError) -> JSONResponse:
        """Answer an invalid request with 422 and the invalid parts, without echoing the values that were sent."""
        details = [ErrorDetail(loc=list(item["loc"]), msg=item["msg"]) for item in exc.errors()]
        error = ErrorBody(code="validation_error", message="The request is not valid", details=details)
        return error_response(422, error, request.state.request_id)
