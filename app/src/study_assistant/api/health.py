"""Liveness and readiness endpoints."""

from typing import Literal

import structlog
from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict

from study_assistant.db import Database, DatabaseUnavailableError
from study_assistant.errors import ErrorBody, ErrorDetail, ErrorEnvelope, code_for, error_response


class HealthResponse(BaseModel):
    """Liveness: the process is up and serving HTTP. Dependencies are not checked here."""

    model_config = ConfigDict(extra="forbid")
    status: Literal["ok"]


class ReadyResponse(BaseModel):
    """Readiness: every dependency the app needs to serve requests has answered."""

    model_config = ConfigDict(extra="forbid")
    status: Literal["ok"]
    checks: dict[str, Literal["ok"]]


def health_router(database: Database) -> APIRouter:
    """Build the router that serves ``GET /health`` and ``GET /ready``."""
    router = APIRouter()
    log: structlog.typing.FilteringBoundLogger = structlog.get_logger(__name__)

    @router.get("/health")
    async def health() -> HealthResponse:
        """Report that the process is up; it answers even when dependencies are down."""
        return HealthResponse(status="ok")

    @router.get("/ready", response_model=ReadyResponse, responses={503: {"model": ErrorEnvelope}})
    async def ready(request: Request) -> ReadyResponse | JSONResponse:
        """Answer 200 when the database answers, otherwise 503 in the error envelope naming the failed check."""
        try:
            await database.ping()
        except DatabaseUnavailableError as exc:
            log.warning("readiness_check_failed", check="db", error=str(exc))
            error = ErrorBody(
                code=code_for(503),
                message="The app is not ready",
                details=[ErrorDetail(loc=["checks", "db"], msg="unavailable")],
            )
            return error_response(503, error, request.state.request_id)
        return ReadyResponse(status="ok", checks={"db": "ok"})

    return router
