"""Liveness and readiness endpoints."""

from typing import Literal

import structlog
from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict

from study_assistant.db import Database, DatabaseUnavailableError
from study_assistant.embed.base import Embedder
from study_assistant.embed.remote import EmbeddingServiceUnavailableError
from study_assistant.errors import ErrorBody, ErrorDetail, ErrorEnvelope, code_for, error_response
from study_assistant.retrieval.rerank import Reranker, RerankServiceUnavailableError

type CheckState = Literal["ok", "unavailable"]


class HealthResponse(BaseModel):
    """Liveness: the process is up and serving HTTP. Dependencies are not checked here."""

    model_config = ConfigDict(extra="forbid")
    status: Literal["ok"]


class ReadyResponse(BaseModel):
    """Readiness: the app can serve requests, and how well.

    "ok" when every dependency answered; "degraded" when the database answered but the embedding or rerank server
    did not, so searches run on a weaker path and say so. A client that measures search quality reads this before a
    run, so it never measures the weaker path by mistake.
    """

    model_config = ConfigDict(extra="forbid")
    status: Literal["ok", "degraded"]
    checks: dict[str, CheckState]


def health_router(database: Database, embedder: Embedder, reranker: Reranker | None) -> APIRouter:
    """Build the router that serves ``GET /health`` and ``GET /ready``; ``reranker`` is checked only when searches use
    one (``None`` otherwise)."""
    router = APIRouter()
    log: structlog.typing.FilteringBoundLogger = structlog.get_logger(__name__)

    @router.get("/health")
    async def health() -> HealthResponse:
        """Report that the process is up; it answers even when dependencies are down."""
        return HealthResponse(status="ok")

    @router.get("/ready", response_model=ReadyResponse, responses={503: {"model": ErrorEnvelope}})
    async def ready(request: Request) -> ReadyResponse | JSONResponse:
        """Check the database, the embedding server and, when searches use it, the rerank server.

        Only the database decides readiness: without it nothing can be served, so 503 in the error envelope, naming
        every failed check. The model servers have a fallback in the search, so while the database answers the
        reply is 200 with each check's state and "degraded" when one of them is down: a load balancer keeps sending
        requests that the app can still serve.
        """
        checks: dict[str, CheckState] = {}
        try:
            await database.ping()
            checks["db"] = "ok"
        except DatabaseUnavailableError as exc:
            log.warning("readiness_check_failed", check="db", error=str(exc))
            checks["db"] = "unavailable"
        try:
            await embedder.ping()
            checks["embedder"] = "ok"
        except EmbeddingServiceUnavailableError as exc:
            log.warning("readiness_check_failed", check="embedder", error=str(exc))
            checks["embedder"] = "unavailable"
        if reranker is not None:
            try:
                await reranker.ping()
                checks["reranker"] = "ok"
            except RerankServiceUnavailableError as exc:
                log.warning("readiness_check_failed", check="reranker", error=str(exc))
                checks["reranker"] = "unavailable"
        if checks["db"] != "ok":
            failed = [check for check, state in checks.items() if state != "ok"]
            error = ErrorBody(
                code=code_for(503),
                message="The app is not ready",
                details=[ErrorDetail(loc=["checks", check], msg="unavailable") for check in failed],
            )
            return error_response(503, error, request.state.request_id)
        status: Literal["ok", "degraded"] = "ok" if all(state == "ok" for state in checks.values()) else "degraded"
        return ReadyResponse(status=status, checks=checks)

    return router
