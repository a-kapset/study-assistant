"""Liveness endpoint."""

from typing import Literal

from fastapi import APIRouter
from pydantic import BaseModel, ConfigDict


class HealthResponse(BaseModel):
    """Liveness: the process is up and serving HTTP. Dependencies are not checked here."""

    model_config = ConfigDict(extra="forbid")
    status: Literal["ok"]


def health_router() -> APIRouter:
    """Build the router that serves ``GET /health``."""
    router = APIRouter()

    @router.get("/health")
    async def health() -> HealthResponse:
        """Report that the process is up; it answers even when dependencies are down."""
        return HealthResponse(status="ok")

    return router
