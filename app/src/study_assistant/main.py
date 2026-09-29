"""ASGI application factory.

Nothing is built at import time: the server calls ``create_app()``
(``uvicorn study_assistant.main:create_app --factory``) and tests call it directly,
so every caller gets a fresh, fully configured app.
"""

from typing import Literal

from fastapi import FastAPI
from pydantic import BaseModel, ConfigDict


class HealthResponse(BaseModel):
    """Liveness: the process is up and serving HTTP. Dependencies are not checked here."""

    model_config = ConfigDict(extra="forbid")
    status: Literal["ok"]


def create_app() -> FastAPI:
    app = FastAPI(title="Study Assistant")

    @app.get("/health")
    async def health() -> HealthResponse:
        return HealthResponse(status="ok")

    return app
