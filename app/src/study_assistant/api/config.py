"""Effective configuration: what the running app was built with, secrets masked."""

from fastapi import APIRouter
from pydantic import BaseModel, ConfigDict, JsonValue

from study_assistant.settings import Settings


class ConfigResponse(BaseModel):
    """The app version and the effective settings, with every secret masked."""

    model_config = ConfigDict(extra="forbid")
    app_version: str
    settings: dict[str, JsonValue]


def config_router(settings: Settings, app_version: str) -> APIRouter:
    """Build the router that serves ``GET /v1/config`` for the settings the app was built with."""
    # JSON mode turns every SecretStr into "**********", so secrets never reach the response.
    response = ConfigResponse(app_version=app_version, settings=settings.model_dump(mode="json"))
    router = APIRouter()

    @router.get("/v1/config")
    async def config() -> ConfigResponse:
        """Return the effective configuration, so a client can record what it ran against."""
        return response

    return router
