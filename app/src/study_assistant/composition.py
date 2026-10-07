"""Composition root: the one place where the app's parts are built from the settings and wired together."""

from importlib.metadata import version

import structlog
from fastapi import FastAPI

from study_assistant.api.config import config_router
from study_assistant.api.health import health_router
from study_assistant.api.middleware import RequestIdMiddleware
from study_assistant.errors import install_error_handlers
from study_assistant.logging_setup import configure_logging
from study_assistant.settings import Settings

DISTRIBUTION_NAME = "study-assistant"


def build_app(settings: Settings) -> FastAPI:
    """Build the app from ``settings``: logging first, then the routes. Nothing here reads the environment."""
    configure_logging(settings.log_level)
    app_version = version(DISTRIBUTION_NAME)

    app = FastAPI(title="Study Assistant", version=app_version)
    # Every request gets an id and every error one envelope, including errors raised before any route runs.
    app.add_middleware(RequestIdMiddleware)
    install_error_handlers(app)
    app.include_router(health_router())
    app.include_router(config_router(settings=settings, app_version=app_version))

    structlog.get_logger().info(
        "app_configured",
        app_version=app_version,
        git_commit=settings.git_commit,
        log_level=settings.log_level,
    )

    return app
