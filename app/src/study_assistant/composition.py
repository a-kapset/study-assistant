"""Composition root: the one place where the app's parts are built from the settings and wired together."""

from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from importlib.metadata import version

import structlog
from fastapi import FastAPI

from study_assistant.api.config import config_router
from study_assistant.api.courses import courses_router
from study_assistant.api.health import health_router
from study_assistant.api.middleware import RequestIdMiddleware
from study_assistant.courses.registry import load_courses
from study_assistant.db import Database, PostgresDatabase
from study_assistant.errors import install_error_handlers
from study_assistant.logging_setup import configure_logging
from study_assistant.settings import Settings

DISTRIBUTION_NAME = "study-assistant"


def build_app(settings: Settings, database: Database | None = None) -> FastAPI:
    """Build the app from ``settings``: logging first, then the courses, the database and the routes.

    The only thing read here is the enabled courses' manifests, so a bad manifest stops startup.
    Nothing reads the environment or connects anywhere: the database opens when the server starts.
    Tests pass their own ``database``; otherwise the app uses PostgreSQL as configured in ``settings``.
    """
    configure_logging(settings.log_level)
    app_version = version(DISTRIBUTION_NAME)
    courses = load_courses(settings.courses.manifest_dir, settings.courses.enabled)
    db = database if database is not None else PostgresDatabase(settings.db)

    @asynccontextmanager
    async def lifespan(_app: FastAPI) -> AsyncGenerator[None]:
        """Open the database when the server starts and close it when the server stops."""
        await db.open()
        try:
            yield
        finally:
            await db.close()

    app = FastAPI(title="Study Assistant", version=app_version, lifespan=lifespan)
    # Every request gets an id and every error one envelope, including errors raised before any route runs.
    app.add_middleware(RequestIdMiddleware)
    install_error_handlers(app)
    app.include_router(health_router(db))
    app.include_router(courses_router(courses))
    app.include_router(config_router(settings=settings, app_version=app_version))

    structlog.get_logger().info(
        "app_configured",
        app_version=app_version,
        git_commit=settings.git_commit,
        log_level=settings.log_level,
        courses=list(courses),
    )

    return app
