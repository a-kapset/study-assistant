"""Composition root: the one place where the app's parts are built from the settings and wired together."""

from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from importlib.metadata import version

import httpx2
import structlog
from fastapi import FastAPI
from langchain_core.language_models import BaseChatModel

from study_assistant.answer.service import Answerer
from study_assistant.api.admin import admin_router
from study_assistant.api.ask import ask_router
from study_assistant.api.config import config_router
from study_assistant.api.courses import courses_router
from study_assistant.api.health import health_router
from study_assistant.api.middleware import RequestIdMiddleware
from study_assistant.api.search import search_router
from study_assistant.courses.registry import load_courses
from study_assistant.db import Database, PostgresDatabase
from study_assistant.embed.base import Embedder
from study_assistant.embed.remote import RemoteEmbedder
from study_assistant.errors import install_error_handlers
from study_assistant.ingest.chunker import ChunkSize
from study_assistant.ingest.repository import PostgresCorpusStore
from study_assistant.ingest.service import CorpusStore, CourseIngestor
from study_assistant.llm.providers import build_chat_model
from study_assistant.logging_setup import configure_logging
from study_assistant.retrieval.repository import PostgresSearchStore, SearchStore
from study_assistant.retrieval.service import VectorSearch
from study_assistant.settings import Settings

DISTRIBUTION_NAME = "study-assistant"


def build_app(
    settings: Settings,
    database: Database | None = None,
    store: CorpusStore | None = None,
    embedder: Embedder | None = None,
    search_store: SearchStore | None = None,
    chat_model: BaseChatModel | None = None,
) -> FastAPI:
    """Build the app from ``settings``: logging first, then the courses, the database, the models and the routes.

    The only thing read here is the enabled courses' manifests, so a bad manifest stops startup, and so do a chunk
    overlap that is not below the chunk size and a model provider the app does not know. Nothing reads the
    environment or connects anywhere: the database opens when the server starts, the embedding server is first
    called by an ingestion or a readiness check, and the model provider by the first question. Tests pass their own
    ``database``, corpus ``store``, ``embedder``, ``search_store`` and ``chat_model``; otherwise the app uses
    PostgreSQL, the embedding server and the model provider as configured in ``settings``.
    """
    configure_logging(settings.log_level)
    app_version = version(DISTRIBUTION_NAME)
    courses = load_courses(settings.courses.manifest_dir, settings.courses.enabled)
    db = database if database is not None else PostgresDatabase(settings.db)
    corpus = store if store is not None else PostgresCorpusStore(db)
    # The connections to the embedding server and the model provider belong to the app, which closes them when the
    # server stops. A client nobody uses opens no connection, so it is made even when tests pass both models.
    http_client = httpx2.AsyncClient()
    if embedder is None:
        embedder = RemoteEmbedder.from_settings(settings.embedding, settings.embedding_service, http_client)
    if chat_model is None:
        chat_model = build_chat_model(settings.llm.answer, settings, http_client)
    size = ChunkSize(size=settings.ingest.chunk_size, overlap=settings.ingest.chunk_overlap)
    ingestor = CourseIngestor(corpus, embedder, settings.courses.source_dir, size)
    search = VectorSearch(search_store if search_store is not None else PostgresSearchStore(db), embedder)
    # The grounded mode searches with the same search that serves /v1/search, so its passages are those the search
    # quality is measured on.
    answerer = Answerer(chat_model, search, settings.ask, settings.search)

    @asynccontextmanager
    async def lifespan(_app: FastAPI) -> AsyncGenerator[None]:
        """Open the database when the server starts; close it and the HTTP connections when it stops."""
        await db.open()
        try:
            yield
        finally:
            await db.close()
            await http_client.aclose()

    app = FastAPI(title="Study Assistant", version=app_version, lifespan=lifespan)
    # Every request gets an id and every error one envelope, including errors raised before any route runs.
    app.add_middleware(RequestIdMiddleware)
    install_error_handlers(app)
    app.include_router(health_router(db, embedder))
    app.include_router(courses_router(courses, corpus, embedder.space))
    app.include_router(search_router(courses, search, settings.search))
    app.include_router(ask_router(courses, answerer))
    app.include_router(admin_router(courses, ingestor, settings.admin.api_key))
    app.include_router(config_router(settings=settings, app_version=app_version))

    structlog.get_logger().info(
        "app_configured",
        app_version=app_version,
        git_commit=settings.git_commit,
        log_level=settings.log_level,
        courses=list(courses),
    )

    return app
