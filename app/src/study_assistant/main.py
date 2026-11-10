"""ASGI application factory.

Nothing is built at import time: the server calls ``create_app()``
(``uvicorn study_assistant.main:create_app --factory``), which reads the settings from the process
environment; tests call ``build_app`` with settings of their own.
"""

from importlib.metadata import version

from fastapi import FastAPI

from study_assistant.composition import DISTRIBUTION_NAME, build_app
from study_assistant.settings import load_settings
from study_assistant.tracing import build_tracer_provider


def create_app() -> FastAPI:
    """Build the app from the process environment, with its traces sent where the ``OTEL_*`` variables say; this is
    the server's entry point."""
    settings = load_settings()
    tracer_provider = build_tracer_provider(DISTRIBUTION_NAME, version(DISTRIBUTION_NAME))
    return build_app(settings, tracer_provider=tracer_provider)
