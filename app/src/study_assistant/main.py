"""ASGI application factory.

Nothing is built at import time: the server calls ``create_app()``
(``uvicorn study_assistant.main:create_app --factory``), which reads the settings from the process
environment; tests call ``build_app`` with settings of their own.
"""

from fastapi import FastAPI

from study_assistant.composition import build_app
from study_assistant.settings import load_settings


def create_app() -> FastAPI:
    """Build the app from the process environment; this is the server's entry point."""
    return build_app(load_settings())
