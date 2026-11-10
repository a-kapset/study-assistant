"""Fixtures and marking for tests that talk to a running system under test."""

from collections.abc import AsyncIterator, Iterator
from pathlib import Path
from typing import cast

import pytest
from pydantic import JsonValue

from evalharness.corpus_db import CorpusReader
from evalharness.settings import HarnessSettings, StackSettings, load_stack_settings
from evalharness.traces import TraceReader
from evalharness.transport import SutTransport

_API_DIR = Path(__file__).parent
# The address of the LLM proxy inside the stack (evaluation-harness/compose.proxy.yaml).
_PROXY_URL = "http://llm-proxy:8100/v1"
# Longest wait for a whole trace: the app sends its spans in batches every few seconds.
_TRACE_WAIT_S = 15.0


@pytest.hookimpl(tryfirst=True)
def pytest_collection_modifyitems(items: list[pytest.Item]) -> None:
    """Mark any test in this folder as api, before marker selection runs, so none can be left unmarked."""
    for item in items:
        if item.path.is_relative_to(_API_DIR):
            item.add_marker(pytest.mark.api)


@pytest.fixture
def settings() -> HarnessSettings:
    """Read the harness settings from the EVAL_* environment of this run."""
    return HarnessSettings()


@pytest.fixture
async def sut(settings: HarnessSettings) -> AsyncIterator[SutTransport]:
    """Open a transport to the system under test for one test and close it afterwards."""
    async with SutTransport(settings) as transport:
        yield transport


@pytest.fixture
def stack(settings: HarnessSettings) -> StackSettings:
    """Read the values shared with the stack (database port, roles, admin key) from its .env file."""
    return load_stack_settings(settings.stack_env_file)


@pytest.fixture
def admin_headers(stack: StackSettings) -> dict[str, str]:
    """The header that carries the stack's admin key."""
    return {"X-Admin-Key": stack.admin_api_key.get_secret_value()}


@pytest.fixture
def corpus(settings: HarnessSettings, stack: StackSettings) -> Iterator[CorpusReader]:
    """Open a read-only connection to the stack's database for one test."""
    with CorpusReader(settings, stack) as reader:
        yield reader


@pytest.fixture
async def proxy_in_front(sut: SutTransport) -> None:
    """Fail unless the app calls its model through the LLM proxy: answers then come from recordings, and a fault
    mark in the request id reaches the proxy instead of a real provider."""
    config = await sut.request("GET", "/v1/config")
    app_settings = cast(dict[str, dict[str, JsonValue]], config.body_json)["settings"]
    base_url = cast(dict[str, JsonValue], app_settings["openai"])["base_url"]
    assert base_url == _PROXY_URL, (
        f"start the stack with evaluation-harness/compose.proxy.yaml, the app uses {base_url}"
    )


@pytest.fixture
async def traces(stack: StackSettings) -> AsyncIterator[TraceReader]:
    """Open a reader of the stack's trace store for one test and close it afterwards."""
    async with TraceReader(f"http://127.0.0.1:{stack.jaeger_published_port}", wait_s=_TRACE_WAIT_S) as reader:
        yield reader
