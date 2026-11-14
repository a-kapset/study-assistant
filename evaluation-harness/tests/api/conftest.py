"""Fixtures and marking for tests that talk to a running system under test."""

import subprocess
import time
from collections.abc import AsyncIterator, Iterator
from pathlib import Path
from typing import cast

import httpx2
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
# The service of the embedding server in the stack's compose file.
_EMBEDDER_SERVICE = "embedder"
# Longest wait for one request while the embedding server is stopped: the app first tries the server several times
# (about 13 s on a developer machine) and only then answers on its weaker path.
_OUTAGE_REQUEST_TIMEOUT_S = 60.0
# Longest wait until the app is ready again after the embedding server was started: the server loads its model first.
_RECOVERY_WAIT_S = 180.0


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


@pytest.fixture(scope="module")
def embedder_stopped() -> Iterator[None]:
    """Stop the container of the stack's embedding server for the tests of one module, then start it again and
    wait until the app reports that everything answers.

    The outage is the real one: the container is stopped with Docker Compose, in the folder of the stack's .env
    file, so the app meets a server that cannot be reached, not a prepared error. A server that does not come back
    is an error of this fixture, so the tests and runs after it never work on a half-broken stack unnoticed.
    """
    settings = HarnessSettings()
    _compose(settings, "stop")
    try:
        yield
    finally:
        _compose(settings, "start")
        _wait_until_ready(settings)


def _compose(settings: HarnessSettings, verb: str) -> None:
    """Run ``docker compose <verb>`` on the embedding server of the stack; a failed command is an error with
    Docker's own message."""
    env_file = settings.stack_env_file.resolve()
    result = subprocess.run(
        ["docker", "compose", "--env-file", str(env_file), verb, _EMBEDDER_SERVICE],
        cwd=env_file.parent,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(f"docker compose {verb} {_EMBEDDER_SERVICE} failed: {result.stderr.strip()}")


def _wait_until_ready(settings: HarnessSettings) -> None:
    """Ask the app's readiness until it says "ok"; raise when it does not within the recovery wait."""
    deadline = time.monotonic() + _RECOVERY_WAIT_S
    seen = "no answer"
    while time.monotonic() < deadline:
        try:
            response = httpx2.get(f"{str(settings.sut_base_url).rstrip('/')}/ready", timeout=10.0)
            seen = response.text[:200]
            if response.status_code == 200 and response.json().get("status") == "ok":
                return
        except httpx2.HTTPError as error:
            seen = type(error).__name__
        time.sleep(1.0)
    raise RuntimeError(
        f"the embedding server was started again, but the app is not ready after {_RECOVERY_WAIT_S:.0f} s: {seen}"
    )


@pytest.fixture
async def patient_sut(settings: HarnessSettings) -> AsyncIterator[SutTransport]:
    """Open a transport that waits long enough for a request the app answers only after it has given up on a
    stopped server; the usual wait would end first and the answer would never be seen."""
    patient = settings.model_copy(update={"request_timeout_s": _OUTAGE_REQUEST_TIMEOUT_S})
    async with SutTransport(patient) as transport:
        yield transport


@pytest.fixture
async def traces(stack: StackSettings) -> AsyncIterator[TraceReader]:
    """Open a reader of the stack's trace store for one test and close it afterwards."""
    async with TraceReader(f"http://127.0.0.1:{stack.jaeger_published_port}", wait_s=_TRACE_WAIT_S) as reader:
        yield reader
