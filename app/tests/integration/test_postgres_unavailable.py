"""The real database code against a local port where nothing listens: no server is needed, only the driver.

The unit tests use a fake database that never fails to open and raises the app's own error when it is
down. These tests check that the real connection pool behaves the same way.
"""

import asyncio
import socket
from collections.abc import Iterator

import pytest
from fastapi.testclient import TestClient
from pydantic import SecretStr

from study_assistant.composition import build_app
from study_assistant.settings import AdminSettings, DbSettings, EmbeddingSettings, Settings


def _closed_port() -> int:
    """Return a local port with nothing listening on it: the system picks a free port, and it is released at once."""
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port: int = sock.getsockname()[1]
    return port


async def _running_loop() -> asyncio.AbstractEventLoop:
    """Return the event loop the app runs on."""
    return asyncio.get_running_loop()


@pytest.fixture
def client() -> Iterator[TestClient]:
    """Start the app with its real database code pointed at a closed port, with a one-second connect timeout.

    The selector event loop is chosen on purpose. On Windows the default loop is one the async database
    driver refuses to run on, and from outside that failure looks exactly like a database that is down,
    so the tests would pass without reaching the driver's connection code. On Linux the selector loop is
    already the default. The loop is checked after startup, so a lost loop setting fails the tests.
    """
    settings = Settings(
        db=DbSettings(
            host="127.0.0.1",
            port=_closed_port(),
            password=SecretStr("integration-test-password"),
            connect_timeout_s=1,
        ),
        embedding=EmbeddingSettings(
            model="intfloat/multilingual-e5-small",
            revision="614241f622f53c4eeff9890bdc4f31cfecc418b3",
            dimension=384,
            passage_prefix="passage: ",
            query_prefix="query: ",
        ),
        admin=AdminSettings(api_key=SecretStr("integration-test-admin-key")),
    )
    with TestClient(build_app(settings), backend_options={"loop_factory": asyncio.SelectorEventLoop}) as test_client:
        assert test_client.portal is not None
        assert isinstance(test_client.portal.call(_running_loop), asyncio.SelectorEventLoop)
        yield test_client


@pytest.mark.covers("US-001", "AC-1")
def test_the_app_starts_and_stays_live_while_the_database_is_down(client: TestClient) -> None:
    """Starts the app while no database listens and asks /health; expects 200.

    Catches a startup that waits for the first database connection: the app would then fail to start
    whenever the database is down or slow, instead of starting and reporting that it is not ready.
    """
    response = client.get("/health")

    assert response.status_code == 200


@pytest.mark.covers("US-001", "AC-2")
def test_ready_answers_503_when_the_real_driver_cannot_connect(client: TestClient) -> None:
    """Asks /ready while no database listens; expects 503, not 500.

    Catches a readiness ping that lets the real driver's failure escape instead of turning it into the
    app's "database unavailable" error, for example when a waiting error is dropped from the caught ones.
    Only the timeout path is reached here: with no server the pool never hands out a connection, so an
    error raised while a query runs on a broken connection is not exercised.
    """
    response = client.get("/ready")

    assert response.status_code == 503
