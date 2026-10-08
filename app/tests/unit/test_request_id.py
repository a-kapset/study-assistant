import json
import uuid

import pytest
from fastapi.testclient import TestClient

from study_assistant.composition import build_app
from study_assistant.db import Database
from study_assistant.settings import Settings


def test_well_formed_incoming_id_is_kept_in_header_and_body(settings: Settings, fake_db: Database) -> None:
    """Guards accepting the client's id: ignoring it would break matching a client's request with the app's logs."""
    with TestClient(build_app(settings, fake_db)) as client:
        response = client.get("/no-such-path", headers={"X-Request-ID": "case-042.rep_3"})

    assert response.headers["X-Request-ID"] == "case-042.rep_3"
    assert response.json()["request_id"] == "case-042.rep_3"


@pytest.mark.parametrize("incoming", ["bad id", "a;b", "x" * 129, ""])
def test_malformed_incoming_id_is_replaced_by_a_new_uuid4(incoming: str, settings: Settings, fake_db: Database) -> None:
    """Guards the id format check: a looser pattern or no length limit would let these through unchanged."""
    with TestClient(build_app(settings, fake_db)) as client:
        response = client.get("/health", headers={"X-Request-ID": incoming})

    returned = response.headers["X-Request-ID"]
    assert returned != incoming
    assert uuid.UUID(returned).version == 4


def test_each_request_without_an_id_gets_its_own(settings: Settings, fake_db: Database) -> None:
    """Guards generating the id per request: one id made once and reused would merge requests in the logs."""
    with TestClient(build_app(settings, fake_db)) as client:
        first = client.get("/health").headers["X-Request-ID"]
        second = client.get("/health").headers["X-Request-ID"]

    assert first != second
    assert uuid.UUID(first).version == 4


def test_unhandled_exception_is_logged_once_as_json_with_the_request_id(
    capsys: pytest.CaptureFixture[str], settings: Settings, fake_db: Database
) -> None:
    """Guards the error log: without the bound id or the traceback a 500 could not be traced back to its cause."""
    app = build_app(settings, fake_db)

    @app.get("/test/crash")
    async def crash() -> None:
        """Raise an exception nobody handles."""
        raise RuntimeError("boom")

    with TestClient(app, raise_server_exceptions=False) as client:
        response = client.get("/test/crash", headers={"X-Request-ID": "case-500"})

    events = [json.loads(line) for line in capsys.readouterr().out.splitlines()]
    errors = [event for event in events if event["event"] == "unhandled_error"]
    assert response.status_code == 500
    assert len(errors) == 1
    assert errors[0]["request_id"] == "case-500"
    assert errors[0]["level"] == "error"
    assert errors[0]["exception"][0]["exc_type"] == "RuntimeError"
