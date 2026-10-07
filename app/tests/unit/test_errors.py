from fastapi import FastAPI
from fastapi.testclient import TestClient

from study_assistant.composition import build_app
from study_assistant.settings import Settings

# Unique text that must never reach a response body: stands in for user input and for exception details.
CANARY = "canary-not-for-clients"


def _app_with_failing_routes() -> FastAPI:
    """Build the real app and add two routes that fail on purpose: one validates its input, one raises."""
    app = build_app(Settings())

    @app.get("/test/items/{number}")
    async def item(number: int) -> dict[str, int]:
        """Accept only an integer, so any other path value is a validation error."""
        return {"number": number}

    @app.get("/test/crash")
    async def crash() -> None:
        """Raise an exception nobody handles, with text that must stay out of the response."""
        raise RuntimeError(CANARY)

    return app


def test_unknown_path_answers_404_in_the_envelope() -> None:
    """Guards the HTTP-error handler: without it, or with a wrong code, an unknown path loses the envelope."""
    with TestClient(build_app(Settings())) as client:
        response = client.get("/no-such-path")

    assert response.status_code == 404
    assert response.json() == {
        "status": "error",
        "error": {"code": "not_found", "message": "Not Found"},
        "request_id": response.headers["X-Request-ID"],
    }


def test_wrong_method_answers_405_and_keeps_allow() -> None:
    """Guards passing the exception's headers on: dropping them would lose the Allow header a 405 must carry."""
    with TestClient(build_app(Settings())) as client:
        response = client.post("/health")

    assert response.status_code == 405
    assert response.headers["Allow"] == "GET"
    assert response.json()["error"]["code"] == "method_not_allowed"


def test_invalid_request_lists_the_invalid_part_but_not_its_value() -> None:
    """Guards the validation handler: 422 with where the input is wrong, never the value that was sent back."""
    with TestClient(_app_with_failing_routes()) as client:
        response = client.get(f"/test/items/{CANARY}")

    assert response.status_code == 422
    body = response.json()
    assert body["error"]["code"] == "validation_error"
    assert [detail["loc"] for detail in body["error"]["details"]] == [["path", "number"]]
    assert body["request_id"] == response.headers["X-Request-ID"]
    assert CANARY not in response.text


def test_unhandled_exception_answers_500_without_its_details() -> None:
    """Guards the middleware's catch-all: removing it or echoing the exception text would break this."""
    with TestClient(_app_with_failing_routes(), raise_server_exceptions=False) as client:
        response = client.get("/test/crash")

    assert response.status_code == 500
    assert response.json() == {
        "status": "error",
        "error": {"code": "internal_server_error", "message": "Internal server error"},
        "request_id": response.headers["X-Request-ID"],
    }
    assert CANARY not in response.text
