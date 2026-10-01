import pytest
from pydantic import ValidationError

from evalharness.observation import Observation

# Fields that every case shares; each case adds only its outcome fields.
_BASE: dict[str, object] = {
    "request_id": "r-1",
    "method": "GET",
    "url": "http://127.0.0.1:8000/health",
    "elapsed_s": 0.01,
    "headers": {},
    "body_text": "",
    "body_json": None,
}


@pytest.mark.parametrize(
    "outcome",
    [
        pytest.param({"status_code": 200, "transport_error": "timeout"}, id="status-and-transport-error"),
        pytest.param({"status_code": None, "transport_error": None}, id="neither"),
        pytest.param({"status_code": None, "transport_error": "connect", "body_text": "x"}, id="body-without-response"),
    ],
)
def test_contradictory_observation_is_rejected(outcome: dict[str, object]) -> None:
    """Removing or weakening the response-or-transport-error rule must break this test."""
    with pytest.raises(ValidationError):
        Observation.model_validate(_BASE | outcome)
