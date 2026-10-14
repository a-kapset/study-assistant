"""The effective-configuration endpoint of the running system under test."""

import pytest

from evalharness.transport import SutTransport


@pytest.mark.covers("US-001", "AC-4")
async def test_config_reports_version_and_settings(sut: SutTransport) -> None:
    """Guards the configuration contract a run will record: 200, the app version and its settings, nothing else."""
    observation = await sut.request("GET", "/v1/config")

    assert observation.transport_error is None, f"system under test unreachable at {observation.url}"
    assert observation.status_code == 200
    body = observation.body_json
    assert isinstance(body, dict)
    assert set(body) == {"app_version", "settings"}
    assert isinstance(body["app_version"], str)
    assert body["app_version"] != ""
    settings = body["settings"]
    assert isinstance(settings, dict)
    assert {"log_level", "git_commit"} <= set(settings)
