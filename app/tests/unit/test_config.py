from importlib.metadata import version

from fastapi.testclient import TestClient
from pydantic import SecretStr

from study_assistant.composition import build_app
from study_assistant.settings import Settings

CANARY = "canary-not-a-real-secret"


def test_config_reports_version_and_effective_settings() -> None:
    """Guards the wiring: the endpoint must show the settings the app was built with and the real version."""
    settings = Settings(log_level="DEBUG", git_commit="abc123")

    with TestClient(build_app(settings)) as client:
        response = client.get("/v1/config")

    assert response.status_code == 200
    body = response.json()
    assert set(body) == {"app_version", "settings"}
    assert body["app_version"] == version("study-assistant")
    assert body["settings"]["log_level"] == "DEBUG"
    assert body["settings"]["git_commit"] == "abc123"


class _SettingsWithSecret(Settings):
    """Real settings plus one secret, standing in for the keys and passwords that come later."""

    api_key: SecretStr = SecretStr(CANARY)


def test_config_never_shows_a_secret() -> None:
    """Guards the masking: dumping settings in Python mode or unwrapping a secret would leak it here."""
    with TestClient(build_app(_SettingsWithSecret())) as client:
        response = client.get("/v1/config")

    assert response.status_code == 200
    assert CANARY not in response.text
    assert response.json()["settings"]["api_key"] == "**********"
