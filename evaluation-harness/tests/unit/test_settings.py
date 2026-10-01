import pytest

from evalharness.settings import HarnessSettings


def test_eval_environment_variables_configure_the_harness(monkeypatch: pytest.MonkeyPatch) -> None:
    """A renamed field or env prefix must break this test instead of silently falling back to defaults.

    These variable names are the harness's public interface (README, CI); unknown EVAL_* variables
    are ignored without an error.
    """
    monkeypatch.setenv("EVAL_SUT_BASE_URL", "http://localhost:9000/api")
    monkeypatch.setenv("EVAL_REQUEST_TIMEOUT_S", "2.5")

    settings = HarnessSettings()

    assert str(settings.sut_base_url) == "http://localhost:9000/api"
    assert settings.request_timeout_s == 2.5
