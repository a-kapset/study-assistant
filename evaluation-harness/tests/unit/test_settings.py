from pathlib import Path

import pytest

from evalharness.settings import HarnessSettings, StackSettings, load_stack_settings

# evaluation-harness/tests/unit/<this file> -> repository root.
REPO_ROOT = Path(__file__).resolve().parents[3]


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


def test_stack_example_file_has_every_value_the_harness_reads(monkeypatch: pytest.MonkeyPatch) -> None:
    """Guards the shared configuration: a value renamed in the stack's .env.example would leave the harness
    unable to read the database or call the admin endpoints, and the api tests would fail on setup."""
    for field in StackSettings.model_fields:
        monkeypatch.delenv(field.upper(), raising=False)

    stack = load_stack_settings(REPO_ROOT / ".env.example")

    assert stack.db_reader_user != stack.db_app_user
