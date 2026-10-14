"""Harness configuration: EVAL_* environment variables, plus the values it shares with the stack's .env file."""

from pathlib import Path

from pydantic import Field, HttpUrl, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class HarnessSettings(BaseSettings):
    """Where the system under test is and how long to wait for it.

    Nothing creates an instance at import time: tests and entry points build it explicitly,
    so each run can point the harness at a different system.
    """

    model_config = SettingsConfigDict(env_prefix="EVAL_", extra="forbid", frozen=True)

    # EVAL_SUT_BASE_URL: root URL of the system under test; request paths are joined to it.
    sut_base_url: HttpUrl = HttpUrl("http://127.0.0.1:8000")

    # EVAL_REQUEST_TIMEOUT_S: per-request timeout in seconds.
    request_timeout_s: float = Field(default=10.0, gt=0)

    # EVAL_STACK_ENV_FILE: the .env file the stack under test runs with; relative paths start from the
    # working directory, and the default fits a run from evaluation-harness/.
    stack_env_file: Path = Path("../.env")

    # EVAL_DB_HOST: the host on which the stack publishes its database port.
    db_host: str = "127.0.0.1"


class StackSettings(BaseSettings):
    """The values of the stack's configuration the harness needs, read from the same .env file the stack uses.

    So a password or a port is set in one place and cannot differ between the stack and the harness.
    A variable set in the environment wins over the file. Other values in the file are ignored.
    """

    model_config = SettingsConfigDict(extra="ignore", frozen=True)

    db_name: str
    db_published_port: int = Field(ge=1, le=65535)
    # The read-only role: every read of stored data goes through it, so a check cannot change what it checks.
    db_reader_user: str
    db_reader_password: SecretStr
    # The app's own role: used only to set up rows a check needs and to remove them afterwards.
    db_app_user: str
    db_app_password: SecretStr
    admin_api_key: SecretStr


def load_stack_settings(env_file: Path) -> StackSettings:
    """Read the stack's values from ``env_file``; without the file, only from the environment."""
    return StackSettings(_env_file=env_file if env_file.is_file() else None)  # type: ignore[call-arg]
