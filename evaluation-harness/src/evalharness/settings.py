"""Harness configuration: EVAL_* environment variables, plus the values it shares with the stack's .env file."""

from pathlib import Path
from typing import Literal, Self

from pydantic import Field, HttpUrl, SecretStr, model_validator
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

    # EVAL_PROXY_LEDGER_FILE: the ledger of the LLM proxy, as the stack writes it to this machine; relative paths
    # start from the working directory, and the default fits a run from evaluation-harness/.
    proxy_ledger_file: Path = Path("runs/proxy_ledger.jsonl")


class ProxySettings(BaseSettings):
    """How the LLM proxy runs: its mode, the real provider behind it and where it keeps what it sees.

    Nothing creates an instance at import time: the proxy's entry point builds it when the process starts.
    """

    model_config = SettingsConfigDict(env_prefix="EVAL_PROXY_", extra="forbid", frozen=True)

    # EVAL_PROXY_MODE: "replay" answers only from recordings and never pays; "record" answers from a recording
    # when there is one and sends only new calls to the provider, which keeps their replies.
    mode: Literal["record", "replay"] = "replay"

    # EVAL_PROXY_UPSTREAM_URL: address of the real provider API.
    upstream_url: HttpUrl = HttpUrl("https://api.openai.com/v1")

    # EVAL_PROXY_UPSTREAM_API_KEY: the real key; only the proxy holds it, and the app runs with a placeholder.
    upstream_api_key: SecretStr = SecretStr("")

    # EVAL_PROXY_UPSTREAM_TIMEOUT_S: longest wait for one provider reply, in seconds.
    upstream_timeout_s: float = Field(default=60.0, gt=0)

    # EVAL_PROXY_CASSETTE_DIR: folder of the recordings; a local course uses a folder that is not committed.
    cassette_dir: Path = Path("cassettes")

    # EVAL_PROXY_LEDGER_FILE: the file that gets one JSON line per call.
    ledger_file: Path = Path("runs/proxy_ledger.jsonl")

    @model_validator(mode="after")
    def _record_needs_a_key(self) -> Self:
        """Stop at start when ``record`` has no key: every new call would fail and nothing would be recorded."""
        if self.mode == "record" and not self.upstream_api_key.get_secret_value():
            raise ValueError("EVAL_PROXY_UPSTREAM_API_KEY is required in record mode")
        return self


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
    # The course whose local materials the stack mounts, and their folder. A relative folder starts from the folder
    # of the .env file (the repository root), as Docker Compose reads it.
    local_course_id: str
    local_course_sources: Path


def load_stack_settings(env_file: Path) -> StackSettings:
    """Read the stack's values from ``env_file``; without the file, only from the environment."""
    return StackSettings(_env_file=env_file if env_file.is_file() else None)  # type: ignore[call-arg]
