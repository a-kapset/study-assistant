"""Harness confuguration, read from environment variables with the EVAL_ prefix."""

from pydantic import Field, HttpUrl
from pydantic_settings import BaseSettings, SettingsConfigDict


class HarnessSettings(BaseSettings):
    """Where the system under test is and how long to wait for it.

    Nothing creates as instance at import time: tests and entry points build it explicitly,
    so each run can piont the harness at a different system
    """

    model_config = SettingsConfigDict(env_prefix="EVAL_", extra="forbid", frozen=True)

    # EVAL_SUT_BASE_URL: root URL of the system under test; request paths are joined to it.
    sut_base_url: HttpUrl = HttpUrl("http://127.0.0.1:8000")

    # EVAL_REQUEST_TIMEOUT_S: per-request timeout in seconds.
    request_timeout_s: float = Field(default=10.0, gt=0)
