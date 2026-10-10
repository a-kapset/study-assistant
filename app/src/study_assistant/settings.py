"""App configuration, read from environment variables with the APP_ prefix.

Nothing reads the environment at import time: the application factory calls ``load_settings()``
and tests build ``Settings`` themselves, so every caller decides which configuration the app gets.
"""

import os
from collections.abc import Mapping
from pathlib import Path
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, SecretStr, field_validator
from pydantic_settings import BaseSettings, NoDecode, SettingsConfigDict

from study_assistant.courses.models import COURSE_ID_PATTERN

ENV_PREFIX = "APP_"
NESTED_DELIMITER = "__"

type LogLevel = Literal["DEBUG", "INFO", "WARNING", "ERROR"]
type CourseId = Annotated[str, Field(pattern=COURSE_ID_PATTERN)]


class DbSettings(BaseModel):
    """Where and how the app connects to PostgreSQL; set as APP_DB__<FIELD>."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    host: str = "localhost"
    port: int = Field(default=5432, ge=1, le=65535)
    name: str = "study_assistant"
    user: str = "study_assistant"
    # No default: a missing password should stop the app at startup, not show up later as a failed login.
    password: SecretStr
    # Seconds to wait for a connection, both when opening one and when taking one from the pool.
    connect_timeout_s: int = Field(default=3, ge=1)


class CoursesSettings(BaseModel):
    """Which courses the app loads at startup and where their manifests are; set as APP_COURSES__<FIELD>."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    # Directory with one folder per course, each holding a course.yaml; relative paths start from the
    # working directory, and the default fits a run from app/.
    manifest_dir: Path = Path("../courses")
    # Course ids separated by commas, e.g. "demo,other_course"; none by default, so courses are enabled
    # only on purpose. NoDecode keeps the raw string, which would otherwise have to be a JSON list.
    enabled: Annotated[tuple[CourseId, ...], NoDecode] = ()

    @field_validator("enabled", mode="before")
    @classmethod
    def _split_comma_list(cls, value: object) -> object:
        """Turn "a, b" from the environment into ("a", "b"); a value that is not a string passes through."""
        if isinstance(value, str):
            return tuple(part.strip() for part in value.split(",") if part.strip())
        return value

    @field_validator("enabled")
    @classmethod
    def _reject_repeats(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        """Refuse a course listed twice instead of quietly loading it once."""
        if len(set(value)) != len(value):
            raise ValueError("a course is listed more than once")
        return value


class Settings(BaseSettings):
    """Everything the app can be configured with.

    Groups of related settings will be nested models, set as APP_GROUP__<FIELD>.
    Secrets are typed ``SecretStr`` so that they are masked wherever settings are shown.
    """

    model_config = SettingsConfigDict(
        env_prefix=ENV_PREFIX, env_nested_delimiter=NESTED_DELIMITER, extra="forbid", frozen=True
    )

    # APP_LOG_LEVEL: the lowest level written to the log.
    log_level: LogLevel = "INFO"

    # APP_GIT_COMMIT: the commit the running code was built from; set by the build, "unknown" otherwise.
    git_commit: str = "unknown"

    # APP_DB__*: the PostgreSQL connection.
    db: DbSettings

    # APP_COURSES__*: which courses are loaded.
    courses: CoursesSettings = Field(default_factory=CoursesSettings)


class UnknownSettingsError(ValueError):
    """The environment has prefixed variables that no setting reads."""


def find_unknown_env_vars(environ: Mapping[str, str], settings_cls: type[BaseSettings]) -> list[str]:
    """Return the sorted names in ``environ`` that carry the settings prefix but match no setting."""
    prefix = settings_cls.model_config.get("env_prefix", "").upper()
    known = _expected_env_names(settings_cls, prefix)
    return sorted(name for name in environ if name.upper().startswith(prefix) and name.upper() not in known)


def _expected_env_names(model: type[BaseModel], prefix: str) -> set[str]:
    """Return every variable name that sets a field of ``model``, including the fields of nested groups.

    A nested group can be set as a whole (one JSON value) or field by field, so both names count.
    Only a field typed exactly as a model is treated as a group; ``Group | None`` would not be.
    """
    names: set[str] = set()
    for field_name, field in model.model_fields.items():
        env_name = f"{prefix}{field_name}".upper()
        names.add(env_name)
        annotation = field.annotation
        if isinstance(annotation, type) and issubclass(annotation, BaseModel):
            names |= _expected_env_names(annotation, env_name + NESTED_DELIMITER)
    return names


def load_settings() -> Settings:
    """Build the settings from the process environment and refuse prefixed variables that no setting reads.

    pydantic-settings silently ignores such variables, so a misspelt name would leave the default in place.
    """
    unknown = find_unknown_env_vars(os.environ, Settings)
    if unknown:
        names = ", ".join(unknown)
        raise UnknownSettingsError(
            f"Unknown {ENV_PREFIX}* environment variables: {names}. Check them against .env.example."
        )
    return Settings()
