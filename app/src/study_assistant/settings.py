"""App configuration, read from environment variables with the APP_ prefix.

Nothing reads the environment at import time: the application factory calls ``load_settings()``
and tests build ``Settings`` themselves, so every caller decides which configuration the app gets.
"""

import os
from collections.abc import Mapping
from typing import Literal

from pydantic import BaseModel
from pydantic_settings import BaseSettings, SettingsConfigDict

ENV_PREFIX = "APP_"
NESTED_DELIMITER = "__"

type LogLevel = Literal["DEBUG", "INFO", "WARNING", "ERROR"]


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
