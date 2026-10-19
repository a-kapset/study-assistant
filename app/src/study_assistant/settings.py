"""App configuration, read from environment variables with the APP_ prefix.

Nothing reads the environment at import time: the application factory calls ``load_settings()``
and tests build ``Settings`` themselves, so every caller decides which configuration the app gets.
"""

import os
from collections.abc import Mapping
from pathlib import Path
from typing import Annotated, Literal, Self

from pydantic import BaseModel, ConfigDict, Field, SecretStr, field_validator, model_validator
from pydantic_settings import BaseSettings, NoDecode, SettingsConfigDict

from study_assistant.courses.models import COURSE_ID_PATTERN

ENV_PREFIX = "APP_"
NESTED_DELIMITER = "__"
# A role name goes into CREATE ROLE and GRANT statements; lower-case letters, digits and "_" keep it a plain
# name that psql and connection settings accept without quoting.
ROLE_NAME_PATTERN = r"^[a-z_][a-z0-9_]*$"
# A Hugging Face model id, "owner/name". It becomes part of a download address and of a folder name, so only
# plain name characters are allowed.
MODEL_ID_PATTERN = r"^[A-Za-z0-9][A-Za-z0-9._-]*/[A-Za-z0-9][A-Za-z0-9._-]*$"
# A full commit hash: a branch or tag can later point to other files, and their vectors would be stored as if
# they came from the same model.
MODEL_REVISION_PATTERN = r"^[0-9a-f]{40}$"


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
    # Directory with one folder per course holding its source files, as <source_dir>/<course_id>/<file>.
    # Kept apart from the manifests because real course material is never stored with them.
    source_dir: Path = Path("../courses")
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


class RoleSettings(BaseModel):
    """A database login role that the migration run creates, or whose password it resets."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    name: str = Field(pattern=ROLE_NAME_PATTERN)
    # No default: the role must get the password its users are given, not one invented here.
    password: SecretStr


class ServerRoleSettings(RoleSettings):
    """The role the app server connects as; set as APP_SERVER_ROLE__<FIELD> for the migration run.

    It may read and write rows only, so the server cannot change the schema or do anything else the owner can.
    """

    name: str = Field(default="app_rw", pattern=ROLE_NAME_PATTERN)


class ReaderRoleSettings(RoleSettings):
    """A role that can only read rows; set as APP_READER_ROLE__<FIELD> for the migration run.

    For checks and inspection from outside the app: whatever such a client does, it cannot change the data.
    """

    name: str = Field(default="app_ro", pattern=ROLE_NAME_PATTERN)


class IngestSettings(BaseModel):
    """How course sources are cut into chunks; set as APP_INGEST__<FIELD>.

    Both values are part of the chunker version stored with every document, so changing them re-chunks
    every source on the next ingestion.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    # Longest chunk, in characters.
    chunk_size: int = Field(default=1000, ge=1)
    # Characters that consecutive chunks of a section share; must be below chunk_size.
    chunk_overlap: int = Field(default=100, ge=0)


class EmbeddingSettings(BaseModel):
    """Which embedding model turns text into vectors, and how; set as APP_EMBEDDING__<FIELD>.

    The model, its revision and the two prefixes name the embedding space that the vectors belong to:
    changing any of them starts a new space, and vectors of different spaces are never compared.
    They have no defaults, so the model is always chosen on purpose.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    # Hugging Face model id, e.g. "intfloat/multilingual-e5-small".
    model: str = Field(pattern=MODEL_ID_PATTERN)
    # Commit of the model's repository that the files are taken from; it alone decides which files are used.
    revision: str = Field(pattern=MODEL_REVISION_PATTERN)
    # Length of every vector. The vector table is created with it before any model is loaded, and loading
    # the model checks that the model really gives vectors of this length. The vector index accepts at
    # most 2,000 dimensions.
    dimension: int = Field(ge=1, le=2000)
    # Text put before every passage and every query, as the model was trained; "" for a model without one.
    # A trailing space is part of the prefix, so .env keeps these values in double quotes.
    passage_prefix: str = Field(max_length=64)
    query_prefix: str = Field(max_length=64)
    # Folder with the downloaded model files; the app only reads it. The default fits a run from app/.
    model_dir: Path = Path("../models")


class AdminSettings(BaseModel):
    """Access to the admin endpoints; set as APP_ADMIN__<FIELD>."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    # No default: without a key the app does not start, so the admin endpoints are never open by accident.
    # Sent by clients in the X-Admin-Key header.
    api_key: SecretStr = Field(min_length=16)


class MigrationSettings(BaseSettings):
    """What the one-shot migration run is configured with: the owner's connection and the app server's role.

    Kept apart from ``Settings``: the app server needs neither the owner's password nor the role setup,
    and stops at startup if either reaches its environment.
    """

    # hide_input_in_errors: a wrong value is never printed in the startup error, so a secret given by mistake
    # does not end up in the log; the field name and the rule it broke say what to fix.
    model_config = SettingsConfigDict(
        env_prefix=ENV_PREFIX,
        env_nested_delimiter=NESTED_DELIMITER,
        extra="forbid",
        frozen=True,
        hide_input_in_errors=True,
    )

    # APP_LOG_LEVEL: the lowest level written to the log.
    log_level: LogLevel = "INFO"

    # APP_GIT_COMMIT: the commit the image was built from; the image sets it for every process it runs.
    git_commit: str = "unknown"

    # APP_DB__*: the owner's connection; the owner can create extensions, tables and roles.
    db: DbSettings

    # APP_SERVER_ROLE__*: the role the app server will connect as.
    server_role: ServerRoleSettings

    # APP_READER_ROLE__*: the read-only role.
    reader_role: ReaderRoleSettings

    # APP_EMBEDDING__*: the embedding model, whose space gets its vector table here.
    embedding: EmbeddingSettings

    @model_validator(mode="after")
    def _distinct_roles(self) -> Self:
        """Refuse one name for both roles: the second grant would quietly change the first role's rights."""
        if self.server_role.name == self.reader_role.name:
            raise ValueError(f"the server role and the reader role have the same name {self.server_role.name!r}")
        return self


class Settings(BaseSettings):
    """Everything the app can be configured with.

    Groups of related settings will be nested models, set as APP_GROUP__<FIELD>.
    Secrets are typed ``SecretStr`` so that they are masked wherever settings are shown.
    """

    # hide_input_in_errors: a wrong value is never printed in the startup error, so a secret given by mistake
    # does not end up in the log; the field name and the rule it broke say what to fix.
    model_config = SettingsConfigDict(
        env_prefix=ENV_PREFIX,
        env_nested_delimiter=NESTED_DELIMITER,
        extra="forbid",
        frozen=True,
        hide_input_in_errors=True,
    )

    # APP_LOG_LEVEL: the lowest level written to the log.
    log_level: LogLevel = "INFO"

    # APP_GIT_COMMIT: the commit the running code was built from; set by the build, "unknown" otherwise.
    git_commit: str = "unknown"

    # APP_DB__*: the PostgreSQL connection.
    db: DbSettings

    # APP_COURSES__*: which courses are loaded.
    courses: CoursesSettings = Field(default_factory=CoursesSettings)

    # APP_INGEST__*: how sources are chunked.
    ingest: IngestSettings = Field(default_factory=IngestSettings)

    # APP_EMBEDDING__*: the embedding model.
    embedding: EmbeddingSettings

    # APP_ADMIN__*: access to the admin endpoints.
    admin: AdminSettings


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
    """Build the app server's settings from the process environment; unknown prefixed variables stop it."""
    _reject_unknown_env_vars(Settings)
    return Settings()


def load_migration_settings() -> MigrationSettings:
    """Build the migration run's settings from the process environment; unknown prefixed variables stop it."""
    _reject_unknown_env_vars(MigrationSettings)
    return MigrationSettings()


def _reject_unknown_env_vars(settings_cls: type[BaseSettings]) -> None:
    """Raise ``UnknownSettingsError`` naming every prefixed variable that ``settings_cls`` does not read.

    pydantic-settings silently ignores such variables, so a misspelt name would leave the default in place.
    """
    unknown = find_unknown_env_vars(os.environ, settings_cls)
    if unknown:
        names = ", ".join(unknown)
        raise UnknownSettingsError(
            f"Unknown {ENV_PREFIX}* environment variables: {names}. Check them against .env.example."
        )
