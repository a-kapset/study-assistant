from collections.abc import Callable
from pathlib import Path
from typing import get_args

import pytest
from pydantic import BaseModel, HttpUrl, SecretStr, ValidationError
from pydantic_settings import BaseSettings, SettingsConfigDict

from study_assistant.settings import (
    AdminSettings,
    DbSettings,
    EmbeddingServiceSettings,
    EmbeddingSettings,
    MigrationSettings,
    ReaderRoleSettings,
    ServerRoleSettings,
    Settings,
    UnknownSettingsError,
    find_unknown_env_vars,
    load_migration_settings,
    load_settings,
)

SECRET_WORDS = {"key", "token", "password", "secret", "dsn"}
REVISION = "614241f622f53c4eeff9890bdc4f31cfecc418b3"
EMBEDDING_ENV = {
    "APP_EMBEDDING__MODEL": "intfloat/multilingual-e5-small",
    "APP_EMBEDDING__REVISION": REVISION,
    "APP_EMBEDDING__DIMENSION": "384",
    "APP_EMBEDDING__PASSAGE_PREFIX": "passage: ",
    "APP_EMBEDDING__QUERY_PREFIX": "query: ",
}
EMBEDDING_SERVICE_URL = "http://embedder:8080"
# A value that must never appear in an error message.
CANARY = "canary-value-7f3a"


def _set_embedding_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Set a valid embedding model choice in the environment, as compose.yaml does."""
    for name, value in EMBEDDING_ENV.items():
        monkeypatch.setenv(name, value)


def _set_app_server_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Set every variable the app server needs and nothing more: password, admin key, model and embedding server."""
    monkeypatch.setenv("APP_DB__PASSWORD", "from-env")
    monkeypatch.setenv("APP_ADMIN__API_KEY", "admin-key-from-env")
    _set_embedding_env(monkeypatch)
    monkeypatch.setenv("APP_EMBEDDING_SERVICE__URL", EMBEDDING_SERVICE_URL)


def _embedding(**changes: object) -> dict[str, object]:
    """Return valid embedding settings as a plain dict, with ``changes`` applied, for validation tests."""
    values: dict[str, object] = {
        "model": "intfloat/multilingual-e5-small",
        "revision": REVISION,
        "dimension": 384,
        "passage_prefix": "passage: ",
        "query_prefix": "query: ",
    }
    return values | changes


@pytest.mark.covers("US-001", "AC-3")
def test_settings_are_read_from_app_variables(monkeypatch: pytest.MonkeyPatch) -> None:
    """Guards the variable names: a renamed field, prefix or group delimiter would fall back to a default or fail."""
    monkeypatch.setenv("APP_LOG_LEVEL", "DEBUG")
    monkeypatch.setenv("APP_GIT_COMMIT", "abc123")
    monkeypatch.setenv("APP_DB__HOST", "db.example")
    monkeypatch.setenv("APP_DB__PASSWORD", "from-env")
    monkeypatch.setenv("APP_ADMIN__API_KEY", "admin-key-from-env")
    monkeypatch.setenv("APP_COURSES__SOURCE_DIR", "/sources")
    monkeypatch.setenv("APP_INGEST__CHUNK_SIZE", "800")
    monkeypatch.setenv("APP_INGEST__CHUNK_OVERLAP", "80")
    _set_embedding_env(monkeypatch)
    monkeypatch.setenv("APP_EMBEDDING__MODEL_DIR", "/models")
    monkeypatch.setenv("APP_EMBEDDING_SERVICE__URL", EMBEDDING_SERVICE_URL)
    monkeypatch.setenv("APP_EMBEDDING_SERVICE__TIMEOUT_SECONDS", "5")

    settings = load_settings()

    assert settings.log_level == "DEBUG"
    assert settings.git_commit == "abc123"
    assert settings.db.host == "db.example"
    assert settings.db.password.get_secret_value() == "from-env"
    assert settings.admin.api_key.get_secret_value() == "admin-key-from-env"
    assert settings.courses.source_dir == Path("/sources")
    assert (settings.ingest.chunk_size, settings.ingest.chunk_overlap) == (800, 80)
    assert settings.embedding == EmbeddingSettings.model_validate(_embedding(model_dir=Path("/models")))
    assert settings.embedding_service == EmbeddingServiceSettings(url=HttpUrl(EMBEDDING_SERVICE_URL), timeout_seconds=5)


@pytest.mark.covers("US-001", "AC-3")
def test_missing_admin_key_stops_startup(monkeypatch: pytest.MonkeyPatch) -> None:
    """Guards the key having no default: with one, the admin endpoints would be open with a key anyone can read."""
    monkeypatch.setenv("APP_DB__PASSWORD", "from-env")
    monkeypatch.delenv("APP_ADMIN__API_KEY", raising=False)

    with pytest.raises(ValidationError, match=r"(?m)^admin$"):
        load_settings()


@pytest.mark.covers("US-001", "AC-3")
def test_short_admin_key_is_refused() -> None:
    """Guards the minimum length: a short key can be guessed by trying, since every guess gets an answer."""
    with pytest.raises(ValidationError, match="at least 16"):
        AdminSettings(api_key=SecretStr("x" * 15))


@pytest.mark.covers("US-001", "AC-3")
def test_missing_database_password_stops_startup(monkeypatch: pytest.MonkeyPatch) -> None:
    """Guards the password having no default: with one, a forgotten password would only show up as a failed login."""
    monkeypatch.setenv("APP_DB__HOST", "db.example")
    monkeypatch.delenv("APP_DB__PASSWORD", raising=False)

    with pytest.raises(ValidationError, match=r"db\.password"):
        load_settings()


@pytest.mark.covers("US-001", "AC-3")
def test_unknown_app_variable_fails_with_its_name(monkeypatch: pytest.MonkeyPatch) -> None:
    """Guards the startup check: a misspelt variable must stop the app, not leave the default in place."""
    monkeypatch.setenv("APP_LOG_LEVL", "DEBUG")

    with pytest.raises(UnknownSettingsError, match="APP_LOG_LEVL"):
        load_settings()


class _Group(BaseModel):
    """A nested settings group, as later groups such as the LLM roles will be."""

    model: str = "default"


class _GroupedSettings(BaseSettings):
    """Settings with one nested group, to check names the real settings do not have yet."""

    model_config = SettingsConfigDict(env_prefix="APP_", env_nested_delimiter="__")

    llm: _Group = _Group()


@pytest.mark.covers("US-001", "AC-3")
def test_nested_group_names_are_known_and_typos_are_not() -> None:
    """Guards the name walk: nested fields must be accepted, a typo inside a group must not."""
    environ = {
        "APP_LLM": "{}",
        "APP_LLM__MODEL": "a",
        "APP_LLM__MODLE": "b",
        "APPDATA": "not ours",
        "PATH": "not ours",
    }

    assert find_unknown_env_vars(environ, _GroupedSettings) == ["APP_LLM__MODLE"]


def _fields(model: type[BaseModel], path: str = "") -> list[tuple[str, object]]:
    """Return (dotted path, annotation) for every field of ``model``, nested groups included."""
    found: list[tuple[str, object]] = []
    for name, field in model.model_fields.items():
        annotation = field.annotation
        found.append((f"{path}{name}", annotation))
        if isinstance(annotation, type) and issubclass(annotation, BaseModel):
            found.extend(_fields(annotation, f"{path}{name}."))
    return found


@pytest.mark.covers("US-001", "AC-4")
@pytest.mark.parametrize("settings_cls", [Settings, MigrationSettings])
def test_secret_like_settings_are_secret_str(settings_cls: type[BaseSettings]) -> None:
    """Guards future fields: a key, token, password or DSN typed as plain str would be shown by /v1/config or a log."""
    plain = [
        path
        for path, annotation in _fields(settings_cls)
        if SECRET_WORDS & set(path.rsplit(".", 1)[-1].split("_"))
        and SecretStr not in (annotation, *get_args(annotation))
    ]

    assert plain == []


@pytest.mark.covers("US-001", "AC-9")
def test_app_server_refuses_role_setup_variables(monkeypatch: pytest.MonkeyPatch) -> None:
    """Guards the split: if the role setup moved into the server's settings, the server would accept it."""
    monkeypatch.setenv("APP_DB__PASSWORD", "app-password")
    monkeypatch.setenv("APP_SERVER_ROLE__PASSWORD", "app-password")

    with pytest.raises(UnknownSettingsError, match="APP_SERVER_ROLE__PASSWORD"):
        load_settings()


@pytest.mark.covers("US-001", "AC-3")
def test_migration_settings_read_owner_role_and_image_commit(monkeypatch: pytest.MonkeyPatch) -> None:
    """Guards the migration run's variable names, including the commit every process of the image is given."""
    monkeypatch.setenv("APP_GIT_COMMIT", "abc123")
    monkeypatch.setenv("APP_DB__USER", "owner_user")
    monkeypatch.setenv("APP_DB__PASSWORD", "owner-password")
    monkeypatch.setenv("APP_SERVER_ROLE__PASSWORD", "app-password")
    monkeypatch.setenv("APP_READER_ROLE__PASSWORD", "reader-password")
    _set_embedding_env(monkeypatch)

    settings = load_migration_settings()

    assert settings.git_commit == "abc123"
    assert settings.db.user == "owner_user"
    assert settings.server_role.name == "app_rw"
    assert settings.server_role.password.get_secret_value() == "app-password"
    assert settings.reader_role.name == "app_ro"
    assert settings.reader_role.password.get_secret_value() == "reader-password"
    assert settings.embedding.revision == REVISION


@pytest.mark.covers("US-001", "AC-9")
def test_one_name_for_both_roles_is_refused() -> None:
    """Guards the distinct names: one role given both sets of grants would end up with whichever ran last."""
    with pytest.raises(ValidationError, match="same name 'shared'"):
        MigrationSettings(
            db=DbSettings(password=SecretStr("owner-password")),
            server_role=ServerRoleSettings(name="shared", password=SecretStr("app-password")),
            reader_role=ReaderRoleSettings(name="shared", password=SecretStr("reader-password")),
            embedding=EmbeddingSettings.model_validate(_embedding()),
        )


@pytest.mark.covers("US-001", "AC-3")
def test_missing_role_password_stops_the_migration_run(monkeypatch: pytest.MonkeyPatch) -> None:
    """Guards the role password having no default: an invented one would show up later as the app's failed login."""
    monkeypatch.setenv("APP_DB__PASSWORD", "owner-password")
    monkeypatch.setenv("APP_SERVER_ROLE__NAME", "app_rw")
    monkeypatch.delenv("APP_SERVER_ROLE__PASSWORD", raising=False)

    with pytest.raises(ValidationError, match=r"server_role\.password"):
        load_migration_settings()


@pytest.mark.covers("US-001", "AC-3")
def test_misspelt_migration_variable_fails_with_its_name(monkeypatch: pytest.MonkeyPatch) -> None:
    """Guards the migration run's startup check: a misspelt role name would silently prepare the default role."""
    monkeypatch.setenv("APP_DB__PASSWORD", "owner-password")
    monkeypatch.setenv("APP_SERVER_ROLE__PASSWORD", "app-password")
    monkeypatch.setenv("APP_SERVER_ROLE__NAM", "learner_rw")

    with pytest.raises(UnknownSettingsError, match="APP_SERVER_ROLE__NAM"):
        load_migration_settings()


@pytest.mark.covers("US-001", "AC-3")
@pytest.mark.parametrize("name", ["App_rw", "app-rw", "1app", "app rw", ""])
def test_role_name_must_be_a_plain_lower_case_name(name: str) -> None:
    """Guards the name rule: a name that needs quoting would be case-sensitive and not found by psql or the app."""
    with pytest.raises(ValidationError):
        ServerRoleSettings(name=name, password=SecretStr("app-password"))


@pytest.mark.covers("US-001", "AC-3")
@pytest.mark.parametrize("missing", ["model", "revision", "dimension", "passage_prefix", "query_prefix"])
def test_embedding_model_has_no_default(missing: str) -> None:
    """Leaves out one field that names the model or its vector space and expects a validation error naming it.

    Catches a default added to such a field: the app would then embed with a model nobody chose, and a
    forgotten variable would not stop startup.
    """
    values = _embedding()
    del values[missing]

    with pytest.raises(ValidationError, match=rf"(?m)^{missing}$"):
        EmbeddingSettings.model_validate(values)


@pytest.mark.covers("US-001", "AC-3")
@pytest.mark.covers("US-004", "AC-22")
@pytest.mark.parametrize("revision", ["main", "v1.0", REVISION[:7], REVISION.upper(), REVISION + "0"])
def test_model_revision_must_be_a_full_commit_hash(revision: str) -> None:
    """Gives a branch, a tag, a short or an upper-case hash as the revision and expects it to be refused.

    Catches a looser rule: a branch or tag can later point to other model files, and their vectors would be
    stored as if they came from the same model.
    """
    with pytest.raises(ValidationError, match="revision"):
        EmbeddingSettings.model_validate(_embedding(revision=revision))


@pytest.mark.covers("US-001", "AC-3")
@pytest.mark.parametrize("model", ["../evil", "owner/../evil", "owner", "owner/name/extra", "/owner/name", ".x/name"])
def test_model_id_must_be_a_plain_owner_and_name(model: str) -> None:
    """Gives model ids that are not a plain "owner/name" and expects each to be refused.

    Catches a looser rule: the id becomes part of a download address and of a folder name, so ".." or an
    extra part could make the app read or write files outside the model folder.
    """
    with pytest.raises(ValidationError, match="model"):
        EmbeddingSettings.model_validate(_embedding(model=model))


@pytest.mark.covers("US-001", "AC-3")
def test_dimension_above_the_vector_index_limit_is_refused() -> None:
    """Gives 2,001 dimensions and expects a validation error at startup.

    Catches a missing upper limit: the database's vector index accepts at most 2,000 dimensions, so a larger
    model would pass startup and fail later, when the migration run creates the index.
    """
    with pytest.raises(ValidationError, match="dimension"):
        EmbeddingSettings.model_validate(_embedding(dimension=2001))


@pytest.mark.covers("US-001", "AC-3")
def test_prefixes_keep_their_trailing_space(monkeypatch: pytest.MonkeyPatch) -> None:
    """Reads the prefixes from the environment and expects them unchanged, trailing space included.

    Catches stripping of the values: the model was trained with "passage: " and "query: ", and without the
    space every vector would silently change.
    """
    _set_app_server_env(monkeypatch)

    settings = load_settings()

    assert (settings.embedding.passage_prefix, settings.embedding.query_prefix) == ("passage: ", "query: ")


@pytest.mark.covers("US-001", "AC-3")
@pytest.mark.parametrize("other_service_env", [{}, {"APP_EMBEDDING_SERVICE__TIMEOUT_SECONDS": "5"}])
def test_missing_embedding_service_url_stops_startup(
    other_service_env: dict[str, str], monkeypatch: pytest.MonkeyPatch
) -> None:
    """Sets everything the app server needs except the embedding server's address and expects startup to stop.

    Runs once with no embedding server variable at all and once with only the timeout set, so both a default
    for the whole group and a default for the address are caught. With either default a forgotten variable
    would pass startup and show up only as a failed connection at the first ingest.
    """
    _set_app_server_env(monkeypatch)
    monkeypatch.delenv("APP_EMBEDDING_SERVICE__URL")
    for name, value in other_service_env.items():
        monkeypatch.setenv(name, value)

    with pytest.raises(ValidationError, match=r"(?m)^embedding_service(\.url)?$"):
        load_settings()


@pytest.mark.covers("US-001", "AC-3")
@pytest.mark.parametrize("url", ["embedder:8080", "embedder", "ftp://embedder", "/v1"])
def test_embedding_service_url_must_be_an_http_address(url: str) -> None:
    """Gives addresses without an http(s) scheme or without a host and expects each to be refused.

    Catches the field typed as a plain string: such a value would pass startup, and the client would fail
    only at the first ingest, with an error about the request instead of one naming the setting.
    """
    with pytest.raises(ValidationError, match="url"):
        EmbeddingServiceSettings.model_validate({"url": url})


@pytest.mark.covers("US-001", "AC-3")
@pytest.mark.parametrize("timeout", [0, -1, 301, 30000])
def test_embedding_service_timeout_must_be_positive_and_at_most_five_minutes(timeout: float) -> None:
    """Gives a zero, negative or too long timeout and expects it to be refused.

    Catches missing limits: zero or less would make every request fail at once, and a value written in
    milliseconds by mistake (30000) would let a hanging server stop ingestion for hours.
    """
    with pytest.raises(ValidationError, match="timeout_seconds"):
        EmbeddingServiceSettings.model_validate({"url": EMBEDDING_SERVICE_URL, "timeout_seconds": timeout})


@pytest.mark.covers("US-001", "AC-3")
@pytest.mark.parametrize("load", [load_settings, load_migration_settings])
def test_startup_errors_name_the_setting_but_not_its_value(
    load: Callable[[], BaseSettings], monkeypatch: pytest.MonkeyPatch
) -> None:
    """Gives an invalid database port and expects an error that names the setting but does not show the value.

    Catches error messages that repeat the input: a secret set in the wrong variable, or a too-short key,
    would then be written to the startup log.
    """
    monkeypatch.setenv("APP_DB__PORT", CANARY)

    with pytest.raises(ValidationError, match=r"db\.port") as error:
        load()

    assert CANARY not in str(error.value)
