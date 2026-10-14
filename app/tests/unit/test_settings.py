from pathlib import Path
from typing import get_args

import pytest
from pydantic import BaseModel, SecretStr, ValidationError
from pydantic_settings import BaseSettings, SettingsConfigDict

from study_assistant.settings import (
    AdminSettings,
    DbSettings,
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

    settings = load_settings()

    assert settings.log_level == "DEBUG"
    assert settings.git_commit == "abc123"
    assert settings.db.host == "db.example"
    assert settings.db.password.get_secret_value() == "from-env"
    assert settings.admin.api_key.get_secret_value() == "admin-key-from-env"
    assert settings.courses.source_dir == Path("/sources")
    assert (settings.ingest.chunk_size, settings.ingest.chunk_overlap) == (800, 80)


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

    settings = load_migration_settings()

    assert settings.git_commit == "abc123"
    assert settings.db.user == "owner_user"
    assert settings.server_role.name == "app_rw"
    assert settings.server_role.password.get_secret_value() == "app-password"
    assert settings.reader_role.name == "app_ro"
    assert settings.reader_role.password.get_secret_value() == "reader-password"


@pytest.mark.covers("US-001", "AC-9")
def test_one_name_for_both_roles_is_refused() -> None:
    """Guards the distinct names: one role given both sets of grants would end up with whichever ran last."""
    with pytest.raises(ValidationError, match="same name 'shared'"):
        MigrationSettings(
            db=DbSettings(password=SecretStr("owner-password")),
            server_role=ServerRoleSettings(name="shared", password=SecretStr("app-password")),
            reader_role=ReaderRoleSettings(name="shared", password=SecretStr("reader-password")),
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
