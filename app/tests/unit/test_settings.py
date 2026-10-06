from typing import get_args

import pytest
from pydantic import BaseModel, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict

from study_assistant.settings import Settings, UnknownSettingsError, find_unknown_env_vars, load_settings

SECRET_WORDS = {"key", "token", "password", "secret", "dsn"}


def test_settings_are_read_from_app_variables(monkeypatch: pytest.MonkeyPatch) -> None:
    """Guards the variable names: a renamed field or prefix would silently fall back to its default."""
    monkeypatch.setenv("APP_LOG_LEVEL", "DEBUG")
    monkeypatch.setenv("APP_GIT_COMMIT", "abc123")

    settings = load_settings()

    assert settings.log_level == "DEBUG"
    assert settings.git_commit == "abc123"


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


def test_secret_like_settings_are_secret_str() -> None:
    """Guards future fields: a key, token, password or DSN typed as plain str would be shown by /v1/config."""
    plain = [
        path
        for path, annotation in _fields(Settings)
        if SECRET_WORDS & set(path.rsplit(".", 1)[-1].split("_"))
        and SecretStr not in (annotation, *get_args(annotation))
    ]

    assert plain == []
