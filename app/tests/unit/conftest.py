"""Shared unit-test setup: settings that do not depend on the shell, and a database the tests can switch off."""

import pytest
from pydantic import SecretStr

from study_assistant.db import DatabaseUnavailableError
from study_assistant.settings import DbSettings, Settings


class FakeDatabase:
    """Stands in for PostgreSQL: answers a ping while ``up`` is true and records when it is opened and closed."""

    def __init__(self) -> None:
        """Start reachable, with nothing recorded yet."""
        self.up = True
        self.events: list[str] = []

    async def open(self) -> None:
        """Record that the app opened the database."""
        self.events.append("open")

    async def close(self) -> None:
        """Record that the app closed the database."""
        self.events.append("close")

    async def ping(self) -> None:
        """Answer while up; otherwise fail the way the real database does."""
        if not self.up:
            raise DatabaseUnavailableError("OperationalError")


@pytest.fixture
def settings() -> Settings:
    """Valid settings with a placeholder password; explicit values win over any APP_* variables in the shell."""
    return Settings(db=DbSettings(password=SecretStr("unit-test-password")))


@pytest.fixture
def fake_db() -> FakeDatabase:
    """A fresh, reachable fake database for one test."""
    return FakeDatabase()
