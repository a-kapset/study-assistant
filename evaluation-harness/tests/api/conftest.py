"""Fixtures and marking for tests that talk to a running system under test."""

from collections.abc import AsyncIterator
from pathlib import Path

import pytest

from evalharness.settings import HarnessSettings
from evalharness.transport import SutTransport

_API_DIR = Path(__file__).parent


@pytest.hookimpl(tryfirst=True)
def pytest_collection_modifyitems(items: list[pytest.Item]) -> None:
    """Mark any test in this folder as api, before marker selection runs, so none can be left unmarked."""
    for item in items:
        if item.path.is_relative_to(_API_DIR):
            item.add_marker(pytest.mark.api)


@pytest.fixture
def settings() -> HarnessSettings:
    """Read the harness settings from the EVAL_* environment of this run."""
    return HarnessSettings()


@pytest.fixture
async def sut(settings: HarnessSettings) -> AsyncIterator[SutTransport]:
    """Open a transport to the system under test for one test and close it afterwards."""
    async with SutTransport(settings) as transport:
        yield transport
