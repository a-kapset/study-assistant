"""Fixtures and marking for tests that talk to a running system under test."""

from collections.abc import AsyncIterator, Iterator
from pathlib import Path

import pytest

from evalharness.corpus_db import CorpusReader
from evalharness.settings import HarnessSettings, StackSettings, load_stack_settings
from evalharness.testdata.cleanup import leftovers, remove_all
from evalharness.testdata.corpus import CorpusTestData, connect_writer
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


@pytest.fixture
def stack(settings: HarnessSettings) -> StackSettings:
    """Read the values shared with the stack (database port, roles, admin key) from its .env file."""
    return load_stack_settings(settings.stack_env_file)


@pytest.fixture
def admin_headers(stack: StackSettings) -> dict[str, str]:
    """The header that carries the stack's admin key."""
    return {"X-Admin-Key": stack.admin_api_key.get_secret_value()}


@pytest.fixture
def corpus(settings: HarnessSettings, stack: StackSettings) -> Iterator[CorpusReader]:
    """Open a read-only connection to the stack's database for one test."""
    with CorpusReader(settings, stack) as reader:
        yield reader


@pytest.fixture
def test_data(settings: HarnessSettings, stack: StackSettings) -> Iterator[CorpusTestData]:
    """Give the test a way to create test data in the app's database, and remove all of it afterwards.

    Before the test, data left by an earlier run (for example one that was killed) is deleted. After the test,
    also when it failed, all test data is deleted again and the database is checked: if anything is left, the
    test ends with an error that lists it.
    """
    remove_all(settings, stack)
    with connect_writer(settings, stack) as conn:
        yield CorpusTestData(conn)
    remove_all(settings, stack)
    left = leftovers(settings, stack)
    assert not left, f"test data is still in the database after cleanup: {left}"
