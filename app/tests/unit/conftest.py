"""Shared unit-test setup: settings that do not depend on the shell, a database the tests can switch off,
and an in-memory corpus store."""

from collections.abc import AsyncGenerator, Sequence
from contextlib import AbstractAsyncContextManager, asynccontextmanager

import pytest
from pydantic import SecretStr

from study_assistant.db import Connection, DatabaseUnavailableError
from study_assistant.ingest.service import CorpusCounts, Edition, PreparedDocument
from study_assistant.settings import AdminSettings, DbSettings, EmbeddingSettings, Settings

ADMIN_KEY = "unit-test-admin-key"
# A valid embedding model choice; unit tests never download or load it.
EMBEDDING = EmbeddingSettings(
    model="intfloat/multilingual-e5-small",
    revision="614241f622f53c4eeff9890bdc4f31cfecc418b3",
    dimension=384,
    passage_prefix="passage: ",
    query_prefix="query: ",
)


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

    def connection(self) -> AbstractAsyncContextManager[Connection]:
        """Refuse: code under unit test reaches stored data through a fake store, never through SQL."""
        raise AssertionError("a unit test must not open a database connection")


class FakeCourseWriter:
    """One course's writes into a ``FakeCorpusStore``, recorded as events."""

    def __init__(self, store: FakeCorpusStore, course_id: str) -> None:
        """Write into ``store`` for ``course_id`` only."""
        self._store = store
        self._course_id = course_id
        self._documents = store.documents.setdefault(course_id, {})

    async def editions(self) -> dict[str, Edition]:
        """Return the stored editions of the course."""
        return {file: document.edition for file, document in self._documents.items()}

    async def save_course(self, title: str, language: str) -> None:
        """Keep the course's title and language."""
        self._store.courses[self._course_id] = (title, language)

    async def delete_documents(self, source_files: Sequence[str]) -> None:
        """Forget these documents; an empty list is recorded too, so a test sees what the service asked for."""
        self._store.events.append(("delete", self._course_id, tuple(source_files)))
        for file in source_files:
            self._documents.pop(file, None)

    async def insert_document(self, document: PreparedDocument) -> None:
        """Keep the document; storing one twice is an error, as the primary key makes it in the database."""
        file = document.source.file
        self._store.events.append(("insert", self._course_id, file))
        if file in self._documents:
            raise AssertionError(f"{file} is already stored")
        self._documents[file] = document

    async def counts(self) -> CorpusCounts:
        """Count from what is stored, the way the database would."""
        return self._store.count(self._course_id)


class FakeCorpusStore:
    """Keeps prepared documents per course in memory; a writer block that fails leaves nothing behind."""

    def __init__(self) -> None:
        """Start empty."""
        self.documents: dict[str, dict[str, PreparedDocument]] = {}
        self.courses: dict[str, tuple[str, str]] = {}
        self.events: list[tuple[object, ...]] = []
        self.writers_opened = 0

    @asynccontextmanager
    async def course_writer(self, course_id: str) -> AsyncGenerator[FakeCourseWriter]:
        """Lend a writer; an exception restores the course's documents as they were, like a rollback."""
        self.writers_opened += 1
        before = dict(self.documents.get(course_id, {}))
        try:
            yield FakeCourseWriter(self, course_id)
        except BaseException:
            self.documents[course_id] = before
            raise

    async def counts(self, course_id: str) -> CorpusCounts:
        """Count what is stored for the course."""
        return self.count(course_id)

    def count(self, course_id: str) -> CorpusCounts:
        """Count documents, headed sections, objectives and chunks of the course."""
        documents = self.documents.get(course_id, {}).values()
        return CorpusCounts(
            documents=len(documents),
            sections=sum(1 for d in documents for s in d.structure.sections if s.id is not None),
            learning_objectives=sum(len(d.structure.objectives) for d in documents),
            chunks=sum(len(d.chunks) for d in documents),
        )


@pytest.fixture
def settings() -> Settings:
    """Valid settings with a placeholder password; explicit values win over any APP_* variables in the shell."""
    return Settings(
        db=DbSettings(password=SecretStr("unit-test-password")),
        embedding=EMBEDDING,
        admin=AdminSettings(api_key=SecretStr(ADMIN_KEY)),
    )


@pytest.fixture
def fake_db() -> FakeDatabase:
    """A fresh, reachable fake database for one test."""
    return FakeDatabase()


@pytest.fixture
def fake_store() -> FakeCorpusStore:
    """A fresh, empty corpus store for one test."""
    return FakeCorpusStore()


@pytest.fixture
def anyio_backend() -> str:
    """Run tests marked ``anyio`` on asyncio, the loop the app server uses; anyio comes with the web framework."""
    return "asyncio"
