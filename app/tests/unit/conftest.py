"""Shared unit-test setup: settings that do not depend on the shell, a database the tests can switch off,
an embedder that needs no server, and an in-memory corpus store."""

import zlib
from collections.abc import AsyncGenerator, Callable, Sequence
from contextlib import AbstractAsyncContextManager, asynccontextmanager

import pytest
from pydantic import HttpUrl, SecretStr

from study_assistant.db import Connection, DatabaseUnavailableError
from study_assistant.embed.base import EmbeddingSpace
from study_assistant.embed.remote import EmbeddingServiceUnavailableError
from study_assistant.ingest.service import CorpusCounts, PreparedDocument, StoredCourse
from study_assistant.settings import (
    AdminSettings,
    DbSettings,
    EmbeddingServiceSettings,
    EmbeddingSettings,
    Settings,
)

ADMIN_KEY = "unit-test-admin-key"
# A valid embedding model choice; unit tests never download or load it.
EMBEDDING = EmbeddingSettings(
    model="intfloat/multilingual-e5-small",
    revision="614241f622f53c4eeff9890bdc4f31cfecc418b3",
    dimension=384,
    passage_prefix="passage: ",
    query_prefix="query: ",
)
# An address nothing listens on; unit tests never call the embedding server.
EMBEDDING_SERVICE = EmbeddingServiceSettings(url=HttpUrl("http://embedder.invalid"))


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


class FakeEmbedder:
    """Stands in for the embedding server: ``vector_of`` each text, every call recorded, and switches for a server
    that is down or answers with fewer vectors."""

    @staticmethod
    def vector_of(text: str) -> list[float]:
        """The vector this fake gives a text: made from the text alone, so a test can tell which text it is of."""
        return [float(zlib.crc32(text.encode("utf-8"))), 1.0]

    def __init__(self) -> None:
        """Start reachable and correct, with nothing recorded yet."""
        self.space = EmbeddingSpace.from_settings(EMBEDDING)
        self.calls: list[list[str]] = []
        self.up = True
        self.error: Exception | None = None
        self.drop_last = False

    async def aembed_documents(self, texts: list[str]) -> list[list[float]]:
        """Record the texts and return their vectors; raise ``error`` instead when it is set."""
        if self.error is not None:
            raise self.error
        self.calls.append(list(texts))
        vectors = [self.vector_of(text) for text in texts]
        return vectors[:-1] if self.drop_last else vectors

    async def aembed_query(self, text: str) -> list[float]:
        """Return the vector of one text."""
        return (await self.aembed_documents([text]))[0]

    async def ping(self) -> None:
        """Answer while up; otherwise fail the way the real client does when the server cannot be reached."""
        if not self.up:
            raise EmbeddingServiceUnavailableError("ConnectError")


class FakeCourseWriter:
    """One course's writes into a ``FakeCorpusStore``, recorded as events."""

    def __init__(self, store: FakeCorpusStore, course_id: str) -> None:
        """Write into ``store`` for ``course_id`` only."""
        self._store = store
        self._course_id = course_id
        self._documents = store.documents.setdefault(course_id, {})
        self._vectors = store.vectors.setdefault(course_id, {})

    async def stored(self) -> StoredCourse:
        """Return the stored editions of the course and its chunks without a vector."""
        return self._store.stored_course(self._course_id)

    async def save_course(self, title: str, language: str) -> None:
        """Keep the course's title and language."""
        self._store.courses[self._course_id] = (title, language)

    async def delete_documents(self, source_files: Sequence[str]) -> None:
        """Forget these documents and the vectors of chunks no stored document has any more, as the database's
        cascade does; an empty list is recorded too, so a test sees what the service asked for."""
        self._store.events.append(("delete", self._course_id, tuple(source_files)))
        for file in source_files:
            self._documents.pop(file, None)
        kept = {chunk.id for document in self._documents.values() for chunk in document.chunks}
        for chunk_id in self._vectors.keys() - kept:
            del self._vectors[chunk_id]

    async def insert_document(self, document: PreparedDocument) -> None:
        """Keep the document; storing one twice is an error, as the primary key makes it in the database."""
        file = document.source.file
        self._store.events.append(("insert", self._course_id, file))
        if file in self._documents:
            raise AssertionError(f"{file} is already stored")
        self._documents[file] = document

    async def insert_vectors(self, vectors: Sequence[tuple[str, list[float]]]) -> None:
        """Keep the vectors; a vector for a chunk that is not stored, or a second one, is an error, as the keys of
        the vector table make it in the database. An empty list is recorded too."""
        self._store.events.append(("vectors", self._course_id, tuple(chunk_id for chunk_id, _ in vectors)))
        stored = {chunk.id for document in self._documents.values() for chunk in document.chunks}
        for chunk_id, vector in vectors:
            if chunk_id not in stored or chunk_id in self._vectors:
                raise AssertionError(f"no vector can be stored for chunk {chunk_id}")
            self._vectors[chunk_id] = vector

    async def counts(self) -> CorpusCounts:
        """Count from what is stored, the way the database would."""
        return self._store.count(self._course_id)

    async def vector_count(self) -> int:
        """Count the course's stored vectors, the way the database would."""
        return self._store.count_vectors(self._course_id)


class FakeCorpusStore:
    """Keeps prepared documents and vectors per course in memory; a writer block that fails leaves nothing behind.

    ``before_write`` runs when a writer opens, after the ingestion read what is stored and computed its vectors:
    a test uses it to play another ingestion that changed the course meanwhile.
    """

    def __init__(self) -> None:
        """Start empty."""
        self.documents: dict[str, dict[str, PreparedDocument]] = {}
        self.vectors: dict[str, dict[str, list[float]]] = {}
        self.courses: dict[str, tuple[str, str]] = {}
        self.events: list[tuple[object, ...]] = []
        self.spaces: list[str] = []
        self.writers_opened = 0
        self.before_write: Callable[[], None] | None = None

    @asynccontextmanager
    async def course_writer(self, course_id: str, space: EmbeddingSpace) -> AsyncGenerator[FakeCourseWriter]:
        """Lend a writer; an exception restores the course's documents and vectors as they were, like a rollback."""
        if self.before_write is not None:
            self.before_write()
        self.writers_opened += 1
        self.spaces.append(space.id)
        documents_before = dict(self.documents.get(course_id, {}))
        vectors_before = dict(self.vectors.get(course_id, {}))
        try:
            yield FakeCourseWriter(self, course_id)
        except BaseException:
            self.documents[course_id] = documents_before
            self.vectors[course_id] = vectors_before
            raise

    async def stored(self, course_id: str, space: EmbeddingSpace) -> StoredCourse:
        """Return what is stored for the course, outside any writer."""
        self.spaces.append(space.id)
        return self.stored_course(course_id)

    async def counts(self, course_id: str) -> CorpusCounts:
        """Count what is stored for the course."""
        return self.count(course_id)

    def stored_course(self, course_id: str) -> StoredCourse:
        """The stored editions of the course and the ids of its stored chunks that have no vector."""
        documents = self.documents.get(course_id, {})
        vectors = self.vectors.get(course_id, {})
        return StoredCourse(
            editions={file: document.edition for file, document in documents.items()},
            chunks_without_vectors=frozenset(
                chunk.id for document in documents.values() for chunk in document.chunks if chunk.id not in vectors
            ),
        )

    def count(self, course_id: str) -> CorpusCounts:
        """Count documents, headed sections, objectives and chunks of the course."""
        documents = self.documents.get(course_id, {}).values()
        return CorpusCounts(
            documents=len(documents),
            sections=sum(1 for d in documents for s in d.structure.sections if s.id is not None),
            learning_objectives=sum(len(d.structure.objectives) for d in documents),
            chunks=sum(len(d.chunks) for d in documents),
        )

    async def vector_count(self, course_id: str, space: EmbeddingSpace) -> int:
        """Count the course's stored vectors outside any writer, recording the space asked for."""
        self.spaces.append(space.id)
        return self.count_vectors(course_id)

    def count_vectors(self, course_id: str) -> int:
        """Count the course's stored vectors."""
        return len(self.vectors.get(course_id, {}))


@pytest.fixture
def settings() -> Settings:
    """Valid settings with a placeholder password; explicit values win over any APP_* variables in the shell."""
    return Settings(
        db=DbSettings(password=SecretStr("unit-test-password")),
        embedding=EMBEDDING,
        embedding_service=EMBEDDING_SERVICE,
        admin=AdminSettings(api_key=SecretStr(ADMIN_KEY)),
    )


@pytest.fixture
def fake_db() -> FakeDatabase:
    """A fresh, reachable fake database for one test."""
    return FakeDatabase()


@pytest.fixture
def fake_embedder() -> FakeEmbedder:
    """A fresh, reachable fake embedder for one test."""
    return FakeEmbedder()


@pytest.fixture
def fake_store() -> FakeCorpusStore:
    """A fresh, empty corpus store for one test."""
    return FakeCorpusStore()


@pytest.fixture
def anyio_backend() -> str:
    """Run tests marked ``anyio`` on asyncio, the loop the app server uses; anyio comes with the web framework."""
    return "asyncio"
