"""Shared unit-test setup: settings that do not depend on the shell, a database the tests can switch off,
an embedder that needs no server, a chat model that needs no provider, and an in-memory corpus store."""

import zlib
from collections.abc import AsyncGenerator, Callable, Sequence
from contextlib import AbstractAsyncContextManager, asynccontextmanager
from dataclasses import replace

import pytest
from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.language_models import BaseChatModel, LanguageModelInput
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.runnables import Runnable, RunnableLambda
from pydantic import BaseModel, Field, HttpUrl, SecretStr

from study_assistant.db import Connection, DatabaseUnavailableError
from study_assistant.embed.base import EmbeddingSpace
from study_assistant.embed.remote import EmbeddingServiceUnavailableError
from study_assistant.ingest.service import CorpusCounts, PreparedDocument, StoredCourse
from study_assistant.retrieval.repository import Coverage, Passage
from study_assistant.settings import (
    AdminSettings,
    DbSettings,
    EmbeddingServiceSettings,
    EmbeddingSettings,
    LlmSettings,
    OpenAISettings,
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
# A valid answer model and an address nothing listens on; unit tests reach the provider only through a mock transport.
LLM = LlmSettings(answer="openai/gpt-5.4-mini")
OPENAI = OpenAISettings(api_key=SecretStr("unit-test-openai-key"), base_url=HttpUrl("http://llm.invalid/v1"))


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


# The coverage of a course whose 12 chunks all have a vector in the configured space.
EMBEDDED = Coverage(chunks=12, vectors=12)
PASSAGE = Passage(
    chunk_id="c1", section_id="1", source_file="rules.pdf", page_start=1, page_end=1, text="Roll two dice.", score=0.9
)


class FakeSearchStore:
    """Stands in for the database: a fully embedded course whose passages are given closest first, as the real
    store returns them, and a record of the number of hits each search asked for."""

    def __init__(self, passages: list[Passage], coverage: Coverage = EMBEDDED) -> None:
        """Hold the passages every search returns and the coverage every course has."""
        self.passages = passages
        self.course_coverage = coverage
        self.asked_k: list[int] = []

    async def coverage(self, course_id: str, space: EmbeddingSpace) -> Coverage:
        """Return the coverage set for this test, whatever the course."""
        return self.course_coverage

    async def nearest(self, course_id: str, space: EmbeddingSpace, vector: list[float], k: int) -> list[Passage]:
        """Record ``k`` and return the first ``k`` passages."""
        self.asked_k.append(k)
        return self.passages[:k]


def scored_passages(*scores: float) -> list[Passage]:
    """Passages with the given scores, closest first, each with its own chunk id."""
    return [replace(PASSAGE, chunk_id=f"c{i}", score=score) for i, score in enumerate(scores, start=1)]


def responses_api_reply(content: list[dict[str, object]]) -> dict[str, object]:
    """Return the body the OpenAI Responses API sends back for one assistant message with these content blocks,
    e.g. ``[{"type": "output_text", "text": "...", "annotations": []}]`` or ``[{"type": "refusal", "refusal": "..."}]``.
    """
    return {
        "id": "resp_unit",
        "object": "response",
        "created_at": 0,
        "status": "completed",
        "model": "gpt-5.4-mini",
        "output": [
            {"type": "message", "id": "msg_unit", "role": "assistant", "status": "completed", "content": content}
        ],
        "parallel_tool_calls": True,
        "tool_choice": "auto",
        "tools": [],
        "usage": {
            "input_tokens": 5,
            "output_tokens": 3,
            "total_tokens": 8,
            "input_tokens_details": {"cached_tokens": 0},
            "output_tokens_details": {"reasoning_tokens": 0},
        },
    }


class FakeChatModel(BaseChatModel):
    """Stands in for the answer model: records the messages of every call and replies with ``reply``, or raises
    ``error`` instead when it is set."""

    reply: str = "Roll two dice."
    error: Exception | None = None
    calls: list[list[BaseMessage]] = Field(default_factory=list)

    @property
    def _llm_type(self) -> str:
        """Name this model in LangChain's own records."""
        return "fake-chat-model"

    def with_structured_output(
        self, schema: dict[str, object] | type, *, include_raw: bool = False, **kwargs: object
    ) -> Runnable[LanguageModelInput, dict[str, object] | BaseModel]:
        """Return a runnable that fails when called: this fake replies with plain text only, so a test of structured
        replies uses the real client with a mock transport instead. The app builds its structured runnable at
        startup, so this method must work even when the test never asks a grounded question."""

        def refuse(_messages: LanguageModelInput) -> dict[str, object] | BaseModel:
            """Fail loudly, naming the fake, instead of answering in a form it cannot honour."""
            raise AssertionError("FakeChatModel gives plain text only; test structured replies on the real client")

        return RunnableLambda(refuse)

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: object,
    ) -> ChatResult:
        """Record the messages, then raise ``error`` or reply with ``reply``."""
        self.calls.append(list(messages))
        if self.error is not None:
            raise self.error
        return ChatResult(generations=[ChatGeneration(message=AIMessage(self.reply))])


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
        llm=LLM,
        openai=OPENAI,
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
def fake_chat_model() -> FakeChatModel:
    """A fresh fake answer model for one test, replying with a fixed text."""
    return FakeChatModel()


@pytest.fixture
def fake_store() -> FakeCorpusStore:
    """A fresh, empty corpus store for one test."""
    return FakeCorpusStore()


@pytest.fixture
def anyio_backend() -> str:
    """Run tests marked ``anyio`` on asyncio, the loop the app server uses; anyio comes with the web framework."""
    return "asyncio"
