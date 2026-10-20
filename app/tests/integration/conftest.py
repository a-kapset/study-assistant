"""Shared setup for the tests that need a real PostgreSQL: a throwaway database in Docker, prepared by the app's
own migration run, and the app's corpus store connected as the app's role.

The container starts once per test session, only when a test asks for it, and is removed at the end, so the
stack's database is never touched. Tests that use it carry the ``database`` marker.
"""

import asyncio
import os
from collections.abc import AsyncIterator, Iterator
from dataclasses import dataclass

import psycopg
import pytest
from psycopg.conninfo import make_conninfo
from pydantic import SecretStr
from testcontainers.community.postgres import PostgresContainer

from study_assistant import migrate
from study_assistant.db import PostgresDatabase
from study_assistant.embed.base import EmbeddingSpace
from study_assistant.ingest.repository import PostgresCorpusStore
from study_assistant.settings import ENV_PREFIX, DbSettings, EmbeddingSettings

# The image the stack runs, so the tests meet the same PostgreSQL and pgvector versions.
POSTGRES_IMAGE = "pgvector/pgvector:0.8.7-pg18-trixie"
DB_NAME = "study_assistant"
OWNER = "study_owner"
OWNER_PASSWORD = "integration-owner-password"
APP_ROLE = "app_rw"
APP_ROLE_PASSWORD = "integration-app-password"
READER_ROLE = "app_ro"
READER_ROLE_PASSWORD = "integration-reader-password"
# The space the migration run prepares first; the vectors of every ingestion in these tests go there.
EMBEDDING = EmbeddingSettings(
    model="intfloat/multilingual-e5-small",
    revision="614241f622f53c4eeff9890bdc4f31cfecc418b3",
    dimension=384,
    passage_prefix="passage: ",
    query_prefix="query: ",
)


@dataclass(frozen=True)
class MigratedDatabase:
    """The throwaway database: where it listens, and the owner's tools to prepare and clean it."""

    host: str
    port: int
    app_role: str = APP_ROLE

    def settings(self, user: str, password: str) -> DbSettings:
        """Connection settings for one role of this database."""
        return DbSettings(host=self.host, port=self.port, name=DB_NAME, user=user, password=SecretStr(password))

    def owner_connection(self) -> psycopg.Connection:
        """Open an autocommit connection as the database owner, for setup and for reading what the app role cannot."""
        conninfo = make_conninfo(host=self.host, port=self.port, dbname=DB_NAME, user=OWNER, password=OWNER_PASSWORD)
        return psycopg.connect(conninfo, autocommit=True)

    def migrate(self, embedding: EmbeddingSettings) -> None:
        """Run the app's migration entry point against this database with ``embedding`` as the configured space.

        The run reads its settings from ``APP_*`` variables, as in the stack, so every such variable of the shell
        is taken away for the run and only the ones for this database are set. A failed run fails the test.
        """
        variables = {
            "DB__HOST": self.host,
            "DB__PORT": str(self.port),
            "DB__NAME": DB_NAME,
            "DB__USER": OWNER,
            "DB__PASSWORD": OWNER_PASSWORD,
            "SERVER_ROLE__NAME": APP_ROLE,
            "SERVER_ROLE__PASSWORD": APP_ROLE_PASSWORD,
            "READER_ROLE__NAME": READER_ROLE,
            "READER_ROLE__PASSWORD": READER_ROLE_PASSWORD,
            "EMBEDDING__MODEL": embedding.model,
            "EMBEDDING__REVISION": embedding.revision,
            "EMBEDDING__DIMENSION": str(embedding.dimension),
            "EMBEDDING__PASSAGE_PREFIX": embedding.passage_prefix,
            "EMBEDDING__QUERY_PREFIX": embedding.query_prefix,
        }
        with pytest.MonkeyPatch.context() as env:
            for name in list(os.environ):
                if name.upper().startswith(ENV_PREFIX):
                    env.delenv(name)
            for name, value in variables.items():
                env.setenv(ENV_PREFIX + name, value)
            exit_code = migrate.main()
        assert exit_code == 0, f"the migration run for space {EmbeddingSpace.from_settings(embedding).id} failed"

    def space_registered(self, space: EmbeddingSpace) -> bool:
        """Tell whether a migration run has already registered ``space`` in this database."""
        with self.owner_connection() as conn:
            row = conn.execute("SELECT 1 FROM embedding_spaces WHERE space_id = %s", (space.id,)).fetchone()
        return row is not None

    def empty_corpus(self) -> None:
        """Delete every course and, through the foreign keys, every row that belongs to one, vectors included."""
        with self.owner_connection() as conn:
            conn.execute("TRUNCATE courses CASCADE")


class FixedLengthEmbedder:
    """Stands in for the embedding server: gives every text a vector of the configured length, made from its text.

    The tests here check how vectors are stored and counted, not what they are, so any vector of the right length
    the table accepts will do.
    """

    def __init__(self, settings: EmbeddingSettings) -> None:
        """Compute vectors of ``settings.dimension`` numbers in the space the settings name."""
        self.settings = settings
        self._space = EmbeddingSpace.from_settings(settings)
        self._dimension = settings.dimension

    @property
    def space(self) -> EmbeddingSpace:
        """The space of the settings this embedder was made with."""
        return self._space

    def with_settings(self, settings: EmbeddingSettings) -> FixedLengthEmbedder:
        """Return an embedder of another configured space, as the app has after a restart with that configuration."""
        return FixedLengthEmbedder(settings)

    async def aembed_documents(self, texts: list[str]) -> list[list[float]]:
        """Return one vector per text: its length in the first place, ones after it."""
        return [[float(len(text))] + [1.0] * (self._dimension - 1) for text in texts]

    async def aembed_query(self, text: str) -> list[float]:
        """Return the vector of one text."""
        return (await self.aembed_documents([text]))[0]

    async def ping(self) -> None:
        """Always answer: there is no server to be down."""


@pytest.fixture
def anyio_backend() -> tuple[str, dict[str, object]]:
    """Run the async tests on a selector event loop: on Windows the async database driver refuses the default one.

    On Linux the selector loop is already the default.
    """
    return "asyncio", {"loop_factory": asyncio.SelectorEventLoop}


@pytest.fixture(scope="session")
def postgres() -> Iterator[MigratedDatabase]:
    """Start the throwaway database and prepare it with the app's migration run, as the stack's first start does.

    Without Docker the container cannot start and every test that needs it fails with the reason; none is skipped.
    """
    with PostgresContainer(POSTGRES_IMAGE, username=OWNER, password=OWNER_PASSWORD, dbname=DB_NAME, driver=None) as pg:
        database = MigratedDatabase(host=pg.get_container_host_ip(), port=int(pg.get_exposed_port(5432)))
        database.migrate(EMBEDDING)
        yield database


@pytest.fixture
async def store(postgres: MigratedDatabase) -> AsyncIterator[PostgresCorpusStore]:
    """The app's corpus store on an empty corpus, connected as the app's role through the app's connection pool."""
    postgres.empty_corpus()
    database = PostgresDatabase(postgres.settings(APP_ROLE, APP_ROLE_PASSWORD))
    await database.open()
    try:
        yield PostgresCorpusStore(database)
    finally:
        await database.close()


@pytest.fixture
def embedder() -> FixedLengthEmbedder:
    """An embedder of the space the migration run prepared first."""
    return FixedLengthEmbedder(EMBEDDING)
