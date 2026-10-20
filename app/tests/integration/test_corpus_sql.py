"""The corpus store's SQL and the migration run against a real PostgreSQL with pgvector.

The unit tests of ingestion run on an in-memory store that counts, filters by course and keeps writers apart the
way the database is meant to. These tests check that the real statements do the same. Every course here is
written by the app's own ingestion from the demo course's files; a second course is the demo course under
another id, so both courses have files of the same names.
"""

import shutil
from pathlib import Path
from typing import TYPE_CHECKING

import anyio
import pytest

from study_assistant.courses.registry import load_course
from study_assistant.embed.base import EmbeddingSpace
from study_assistant.ingest.chunker import ChunkSize
from study_assistant.ingest.repository import PostgresCorpusStore
from study_assistant.ingest.service import CorpusCounts, CourseIngestor

if TYPE_CHECKING:
    from .conftest import FixedLengthEmbedder, MigratedDatabase

pytestmark = [pytest.mark.database, pytest.mark.anyio]

# The repository's own course folder: app/tests/integration/<this file> -> repository root.
REPO_COURSES = Path(__file__).resolve().parents[3] / "courses"
SIZE = ChunkSize(size=1000, overlap=100)
OTHER_COURSE = "demo_copy"
# The demo course as its manifest describes it: 4 units, 11 objectives, 12 chunks at 1000/100.
DEMO_COUNTS = CorpusCounts(documents=1, sections=4, learning_objectives=11, chunks=12)


async def _ingest_demo_and_its_copy(store: PostgresCorpusStore, embedder: FixedLengthEmbedder, tmp_path: Path) -> None:
    """Ingest the demo course, then the same files as a second course, both with a vector for every chunk."""
    demo = load_course(REPO_COURSES, "demo")
    await CourseIngestor(store, embedder, REPO_COURSES, SIZE).ingest(demo)
    shutil.copytree(REPO_COURSES / "demo", tmp_path / OTHER_COURSE)
    await CourseIngestor(store, embedder, tmp_path, SIZE).ingest(demo.model_copy(update={"id": OTHER_COURSE}))


def _chunk_ids(postgres: MigratedDatabase, course_id: str) -> frozenset[str]:
    """Read a course's chunk ids as the database owner."""
    with postgres.owner_connection() as conn:
        rows = conn.execute("SELECT chunk_id FROM chunks WHERE course_id = %s", (course_id,)).fetchall()
    return frozenset(row[0] for row in rows)


@pytest.mark.covers("US-004", "AC-2")
async def test_course_counts_hold_only_that_courses_rows(
    store: PostgresCorpusStore, embedder: FixedLengthEmbedder, tmp_path: Path
) -> None:
    """Stores the demo course and a copy of it under another id, then counts the demo course.

    Expects the demo course's own structure: 1 document, 4 headed sections, 11 objectives, 12 chunks. Catches a
    count that loses its course filter, which would count both courses, and a section count that also counts the
    text before the first heading.
    """
    await _ingest_demo_and_its_copy(store, embedder, tmp_path)

    counts = await store.counts("demo")

    assert counts == DEMO_COUNTS, f"the demo course's counts include rows that are not its own: {counts}"


@pytest.mark.covers("US-004", "AC-2")
async def test_vector_count_holds_only_that_courses_vectors(
    store: PostgresCorpusStore, embedder: FixedLengthEmbedder, tmp_path: Path
) -> None:
    """Stores the demo course and its copy with a vector for every chunk, then counts the demo course's vectors.

    Expects 12, one per demo chunk. Catches a vector count that loses its course filter: it would report the
    vectors of both courses, and the course endpoints would show more vectors than the course has.
    """
    await _ingest_demo_and_its_copy(store, embedder, tmp_path)

    count = await store.vector_count("demo", embedder.space)

    assert count == DEMO_COUNTS.chunks, f"the demo course has {DEMO_COUNTS.chunks} chunks but {count} vectors counted"


@pytest.mark.covers("US-004", "AC-19")
async def test_chunks_without_a_vector_are_the_courses_chunks_missing_from_the_asked_space(
    postgres: MigratedDatabase, store: PostgresCorpusStore, embedder: FixedLengthEmbedder, tmp_path: Path
) -> None:
    """Stores both courses with vectors, then lets the migration run prepare a second space, as a changed
    passage prefix in the configuration does.

    Expects no demo chunk missing in the first space and every demo chunk, and only those, missing in the new one.
    Catches a query that looks for vectors the wrong way round (chunks with a vector instead of without), or that
    loses its course filter and would ask for vectors of another course's chunks.
    """
    await _ingest_demo_and_its_copy(store, embedder, tmp_path)
    new_settings = embedder.settings.model_copy(update={"passage_prefix": "passage:"})
    postgres.migrate(new_settings)
    new_space = EmbeddingSpace.from_settings(new_settings)

    in_first_space = await store.stored("demo", embedder.space)
    in_new_space = await store.stored("demo", new_space)

    assert in_first_space.chunks_without_vectors == frozenset(), (
        f"{len(in_first_space.chunks_without_vectors)} demo chunks reported without a vector in the space they have"
    )
    assert in_new_space.chunks_without_vectors == _chunk_ids(postgres, "demo"), (
        f"the new space should miss exactly the demo chunks, got {len(in_new_space.chunks_without_vectors)} ids"
    )


async def test_a_second_writer_of_the_same_course_waits_for_the_first(
    store: PostgresCorpusStore, embedder: FixedLengthEmbedder
) -> None:
    """Opens a course writer for the demo course and, while it is open, tries to open a second one.

    Expects the second writer to wait while the first is open and to get in once the first has ended. Catches
    a writer that no longer takes the course's lock: two ingestions of one course would then plan from the same
    stored state and both write the same documents.
    """
    second_entered = anyio.Event()

    async def open_second_writer() -> None:
        """Open a second writer of the demo course and note when it gets in."""
        async with store.course_writer("demo", embedder.space):
            second_entered.set()

    async with anyio.create_task_group() as tasks:
        async with store.course_writer("demo", embedder.space):
            tasks.start_soon(open_second_writer)
            with anyio.move_on_after(1):
                await second_entered.wait()
            assert not second_entered.is_set(), "a second writer of the course got in while the first was open"
        with anyio.fail_after(5):
            await second_entered.wait()


def test_the_app_role_may_write_the_vectors_of_a_space_the_run_just_created(
    postgres: MigratedDatabase, embedder: FixedLengthEmbedder
) -> None:
    """Runs the migration with a changed query prefix, which names a space no earlier run created, then asks the
    database whether the app's role may insert into that space's new vector table.

    Expects yes. Catches a run that grants the role its rights before it creates the space's table: the grants
    reach only tables that exist, so the app would fail to store any vector after a new space is configured.
    """
    new_settings = embedder.settings.model_copy(update={"query_prefix": "query:"})
    postgres.migrate(new_settings)
    table = EmbeddingSpace.from_settings(new_settings).vector_table

    with postgres.owner_connection() as conn:
        row = conn.execute("SELECT has_table_privilege(%s, %s, 'INSERT')", (postgres.app_role, table)).fetchone()

    assert row == (True,), f"the app role may not insert into {table}"
