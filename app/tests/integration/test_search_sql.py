"""The search store's two statements against a real PostgreSQL with pgvector: which passages a search reads and in
what order, and how a course's chunks and vectors are counted per embedding space.

The unit tests of the search rules use a store that returns given passages closest first and given counts; these
tests check that the real statements produce exactly that. Courses get their rows through a real ingestion, and a
new embedding space through the migration run, the way the stack gets them.
"""

import math
import shutil
from collections.abc import AsyncIterator
from pathlib import Path
from typing import TYPE_CHECKING

import pytest

from study_assistant.courses.models import CourseManifest
from study_assistant.courses.registry import load_course
from study_assistant.db import PostgresDatabase
from study_assistant.ingest.chunker import ChunkSize
from study_assistant.ingest.repository import PostgresCorpusStore
from study_assistant.ingest.service import CourseIngestor
from study_assistant.retrieval.repository import Coverage, PostgresSearchStore

if TYPE_CHECKING:
    from .conftest import FixedLengthEmbedder, MigratedDatabase

from .conftest import APP_ROLE, APP_ROLE_PASSWORD

pytestmark = [pytest.mark.database, pytest.mark.anyio]

# The repository's own course folder: app/tests/integration/<this file> -> repository root.
REPO_COURSES = Path(__file__).resolve().parents[3] / "courses"
SIZE = ChunkSize(size=1000, overlap=100)
DEMO_CHUNKS = 12
# A second course with the demo course's files, so both courses hold passages with the same texts and scores.
OTHER_COURSE = "demo_copy"
# An invented revision of the configured model that only this file configures, so its space is new here.
NEW_REVISION = "fedcba9876543210fedcba9876543210fedcba98"


@pytest.fixture
async def search_store(postgres: MigratedDatabase) -> AsyncIterator[PostgresSearchStore]:
    """The app's search store, connected as the app's role through the app's connection pool."""
    database = PostgresDatabase(postgres.settings(APP_ROLE, APP_ROLE_PASSWORD))
    await database.open()
    try:
        yield PostgresSearchStore(database)
    finally:
        await database.close()


async def _ingest_two_courses(store: PostgresCorpusStore, embedder: FixedLengthEmbedder, root: Path) -> None:
    """Ingest the demo course and a copy of it under another id, both from folders under ``root``."""
    demo = load_course(REPO_COURSES, "demo")
    for course_id in ("demo", OTHER_COURSE):
        shutil.copytree(REPO_COURSES / "demo", root / course_id)
        manifest: CourseManifest = demo.model_copy(update={"id": course_id})
        await CourseIngestor(store, embedder, root, SIZE).ingest(manifest)


def _cosine(a: list[float], b: list[float]) -> float:
    """Cosine similarity of two vectors, computed here without the database."""
    dot = sum(x * y for x, y in zip(a, b, strict=True))
    return dot / (math.sqrt(sum(x * x for x in a)) * math.sqrt(sum(y * y for y in b)))


@pytest.mark.covers("US-006", "AC-1")
async def test_nearest_returns_the_k_closest_passages_of_the_requested_course_with_their_similarity(
    postgres: MigratedDatabase,
    store: PostgresCorpusStore,
    search_store: PostgresSearchStore,
    embedder: FixedLengthEmbedder,
    tmp_path: Path,
) -> None:
    """Ingests the demo course and a copy of it with the same texts, then asks for the 5 passages of the demo
    course closest to a question.

    Expects demo chunks only, and the scores of the 5 most similar demo chunks, computed here from the same vectors,
    highest first. Catches a statement without the course filter (the copy has passages with the same scores), the
    wrong sort direction, a missing limit, and a distance reported as the score.
    """
    await _ingest_two_courses(store, embedder, tmp_path)
    with postgres.owner_connection() as conn:
        demo_chunks: dict[str, str] = dict(
            conn.execute("SELECT chunk_id, text FROM chunks WHERE course_id = 'demo'").fetchall()
        )
    query = await embedder.aembed_query("How many dice does a player roll on the first turn?")
    passage_vectors = await embedder.aembed_documents(list(demo_chunks.values()))
    expected = sorted((_cosine(query, vector) for vector in passage_vectors), reverse=True)[:5]

    passages = await search_store.nearest("demo", embedder.space, query, k=5)

    foreign = [p.chunk_id for p in passages if p.chunk_id not in demo_chunks]
    assert foreign == [], f"the search of the demo course returned chunks of another course: {foreign}"
    scores = [p.score for p in passages]
    assert scores == pytest.approx(expected, abs=1e-6), f"expected the top-5 similarities {expected}, got {scores}"


@pytest.mark.covers("US-006", "AC-5")
async def test_coverage_counts_the_courses_chunks_and_those_with_a_vector_in_the_given_space(
    postgres: MigratedDatabase,
    store: PostgresCorpusStore,
    search_store: PostgresSearchStore,
    embedder: FixedLengthEmbedder,
    tmp_path: Path,
) -> None:
    """Ingests the demo course and its copy, then configures a new revision of the model, so the migration run
    prepares a new, still empty space.

    Expects 12 chunks with 12 vectors for the demo course in the first space and 12 chunks with 0 vectors in the
    new one. Catches counts that include the other course, vectors counted in another space's table, and chunks
    counted as embedded without a vector, any of which would let a search run over part of a course.
    """
    await _ingest_two_courses(store, embedder, tmp_path)
    new_space = embedder.with_settings(embedder.settings.model_copy(update={"revision": NEW_REVISION})).space
    assert not postgres.space_registered(new_space), "another test already prepared this space"
    postgres.migrate(embedder.settings.model_copy(update={"revision": NEW_REVISION}))

    counted = (
        await search_store.coverage("demo", embedder.space),
        await search_store.coverage("demo", new_space),
    )

    expected = (Coverage(DEMO_CHUNKS, DEMO_CHUNKS), Coverage(DEMO_CHUNKS, 0))
    assert counted == expected, f"expected {expected} for the first and the new space, got {counted}"
