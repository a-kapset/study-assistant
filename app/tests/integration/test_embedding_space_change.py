"""A change of the embedding space, the way it happens in the stack: the configuration names another space, the
migration run prepares it, and the next ingestion fills it, all against a real PostgreSQL with pgvector.

The unit test of the same rule plays the change by clearing the in-memory store's vectors. Here nothing is
cleared: the new space really comes from the migration run, and the old one stays in the database next to it.
"""

from pathlib import Path
from typing import TYPE_CHECKING

import pytest

from study_assistant.courses.registry import load_course
from study_assistant.ingest.chunker import ChunkSize
from study_assistant.ingest.repository import PostgresCorpusStore
from study_assistant.ingest.service import CourseIngestor, DocumentReport

if TYPE_CHECKING:
    from .conftest import FixedLengthEmbedder, MigratedDatabase

pytestmark = [pytest.mark.database, pytest.mark.anyio]

# The repository's own course folder: app/tests/integration/<this file> -> repository root.
REPO_COURSES = Path(__file__).resolve().parents[3] / "courses"
SIZE = ChunkSize(size=1000, overlap=100)
DEMO_FILE = "vardolin_rules.pdf"
DEMO_CHUNKS = 12
# An invented later revision of the configured model; only this test configures it, so its space is new here.
NEW_REVISION = "0123456789abcdef0123456789abcdef01234567"


@pytest.mark.covers("US-004", "AC-19")
async def test_a_new_embedding_space_gets_vectors_for_the_stored_chunks_and_the_old_space_keeps_its_own(
    postgres: MigratedDatabase, store: PostgresCorpusStore, embedder: FixedLengthEmbedder
) -> None:
    """Ingests the demo course, then configures a new revision of the model: the migration run prepares its space
    and the app, now with an embedder of that space, ingests the same files again.

    Expects the outcome "embedded" for the demo document, a vector for each of its 12 chunks in the new space,
    and the 12 vectors of the old space still there. Catches a space change that leaves the stored chunks without
    vectors, that rebuilds the unchanged document instead of only adding its vectors, and a migration run that
    removes the vectors of the old space when it prepares a new one. The space must not be registered before
    the run, or the run would prepare nothing and the test would not see a change at all.
    """
    demo = load_course(REPO_COURSES, "demo")
    await CourseIngestor(store, embedder, REPO_COURSES, SIZE).ingest(demo)
    new_settings = embedder.settings.model_copy(update={"revision": NEW_REVISION})
    new_embedder = embedder.with_settings(new_settings)
    assert not postgres.space_registered(new_embedder.space), "another test already prepared this space"
    postgres.migrate(new_settings)

    report = await CourseIngestor(store, new_embedder, REPO_COURSES, SIZE).ingest(demo)

    assert report.documents == [DocumentReport(source_file=DEMO_FILE, outcome="embedded")], (
        f"an unchanged document in a new space should only get its vectors, got {report.documents}"
    )
    new_vectors = await store.vector_count("demo", new_embedder.space)
    assert new_vectors == DEMO_CHUNKS, f"the new space has {new_vectors} vectors for {DEMO_CHUNKS} chunks"
    old_vectors = await store.vector_count("demo", embedder.space)
    assert old_vectors == DEMO_CHUNKS, f"the old space lost vectors in the change: {old_vectors} of {DEMO_CHUNKS}"
