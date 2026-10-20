"""Ingestion's deletes against a real PostgreSQL: a document dropped from the manifest, a source file saved again,
and a second course with files of the same names.

Each state is reached the way the app meets it: the course folder and the manifest change, then ingestion runs
again. The course folder is a copy of the demo course in a temporary folder, so its files can change. A file
"saved again" gets bytes after the end of the PDF: the text stays the same and the file's hash changes, as when a
PDF is exported once more.
"""

import shutil
from pathlib import Path
from typing import TYPE_CHECKING

import pytest

from study_assistant.courses.models import CourseManifest
from study_assistant.courses.registry import load_course
from study_assistant.ingest.chunker import ChunkSize
from study_assistant.ingest.repository import PostgresCorpusStore
from study_assistant.ingest.service import CorpusCounts, CourseIngestor, DocumentReport, IngestReport

if TYPE_CHECKING:
    from .conftest import FixedLengthEmbedder, MigratedDatabase

pytestmark = [pytest.mark.database, pytest.mark.anyio]

# The repository's own course folder: app/tests/integration/<this file> -> repository root.
REPO_COURSES = Path(__file__).resolve().parents[3] / "courses"
SIZE = ChunkSize(size=1000, overlap=100)
DEMO_FILE = "vardolin_rules.pdf"
# A second source of the course: the demo PDF saved again, so its chunks get ids of their own.
EXTRA_FILE = "vardolin_rules_extra.pdf"
OTHER_COURSE = "demo_copy"
# The demo PDF as its manifest describes it: 1 document, 4 units, 11 objectives, 12 chunks at 1000/100.
DEMO_COUNTS = CorpusCounts(documents=1, sections=4, learning_objectives=11, chunks=12)


def _save_again(path: Path) -> None:
    """Change a PDF's bytes but not its text, as a new export of the same document does."""
    with path.open("ab") as pdf:
        pdf.write(b"\n% saved again\n")


def _course_folder(root: Path, course_id: str) -> Path:
    """Copy the demo course's files to ``root/course_id`` and add the extra source next to the demo PDF."""
    folder = root / course_id
    shutil.copytree(REPO_COURSES / "demo", folder)
    shutil.copyfile(folder / DEMO_FILE, folder / EXTRA_FILE)
    _save_again(folder / EXTRA_FILE)
    return folder


def _manifest(course_id: str, *files: str) -> CourseManifest:
    """The demo course's manifest under ``course_id``, listing the given files with the demo PDF's rules."""
    demo = load_course(REPO_COURSES, "demo")
    sources = tuple(demo.sources[0].model_copy(update={"file": file}) for file in files)
    return demo.model_copy(update={"id": course_id, "sources": sources})


async def _ingest(
    store: PostgresCorpusStore, embedder: FixedLengthEmbedder, root: Path, manifest: CourseManifest
) -> IngestReport:
    """Ingest a course from its folder under ``root``, with a vector for every new chunk."""
    return await CourseIngestor(store, embedder, root, SIZE).ingest(manifest)


def _stored(postgres: MigratedDatabase, course_id: str) -> tuple[dict[str, str], frozenset[str]]:
    """Read a course's documents (file to hash) and chunk ids as the database owner."""
    with postgres.owner_connection() as conn:
        documents = conn.execute("SELECT source_file, sha256 FROM documents WHERE course_id = %s", (course_id,))
        files = {file: sha256 for file, sha256 in documents.fetchall()}
        chunks = conn.execute("SELECT chunk_id FROM chunks WHERE course_id = %s", (course_id,))
        return files, frozenset(row[0] for row in chunks.fetchall())


@pytest.mark.covers("US-004", "AC-6")
async def test_document_dropped_from_the_manifest_is_deleted_with_its_rows(
    postgres: MigratedDatabase, store: PostgresCorpusStore, embedder: FixedLengthEmbedder, tmp_path: Path
) -> None:
    """Ingests a course with two sources, then ingests it again with a manifest that lists only one of them.

    Expects the dropped file reported as removed and only the remaining document with its rows and vectors left.
    Catches a delete that misses the dropped document or deletes the wrong one, and a dropped document whose
    sections, objectives, chunks or vectors stay behind, where search would still find them. Both files have the
    same structure, so the stored file names, not the counts alone, show which document is left.
    """
    _course_folder(tmp_path, "demo")
    await _ingest(store, embedder, tmp_path, _manifest("demo", DEMO_FILE, EXTRA_FILE))

    report = await _ingest(store, embedder, tmp_path, _manifest("demo", DEMO_FILE))

    assert report.removed == [EXTRA_FILE], f"the dropped file should be reported as removed, got {report.removed}"
    files = set(_stored(postgres, "demo")[0])
    assert files == {DEMO_FILE}, f"only {DEMO_FILE} should be stored, found {sorted(files)}"
    counts = await store.counts("demo")
    assert counts == DEMO_COUNTS, f"rows of the dropped document are left: {counts}"
    vectors = await store.vector_count("demo", embedder.space)
    assert vectors == DEMO_COUNTS.chunks, f"{vectors} vectors left for {DEMO_COUNTS.chunks} chunks"


@pytest.mark.covers("US-004", "AC-4")
async def test_source_file_saved_again_is_rebuilt_without_its_old_rows(
    store: PostgresCorpusStore, embedder: FixedLengthEmbedder, tmp_path: Path
) -> None:
    """Ingests the demo course, saves its PDF again (same text, new hash) and ingests once more.

    Expects the document reported as replaced, and its rows and vectors there once, under the new edition only.
    Catches a replaced document whose old chunks or vectors are not deleted, so every edition of a file would
    add its text to the course again.
    """
    folder = _course_folder(tmp_path, "demo")
    manifest = _manifest("demo", DEMO_FILE)
    await _ingest(store, embedder, tmp_path, manifest)
    _save_again(folder / DEMO_FILE)

    report = await _ingest(store, embedder, tmp_path, manifest)

    assert report.documents == [DocumentReport(source_file=DEMO_FILE, outcome="replaced")], (
        f"a file with new bytes should be rebuilt, got {report.documents}"
    )
    counts = await store.counts("demo")
    assert counts == DEMO_COUNTS, f"rows of the old edition are left: {counts}"
    vectors = await store.vector_count("demo", embedder.space)
    assert vectors == DEMO_COUNTS.chunks, f"{vectors} vectors left for {DEMO_COUNTS.chunks} chunks"


@pytest.mark.covers("US-004", "AC-7")
async def test_ingest_of_one_course_leaves_a_course_with_the_same_file_names_untouched(
    postgres: MigratedDatabase, store: PostgresCorpusStore, embedder: FixedLengthEmbedder, tmp_path: Path
) -> None:
    """Ingests two courses with the same two files, then changes one course so that its ingestion deletes in
    both ways: one file dropped from its manifest, the other saved again.

    Expects the other course's documents, chunks and vectors exactly as before. Catches a delete of documents
    that filters by file name only: it would delete the other course's documents of the same names too.
    """
    demo_folder = _course_folder(tmp_path, "demo")
    _course_folder(tmp_path, OTHER_COURSE)
    await _ingest(store, embedder, tmp_path, _manifest("demo", DEMO_FILE, EXTRA_FILE))
    await _ingest(store, embedder, tmp_path, _manifest(OTHER_COURSE, DEMO_FILE, EXTRA_FILE))
    other_before = _stored(postgres, OTHER_COURSE)
    other_vectors_before = await store.vector_count(OTHER_COURSE, embedder.space)
    _save_again(demo_folder / DEMO_FILE)

    await _ingest(store, embedder, tmp_path, _manifest("demo", DEMO_FILE))

    assert _stored(postgres, OTHER_COURSE) == other_before, "the other course's documents or chunks changed"
    other_vectors = await store.vector_count(OTHER_COURSE, embedder.space)
    assert other_vectors == other_vectors_before, (
        f"the other course had {other_vectors_before} vectors and has {other_vectors} now"
    )
