import dataclasses
import re
import shutil
from pathlib import Path
from typing import TYPE_CHECKING

import pytest

from study_assistant.courses.models import CourseManifest, CourseSource
from study_assistant.courses.registry import load_course
from study_assistant.embed.remote import EmbeddingServiceUnavailableError
from study_assistant.ingest.chunker import ChunkSize
from study_assistant.ingest.service import (
    CorpusCounts,
    CourseIngestor,
    DocumentReport,
    Edition,
    IngestConflictError,
    IngestError,
    outcome,
    prepare_course,
    prepare_document,
    rules_hash,
)

if TYPE_CHECKING:
    from conftest import FakeCorpusStore, FakeEmbedder

# The async tests run on the anyio plugin, which the web framework already brings.
pytestmark = pytest.mark.anyio

# The repository's own course folder: app/tests/unit/<this file> -> repository root.
REPO_COURSES = Path(__file__).resolve().parents[3] / "courses"
SIZE = ChunkSize(size=1000, overlap=100)
DEMO_FILE = "vardolin_rules.pdf"
# The demo course as its manifest describes it: 4 units, 11 objectives, 12 chunks at 1000/100.
DEMO_COUNTS = CorpusCounts(documents=1, sections=4, learning_objectives=11, chunks=12)
DEMO_CHUNKS = 12


def _demo() -> CourseManifest:
    """Load the demo course's manifest."""
    return load_course(REPO_COURSES, "demo")


def _with_sources(manifest: CourseManifest, *sources: CourseSource) -> CourseManifest:
    """Return the manifest with its source list replaced."""
    return manifest.model_copy(update={"sources": sources})


def _ingestor(store: FakeCorpusStore, embedder: FakeEmbedder, source_dir: Path = REPO_COURSES) -> CourseIngestor:
    """An ingestor over the fake store and embedder, reading sources from ``source_dir``/<course_id>."""
    return CourseIngestor(store, embedder, source_dir, SIZE)


@pytest.mark.covers("US-004", "AC-2")
async def test_first_ingestion_adds_the_document_and_reports_the_stored_counts(
    fake_store: FakeCorpusStore, fake_embedder: FakeEmbedder
) -> None:
    """Guards the add path and the report: a new document must be written whole and counted from the store."""
    report = await _ingestor(fake_store, fake_embedder).ingest(_demo())

    assert report.documents == [DocumentReport(source_file=DEMO_FILE, outcome="added")]
    assert report.removed == []
    assert report.corpus == DEMO_COUNTS
    assert fake_store.courses["demo"] == ("Demo: The Rules of Vardolin", "en")


@pytest.mark.covers("US-004", "AC-3")
async def test_second_ingestion_of_the_same_files_writes_nothing(
    fake_store: FakeCorpusStore, fake_embedder: FakeEmbedder
) -> None:
    """Guards idempotency: an unchanged document must be kept, not deleted and written again, and gets no vectors."""
    ingestor = _ingestor(fake_store, fake_embedder)
    await ingestor.ingest(_demo())
    stored = fake_store.documents["demo"][DEMO_FILE]
    fake_store.events.clear()

    report = await ingestor.ingest(_demo())

    assert report.documents == [DocumentReport(source_file=DEMO_FILE, outcome="unchanged")]
    assert fake_store.events == [("delete", "demo", ()), ("vectors", "demo", ())]
    assert fake_store.documents["demo"][DEMO_FILE] is stored
    assert report.corpus == DEMO_COUNTS


@pytest.mark.covers("US-004", "AC-4")
async def test_changed_document_is_deleted_before_it_is_written_again(
    fake_store: FakeCorpusStore, fake_embedder: FakeEmbedder
) -> None:
    """Guards the replace path: writing without deleting first would keep the old edition's rows or fail on keys."""
    ingestor = _ingestor(fake_store, fake_embedder)
    await ingestor.ingest(_demo())
    _make_stored_edition_older(fake_store)
    fake_store.events.clear()

    report = await ingestor.ingest(_demo())

    assert report.documents == [DocumentReport(source_file=DEMO_FILE, outcome="replaced")]
    assert fake_store.events[:2] == [("delete", "demo", (DEMO_FILE,)), ("insert", "demo", DEMO_FILE)]
    assert report.corpus == DEMO_COUNTS


@pytest.mark.covers("US-004", "AC-6")
async def test_document_no_longer_in_the_manifest_is_deleted(
    fake_store: FakeCorpusStore, fake_embedder: FakeEmbedder
) -> None:
    """Guards removal: a source dropped from the manifest must not stay searchable."""
    ingestor = _ingestor(fake_store, fake_embedder)
    await ingestor.ingest(_demo())
    fake_store.documents["demo"]["dropped.pdf"] = fake_store.documents["demo"][DEMO_FILE]
    fake_store.events.clear()

    report = await ingestor.ingest(_demo())

    assert report.removed == ["dropped.pdf"]
    assert fake_store.events[0] == ("delete", "demo", ("dropped.pdf",))
    assert report.corpus == DEMO_COUNTS


@pytest.mark.covers("US-004", "AC-2", "AC-17")
async def test_reported_counts_are_read_back_from_the_store(
    fake_store: FakeCorpusStore, fake_embedder: FakeEmbedder, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Sets the store's counts to values ingestion cannot produce and expects exactly those in the report.

    Catches counts taken from what was prepared or computed: they would claim rows or vectors the store may not hold.
    """
    stored = CorpusCounts(documents=7, sections=7, learning_objectives=7, chunks=7)
    monkeypatch.setattr(fake_store, "count", lambda course_id: stored)
    monkeypatch.setattr(fake_store, "count_vectors", lambda course_id: 7)

    report = await _ingestor(fake_store, fake_embedder).ingest(_demo())

    assert report.corpus == stored
    assert report.vectors == 7


@pytest.mark.covers("US-004", "AC-8")
async def test_a_missing_later_source_stops_the_run_before_anything_is_written(
    fake_store: FakeCorpusStore, fake_embedder: FakeEmbedder, tmp_path: Path
) -> None:
    """Guards reading every source first: a failure after the first file must not leave a half-updated course."""
    (tmp_path / "demo").mkdir()
    shutil.copy(REPO_COURSES / "demo" / DEMO_FILE, tmp_path / "demo" / DEMO_FILE)
    demo = _demo()
    missing = demo.sources[0].model_copy(update={"file": "missing.pdf"})

    with pytest.raises(IngestError, match=r"missing\.pdf: cannot read the file"):
        await _ingestor(fake_store, fake_embedder, tmp_path).ingest(_with_sources(demo, demo.sources[0], missing))
    assert fake_store.writers_opened == 0


def _make_stored_edition_older(store: FakeCorpusStore) -> None:
    """Mark the stored demo document as made from an older file, so the next ingestion replaces it."""
    stored = store.documents["demo"][DEMO_FILE]
    store.documents["demo"][DEMO_FILE] = dataclasses.replace(
        stored, edition=dataclasses.replace(stored.edition, sha256="an-older-edition")
    )


@pytest.mark.covers("US-004", "AC-17")
async def test_every_chunk_of_an_added_document_gets_the_vector_of_its_text(
    fake_store: FakeCorpusStore, fake_embedder: FakeEmbedder
) -> None:
    """Ingests the demo course once and expects each stored chunk to hold the vector computed from its own text,
    reported under the embedder's space.

    Catches vectors joined to the wrong chunks (an order mix-up stores a neighbour's vector with a success status),
    chunks left without a vector, and a report that names another space.
    """
    report = await _ingestor(fake_store, fake_embedder).ingest(_demo())

    chunks = fake_store.documents["demo"][DEMO_FILE].chunks
    assert fake_store.vectors["demo"] == {chunk.id: fake_embedder.vector_of(chunk.text) for chunk in chunks}
    assert len(chunks) == report.vectors == DEMO_CHUNKS
    assert report.embedding_space == fake_embedder.space.id


@pytest.mark.covers("US-004", "AC-19")
async def test_unchanged_document_with_its_vectors_gets_nothing_computed(
    fake_store: FakeCorpusStore, fake_embedder: FakeEmbedder
) -> None:
    """Ingests twice and expects the second run to send nothing to the embedding server.

    Catches vectors computed again for every unchanged document: each ingestion of the first course would then
    wait for the whole course to be embedded again.
    """
    ingestor = _ingestor(fake_store, fake_embedder)
    await ingestor.ingest(_demo())
    fake_embedder.calls.clear()

    await ingestor.ingest(_demo())

    assert fake_embedder.calls == []


@pytest.mark.covers("US-004", "AC-19")
async def test_unchanged_document_without_vectors_gets_only_its_vectors(
    fake_store: FakeCorpusStore, fake_embedder: FakeEmbedder
) -> None:
    """Ingests once, removes the stored vectors (as after a change of model), ingests again and expects the outcome
    "embedded": every chunk gets its vector and the document and its chunks stay as they were.

    Catches a document left without vectors because its rows are unchanged, and a document rebuilt only to get
    its vectors.
    """
    ingestor = _ingestor(fake_store, fake_embedder)
    await ingestor.ingest(_demo())
    stored = fake_store.documents["demo"][DEMO_FILE]
    fake_store.vectors["demo"].clear()
    fake_store.events.clear()

    report = await ingestor.ingest(_demo())

    assert report.documents == [DocumentReport(source_file=DEMO_FILE, outcome="embedded")]
    assert fake_store.documents["demo"][DEMO_FILE] is stored
    assert fake_store.events == [("delete", "demo", ()), ("vectors", "demo", tuple(c.id for c in stored.chunks))]


@pytest.mark.covers("US-004", "AC-20")
async def test_replaced_document_gets_a_vector_for_every_new_chunk(
    fake_store: FakeCorpusStore, fake_embedder: FakeEmbedder
) -> None:
    """Ingests once, marks the stored document as older and ingests again; expects every chunk of the new edition
    to be sent to the embedding server and stored with its vector.

    Catches a replaced document that keeps only the vectors its old chunks had, or none at all.
    """
    ingestor = _ingestor(fake_store, fake_embedder)
    await ingestor.ingest(_demo())
    _make_stored_edition_older(fake_store)
    fake_embedder.calls.clear()

    report = await ingestor.ingest(_demo())

    chunks = fake_store.documents["demo"][DEMO_FILE].chunks
    assert fake_embedder.calls == [[chunk.text for chunk in chunks]]
    assert report.vectors == DEMO_CHUNKS


@pytest.mark.covers("US-004", "AC-17")
async def test_unreachable_embedding_server_stops_the_run_before_anything_is_written(
    fake_store: FakeCorpusStore, fake_embedder: FakeEmbedder
) -> None:
    """Makes the embedder fail as an unreachable server does and expects that error, with no write opened.

    Catches chunks stored without vectors when the server is down: search would then miss that text with no error.
    """
    fake_embedder.error = EmbeddingServiceUnavailableError("ConnectError")

    with pytest.raises(EmbeddingServiceUnavailableError):
        await _ingestor(fake_store, fake_embedder).ingest(_demo())
    assert fake_store.writers_opened == 0


@pytest.mark.covers("US-004", "AC-17")
async def test_fewer_vectors_than_chunks_stops_the_run_before_anything_is_written(
    fake_store: FakeCorpusStore, fake_embedder: FakeEmbedder
) -> None:
    """Makes the embedder return one vector less than it was given texts and expects a ValueError, with no write.

    Catches a lenient join of chunks and vectors: the last chunk would silently get no vector.
    """
    fake_embedder.drop_last = True

    with pytest.raises(ValueError, match="zip"):
        await _ingestor(fake_store, fake_embedder).ingest(_demo())
    assert fake_store.writers_opened == 0


async def test_chunks_needing_vectors_that_were_not_computed_are_a_conflict(
    fake_store: FakeCorpusStore, fake_embedder: FakeEmbedder
) -> None:
    """Ingests once; then, between the read and the write of the second run, another ingestion removes the vectors.
    Expects ``IngestConflictError`` and nothing written by this run.

    Catches a plan trusted from before the write: the run would store chunks without vectors, or fail on a missing
    vector halfway through its writes.
    """
    ingestor = _ingestor(fake_store, fake_embedder)
    await ingestor.ingest(_demo())
    fake_store.events.clear()
    fake_store.before_write = fake_store.vectors["demo"].clear

    with pytest.raises(IngestConflictError, match=f"{DEMO_CHUNKS} chunks have no vector"):
        await ingestor.ingest(_demo())
    assert fake_store.events == []


async def test_a_change_meanwhile_that_needs_no_new_vectors_is_not_a_conflict(
    fake_store: FakeCorpusStore, fake_embedder: FakeEmbedder
) -> None:
    """Starts with an empty store; between the read and the write, another ingestion stores the course with its
    vectors. Expects the run to succeed, write nothing new and report the document as unchanged.

    Catches a conflict raised whenever the stored state changed: two ingestions started together would fail for
    nothing, though the second has no work left.
    """
    finished = type(fake_store)()
    await _ingestor(finished, fake_embedder).ingest(_demo())

    def other_ingestion_finished() -> None:
        """Put into the store what another ingestion of the same files stored, just before this run writes."""
        fake_store.documents.update(finished.documents)
        fake_store.vectors.update(finished.vectors)

    fake_store.before_write = other_ingestion_finished

    report = await _ingestor(fake_store, fake_embedder).ingest(_demo())

    assert report.documents == [DocumentReport(source_file=DEMO_FILE, outcome="unchanged")]
    assert fake_store.events == [("delete", "demo", ()), ("vectors", "demo", ())]


@pytest.mark.covers("US-004", "AC-8")
def test_source_that_is_not_a_pdf_is_refused(tmp_path: Path) -> None:
    """Guards the format check: another format would reach the PDF parser and fail with an obscure error."""
    (tmp_path / "notes.md").write_text("Unit 1. Notes", encoding="utf-8")
    demo = _demo()
    source = demo.sources[0].model_copy(update={"file": "notes.md"})

    with pytest.raises(IngestError, match=r"notes\.md: only PDF"):
        prepare_document(demo, source, tmp_path, SIZE)


@pytest.mark.covers("US-004", "AC-14")
def test_question_bank_sources_are_not_chunked() -> None:
    """Guards the role filter: a question bank is not course text, and its answers must not become chunks."""
    demo = _demo()
    bank = CourseSource(file="bank.json", role="question_bank", authority="primary")

    prepared = prepare_course(_with_sources(demo, demo.sources[0], bank), REPO_COURSES / "demo", SIZE)

    assert [document.source.file for document in prepared] == [DEMO_FILE]


@pytest.mark.covers("US-004", "AC-3")
def test_outcome_of_a_new_and_of_an_identical_document() -> None:
    """Guards the two simple cases: a new document is added, an identical one is kept."""
    edition = Edition("a", "b", "c", "d", "e", "f")

    assert outcome(None, edition) == "added"
    assert outcome(dataclasses.replace(edition), edition) == "unchanged"


@pytest.mark.covers("US-004", "AC-4")
@pytest.mark.parametrize("field", [field.name for field in dataclasses.fields(Edition)])
def test_any_change_in_what_a_document_was_made_from_replaces_it(field: str) -> None:
    """Guards the comparison: checking only the file hash would keep rows made by an older parser or rules."""
    current = Edition("a", "b", "c", "d", "e", "f")

    assert outcome(dataclasses.replace(current, **{field: "changed"}), current) == "replaced"


def _changed_rules(manifest: CourseManifest, change: str) -> CourseManifest:
    """Return the manifest with one value that shapes the demo source's rows changed."""
    source = manifest.sources[0]
    structure = manifest.structure
    match change:
        case "language":
            return manifest.model_copy(update={"language": "de"})
        case "role":
            return _with_sources(manifest, source.model_copy(update={"role": "glossary"}))
        case "authority":
            return _with_sources(manifest, source.model_copy(update={"authority": "supplementary"}))
        case "noise_lines":
            lines = (*source.noise_lines, re.compile("Draft"))
            return _with_sources(manifest, source.model_copy(update={"noise_lines": lines}))
        case "section_heading":
            rules = structure.model_copy(update={"section_heading": re.compile(r"^Part (?P<id>\d+)\. (?P<title>.+)$")})
        case "objective_id":
            rules = structure.model_copy(update={"objective_id": re.compile(r"OBJ-\d{2}")})
        case _:
            rules = structure.model_copy(update={"knowledge_level": re.compile(r"\((?P<level>K\d)\)")})
    return manifest.model_copy(update={"structure": rules})


@pytest.mark.covers("US-004", "AC-4")
@pytest.mark.parametrize(
    "change", ["language", "role", "authority", "noise_lines", "section_heading", "objective_id", "knowledge_level"]
)
def test_every_rule_that_shapes_a_document_changes_its_rules_hash(change: str) -> None:
    """Guards the hash's coverage: a manifest value left out would let its edit keep rows made under the old value."""
    demo = _demo()
    changed = _changed_rules(demo, change)

    assert rules_hash(changed, changed.sources[0]) != rules_hash(demo, demo.sources[0])


@pytest.mark.covers("US-004", "AC-5")
def test_title_and_description_do_not_rebuild_documents() -> None:
    """Guards the hash's scope: wording shown to learners does not change any stored row, so it must not re-ingest."""
    demo = _demo()
    renamed = demo.model_copy(update={"title": "Another title", "description": "Another description."})

    assert rules_hash(renamed, renamed.sources[0]) == rules_hash(demo, demo.sources[0])
