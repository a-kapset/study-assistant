"""API tests for course ingestion.

Each test calls the admin endpoint for the demo course, then reads the database directly to see what was
really stored. The answer of the API alone is not enough: it can say one thing while the tables hold another.
Data the app cannot create through its API is made with the ``test_data`` fixture and removed by it.
"""

import pytest
from pydantic import JsonValue

from evalharness.corpus_db import CorpusReader
from evalharness.testdata import name
from evalharness.testdata.corpus import CorpusTestData
from evalharness.transport import SutTransport

INGEST_DEMO = "/v1/admin/courses/demo/ingest"
DEMO_FILE = "vardolin_rules.pdf"
# The demo course's structure as its text defines it (courses/demo/vardolin_rules.md): one source file,
# four units, eleven objectives. Chunk counts depend on the configured chunk size, so they are compared
# with the database instead.
DEMO_STRUCTURE = {"documents": 1, "sections": 4, "learning_objectives": 11}

# Test data names; they carry the harness prefix, so the cleanup finds them.
FOREIGN_COURSE = name("foreign")
STALE_FILE = name("stale.pdf")


async def _ingest_demo(sut: SutTransport, headers: dict[str, str]) -> dict[str, JsonValue]:
    """Send an ingest request for the demo course and return the JSON report.

    The test stops at once if the app does not answer with status 200; the message shows the response body.
    """
    observation = await sut.request("POST", INGEST_DEMO, headers=headers)
    assert observation.transport_error is None, f"system under test unreachable at {observation.url}"
    assert observation.status_code == 200, f"ingest failed: {observation.body_text}"
    assert isinstance(observation.body_json, dict)
    return observation.body_json


@pytest.mark.parametrize("headers", [{}, {"X-Admin-Key": "not-the-admin-key"}])
async def test_ingest_without_the_admin_key_is_refused(sut: SutTransport, headers: dict[str, str]) -> None:
    """Without the admin key, or with a wrong one, ingest must answer 401.

    Catches a deployed app whose admin routes run without the key check: anyone who reaches the port could then
    rewrite a course.
    """
    observation = await sut.request("POST", INGEST_DEMO, headers=headers)

    assert observation.status_code == 401
    assert isinstance(observation.body_json, dict)
    error = observation.body_json["error"]
    assert isinstance(error, dict)
    assert error["code"] == "unauthorized"


async def test_repeated_ingest_keeps_the_same_chunks(
    sut: SutTransport, admin_headers: dict[str, str], corpus: CorpusReader
) -> None:
    """Ingest the same files twice: the second ingest must change nothing.

    The report must say "unchanged", and the chunk ids in the database must be the same as after the first ingest.
    Catches chunks that are added a second time or get new ids on every run.
    """
    await _ingest_demo(sut, admin_headers)
    chunks_before = corpus.chunk_ids("demo")

    report = await _ingest_demo(sut, admin_headers)

    assert report["documents"] == [{"source_file": DEMO_FILE, "outcome": "unchanged"}]
    assert corpus.chunk_ids("demo") == chunks_before


async def test_reported_counts_are_the_stored_rows(
    sut: SutTransport, admin_headers: dict[str, str], corpus: CorpusReader
) -> None:
    """The counts in the ingest report and in GET /v1/courses/demo must equal the rows in the database.

    Catches counts that the app takes from memory instead of reading them from the database. The demo course's
    known structure (1 document, 4 sections, 11 objectives) also catches a counting mistake that the app and the
    harness could share.
    """
    report = await _ingest_demo(sut, admin_headers)
    detail = await sut.request("GET", "/v1/courses/demo")

    stored = corpus.counts("demo")
    assert isinstance(detail.body_json, dict)
    assert report["corpus"] == detail.body_json["corpus"] == stored
    assert {name: stored[name] for name in DEMO_STRUCTURE} == DEMO_STRUCTURE


async def test_document_dropped_from_the_manifest_is_deleted(
    sut: SutTransport, admin_headers: dict[str, str], corpus: CorpusReader, test_data: CorpusTestData
) -> None:
    """A document that is no longer in the manifest must be deleted from the database.

    The test adds such a document to the demo course, then ingests. The report must list it as removed, and the
    database must keep only the real demo document. Otherwise search would still find text that is no longer
    part of the course.
    """
    await _ingest_demo(sut, admin_headers)
    test_data.add_document("demo", STALE_FILE)

    report = await _ingest_demo(sut, admin_headers)

    assert report["removed"] == [STALE_FILE]
    assert set(corpus.documents("demo")) == {DEMO_FILE}


async def test_changed_document_is_replaced(
    sut: SutTransport, admin_headers: dict[str, str], corpus: CorpusReader, test_data: CorpusTestData
) -> None:
    """A document stored from another version of its file must be rebuilt from the current file.

    The test fakes an older version by changing the stored file hash, then ingests. The report must say
    "replaced", the database must hold the real hash again, and the chunks must be the same as before: no old
    chunks left and none missing.
    """
    await _ingest_demo(sut, admin_headers)
    current_sha256 = corpus.documents("demo")[DEMO_FILE]
    chunks_before = corpus.chunk_ids("demo")
    test_data.age_document("demo", DEMO_FILE)

    report = await _ingest_demo(sut, admin_headers)

    assert report["documents"] == [{"source_file": DEMO_FILE, "outcome": "replaced"}]
    assert corpus.documents("demo")[DEMO_FILE] == current_sha256
    assert corpus.chunk_ids("demo") == chunks_before


async def test_ingest_leaves_other_courses_untouched(
    sut: SutTransport, admin_headers: dict[str, str], corpus: CorpusReader, test_data: CorpusTestData
) -> None:
    """Ingest of the demo course must not change another course, even one with documents of the same file names.

    The other course is a full copy of the demo course (documents, sections, objectives, chunks) plus a document
    named like the demo's removed one. The demo ingest here deletes in both ways: a document removed from the
    manifest and a replaced one. A delete of documents or chunks that filters only by file name, not by course,
    would also delete rows of the other course.
    """
    await _ingest_demo(sut, admin_headers)
    test_data.copy_course("demo", FOREIGN_COURSE)
    test_data.add_document(FOREIGN_COURSE, STALE_FILE)
    test_data.add_document("demo", STALE_FILE)
    test_data.age_document("demo", DEMO_FILE)
    foreign_documents = corpus.documents(FOREIGN_COURSE)
    foreign_chunks = corpus.chunk_ids(FOREIGN_COURSE)

    await _ingest_demo(sut, admin_headers)

    assert corpus.documents(FOREIGN_COURSE) == foreign_documents
    assert corpus.chunk_ids(FOREIGN_COURSE) == foreign_chunks
