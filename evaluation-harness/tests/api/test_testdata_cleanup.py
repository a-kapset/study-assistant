"""The harness's own test data cleanup, checked against the app's real database."""

from evalharness.corpus_db import CorpusReader
from evalharness.settings import HarnessSettings, StackSettings
from evalharness.testdata import name
from evalharness.testdata.cleanup import remove_all
from evalharness.testdata.corpus import AGED_SHA256, CorpusTestData
from evalharness.transport import SutTransport

INGEST_DEMO = "/v1/admin/courses/demo/ingest"
DEMO_FILE = "vardolin_rules.pdf"


async def test_cleanup_removes_every_kind_of_test_data(
    sut: SutTransport,
    admin_headers: dict[str, str],
    settings: HarnessSettings,
    stack: StackSettings,
    corpus: CorpusReader,
    test_data: CorpusTestData,
) -> None:
    """``remove_all`` must delete every kind of data that ``CorpusTestData`` can create.

    The test creates each kind once: a copy of the demo course with all its rows, an extra document in the demo
    course, and a demo document with a fake hash. After ``remove_all`` it reads the rows one by one, not through the
    cleanup's own query, so a mistake shared by the cleanup and its leftover check is still caught. A kind of
    data the cleanup misses would stay in the database after a run that was stopped before its tests could fix it.
    """
    assert (await sut.request("POST", INGEST_DEMO, headers=admin_headers)).status_code == 200
    test_data.copy_course("demo", name("foreign"))
    test_data.add_document("demo", name("stale.pdf"))
    test_data.age_document("demo", DEMO_FILE)

    remove_all(settings, stack)

    assert corpus.documents(name("foreign")) == {}
    assert corpus.chunk_ids(name("foreign")) == []
    assert name("stale.pdf") not in corpus.documents("demo")
    assert AGED_SHA256 not in corpus.documents("demo").values()
    # The aged demo document was deleted with the test data; store it again for the tests that follow.
    assert (await sut.request("POST", INGEST_DEMO, headers=admin_headers)).status_code == 200
