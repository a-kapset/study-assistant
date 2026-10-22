from collections.abc import Mapping, Sequence

import httpx2
import pytest
import structlog
from structlog.testing import capture_logs

from study_assistant.api.courses import CoursesResponse, CourseSummary
from study_assistant.errors import ErrorBody, ErrorEnvelope
from study_assistant.ingest.service import CorpusCounts, DocumentReport, IngestReport
from study_assistant.ingest_courses import ingest_all

# Nothing listens there: every request goes to the fake app server below.
BASE_URL = "http://app.invalid"
COURSES = ("demo", "demo_second")


def _listing(course_ids: Sequence[str]) -> dict[str, object]:
    """Build the body of GET /v1/courses with the app's own response model, so the fake answers what the app does."""
    summaries = [
        CourseSummary(id=course_id, title=course_id, description="A course for tests.", language="en")
        for course_id in course_ids
    ]
    return CoursesResponse(courses=summaries).model_dump(mode="json")


def _report(course_id: str) -> dict[str, object]:
    """Build the body of a successful ingestion with the app's own report model."""
    return IngestReport(
        course_id=course_id,
        documents=[DocumentReport(source_file="rules.pdf", outcome="unchanged")],
        removed=[],
        corpus=CorpusCounts(documents=1, sections=3, learning_objectives=2, chunks=12),
        embedding_space="test_space",
        vectors=12,
    ).model_dump(mode="json")


def _failure(status: int, code: str, message: str) -> httpx2.Response:
    """Build a failed answer in the app's error envelope."""
    envelope = ErrorEnvelope(error=ErrorBody(code=code, message=message), request_id="test-request")
    return httpx2.Response(status, json=envelope.model_dump(mode="json", exclude_none=True))


class FakeAppServer:
    """Stands in for the running app server: lists its courses and answers the ingestion of each one.

    A course in ``failures`` gets that answer; a course in ``silent`` gets no answer at all (the request times
    out, as when an ingestion takes longer than the run waits); every other course is ingested. Each request
    is recorded as method and path.
    """

    def __init__(
        self,
        course_ids: Sequence[str] = COURSES,
        failures: Mapping[str, httpx2.Response] | None = None,
        silent: frozenset[str] = frozenset(),
    ) -> None:
        """Serve ``course_ids`` with the given failed and silent courses."""
        self.course_ids = course_ids
        self.failures = failures or {}
        self.silent = silent
        self.requests: list[tuple[str, str]] = []

    def handle(self, request: httpx2.Request) -> httpx2.Response:
        """Answer one request the way the app server would, and record it."""
        self.requests.append((request.method, request.url.path))
        if request.method == "GET" and request.url.path == "/v1/courses":
            return httpx2.Response(200, json=_listing(self.course_ids))
        # The ingestion path is /v1/admin/courses/{course_id}/ingest.
        course_id = request.url.path.split("/")[-2]
        if course_id in self.silent:
            raise httpx2.ReadTimeout("timed out", request=request)
        return self.failures.get(course_id) or httpx2.Response(200, json=_report(course_id))

    def ingested(self) -> list[str]:
        """Return the ids of the courses whose ingestion was requested, in order."""
        return [path.split("/")[-2] for method, path in self.requests if method == "POST"]

    def client(self) -> httpx2.Client:
        """Return a client whose requests reach this fake instead of the network."""
        return httpx2.Client(base_url=BASE_URL, transport=httpx2.MockTransport(self.handle))


@pytest.mark.covers("US-004", "AC-23")
def test_every_course_the_app_serves_is_ingested() -> None:
    """The app serves two courses and both ingest fine: the run asks for each one's ingestion at the admin path
    and ends with 0. Catches a run that skips a course, calls a wrong path, or fails although nothing failed."""
    app = FakeAppServer()

    with app.client() as client:
        code = ingest_all(client, structlog.get_logger())

    assert code == 0, f"the run ended with {code} although every course was ingested"
    expected = [("POST", f"/v1/admin/courses/{course_id}/ingest") for course_id in COURSES]
    assert app.requests[1:] == expected, f"the ingestion requests were {app.requests[1:]}, not one per course"


@pytest.mark.covers("US-004", "AC-23")
def test_a_failed_course_fails_the_run() -> None:
    """The first course's ingestion answers 422 and the second one succeeds: the run must end with 1.
    Catches a run that logs the failure but still ends with 0, so Compose and CI would report a full corpus."""
    app = FakeAppServer(failures={"demo": _failure(422, "ingest_failed", "rules.pdf cannot be read")})

    with app.client() as client:
        code = ingest_all(client, structlog.get_logger())

    assert code == 1, f"the run ended with {code} although the course 'demo' failed"


@pytest.mark.covers("US-004", "AC-23")
def test_the_courses_after_a_failed_one_are_still_ingested() -> None:
    """The first course's ingestion answers 422: the second course must still be ingested.
    Catches a run that stops at the first failure, which would leave later courses empty and name only one problem."""
    app = FakeAppServer(failures={"demo": _failure(422, "ingest_failed", "rules.pdf cannot be read")})

    with app.client() as client:
        ingest_all(client, structlog.get_logger())

    assert app.ingested() == list(COURSES), f"after the failed course only {app.ingested()} were ingested"


@pytest.mark.covers("US-004", "AC-23")
def test_a_course_without_an_answer_counts_as_failed_and_the_rest_still_run() -> None:
    """The first course's ingestion gets no answer in time (a timeout): the run must go on to the second course and
    end with 1. Catches a timeout that escapes as an exception, which would stop the run before the other courses."""
    app = FakeAppServer(silent=frozenset({"demo"}))

    with app.client() as client:
        code = ingest_all(client, structlog.get_logger())

    assert app.ingested() == list(COURSES), f"after the timeout only {app.ingested()} were ingested"
    assert code == 1, f"the run ended with {code} although the course 'demo' got no answer"


@pytest.mark.covers("US-004", "AC-23")
def test_the_log_names_the_failed_course_and_the_app_error_code() -> None:
    """The first course's ingestion answers 422 with the code ingest_failed: the log must have a failure line for
    that course with the app's code and message. Catches a failure line without the course or the reason, which
    would leave the operator to search the app server's log."""
    app = FakeAppServer(failures={"demo": _failure(422, "ingest_failed", "rules.pdf cannot be read")})

    with capture_logs() as logs, app.client() as client:
        ingest_all(client, structlog.get_logger())

    failures = [entry for entry in logs if entry["event"] == "course_ingest_failed"]
    expected = {"course_id": "demo", "status": 422, "error": "ingest_failed: rules.pdf cannot be read"}
    assert [{key: entry.get(key) for key in expected} for entry in failures] == [expected], (
        f"the failure lines were {failures}, not one line naming 'demo' and the app's error"
    )
