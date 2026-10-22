"""One-shot ingestion run: asks the running app server to ingest every course it serves, then exits.

Compose starts it once the app server is ready, so every start of the stack ends with the stored corpus in line
with the course files, also on an empty database. A course whose files are unchanged costs nothing.

It only calls the app server's admin endpoint, so the ingestion itself, its lock, its embedding calls and its
logs stay in one place. Every course is tried even after one has failed, so one run names every broken course;
the exit code is 1 if any failed, which Compose and CI report.
"""

import sys

import httpx2
import structlog

from study_assistant.api.admin import ADMIN_KEY_HEADER
from study_assistant.logging_setup import configure_logging
from study_assistant.settings import load_ingest_job_settings

COURSES_PATH = "/v1/courses"


def ingest_all(client: httpx2.Client, log: structlog.typing.FilteringBoundLogger) -> int:
    """Ingest every course the app server lists, through ``client``; return 0 if all succeeded, otherwise 1."""
    try:
        listing = client.get(COURSES_PATH)
    except httpx2.TransportError as exc:
        log.error("ingest_run_failed", error=f"the app server could not be reached ({exc!r})")
        return 1
    if not listing.is_success:
        log.error("ingest_run_failed", error=f"listing the courses answered {listing.status_code}")
        return 1
    course_ids = [course["id"] for course in listing.json()["courses"]]
    failed = [course_id for course_id in course_ids if not ingest_course(client, course_id, log)]
    if failed:
        log.error("ingest_run_failed", courses=course_ids, failed=failed)
        return 1
    log.info("ingest_run_done", courses=course_ids)
    return 0


def ingest_course(client: httpx2.Client, course_id: str, log: structlog.typing.FilteringBoundLogger) -> bool:
    """Ask the app server to ingest one course and log what it answered; return whether it succeeded."""
    try:
        response = client.post(f"/v1/admin/courses/{course_id}/ingest")
    except httpx2.TransportError as exc:
        # A timeout lands here too: the app server may still finish the ingestion, but this run cannot know it.
        log.error("course_ingest_failed", course_id=course_id, error=f"no answer from the app server ({exc!r})")
        return False
    if not response.is_success:
        log.error("course_ingest_failed", course_id=course_id, status=response.status_code, error=_error_of(response))
        return False
    report = response.json()
    log.info(
        "course_ingested",
        course_id=course_id,
        documents={document["source_file"]: document["outcome"] for document in report["documents"]},
        removed=report["removed"],
        vectors=report["vectors"],
    )
    return True


def _error_of(response: httpx2.Response) -> str:
    """Return the error code and message of a failed answer, or the start of its body if it is not the app's error."""
    try:
        error = response.json()["error"]
        return f"{error['code']}: {error['message']}"
    except ValueError, KeyError, TypeError:
        # Something other than the app answered, e.g. a proxy page; its first characters say enough.
        return response.text[:200]


def main() -> int:
    """Read the settings, ingest every course through the app server and return the exit code."""
    settings = load_ingest_job_settings()
    configure_logging(settings.log_level)
    log = structlog.get_logger().bind(git_commit=settings.git_commit)
    headers = {ADMIN_KEY_HEADER: settings.admin.api_key.get_secret_value()}
    with httpx2.Client(
        base_url=str(settings.server.url), headers=headers, timeout=settings.server.timeout_seconds
    ) as client:
        return ingest_all(client, log)


if __name__ == "__main__":
    sys.exit(main())
