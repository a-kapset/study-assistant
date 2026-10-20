"""Admin endpoints: operations that change what the app serves, open only to callers with the admin key."""

import secrets
from collections.abc import Mapping
from typing import Annotated

import structlog
from fastapi import APIRouter, Depends, Header, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import SecretStr

from study_assistant.courses.models import CourseManifest
from study_assistant.embed.remote import EmbeddingServiceRequestError, EmbeddingServiceUnavailableError
from study_assistant.errors import ErrorBody, ErrorEnvelope, error_response
from study_assistant.ingest.service import CourseIngestor, IngestConflictError, IngestError, IngestReport

ADMIN_KEY_HEADER = "X-Admin-Key"


def admin_router(courses: Mapping[str, CourseManifest], ingestor: CourseIngestor, api_key: SecretStr) -> APIRouter:
    """Build the router for the admin endpoints; every one of them needs the admin key."""
    expected = api_key.get_secret_value().encode("utf-8")
    log: structlog.typing.FilteringBoundLogger = structlog.get_logger(__name__)

    async def require_admin_key(key: Annotated[str | None, Header(alias=ADMIN_KEY_HEADER)] = None) -> None:
        """Refuse a request without the right key, before anything else about it is looked at."""
        # compare_digest takes as long for a near miss as for a wrong first byte, so timing gives nothing away.
        if key is None or not secrets.compare_digest(key.encode("utf-8"), expected):
            raise HTTPException(status_code=401, detail=f"A valid {ADMIN_KEY_HEADER} header is required")

    router = APIRouter(prefix="/v1/admin", dependencies=[Depends(require_admin_key)])

    @router.post(
        "/courses/{course_id}/ingest",
        response_model=IngestReport,
        responses={
            401: {"model": ErrorEnvelope},
            404: {"model": ErrorEnvelope},
            409: {"model": ErrorEnvelope},
            422: {"model": ErrorEnvelope},
            502: {"model": ErrorEnvelope},
            503: {"model": ErrorEnvelope},
        },
    )
    async def ingest_course(course_id: str, request: Request) -> IngestReport | JSONResponse:
        """Bring the stored corpus of a course in line with its source files; repeating it changes nothing.

        Every failure leaves the stored corpus as it was: a source that cannot be read is a 422, another ingestion
        of the course that changed it meanwhile a 409, an embedding server that answers with an error a 502, and one
        that cannot be reached a 503.
        """
        manifest = courses.get(course_id)
        if manifest is None:
            raise HTTPException(status_code=404, detail=f"No course '{course_id}' is served by this app")
        try:
            report = await ingestor.ingest(manifest)
        except IngestError as exc:
            return _failed(request, course_id, 422, "ingest_failed", str(exc))
        except IngestConflictError as exc:
            return _failed(request, course_id, 409, "ingest_conflict", str(exc))
        except EmbeddingServiceRequestError as exc:
            return _failed(request, course_id, 502, "embedding_failed", str(exc))
        except EmbeddingServiceUnavailableError as exc:
            message = f"the embedding server could not be reached ({exc})"
            return _failed(request, course_id, 503, "embedding_unavailable", message)
        log.info(
            "ingest_done",
            course_id=course_id,
            documents={document.source_file: document.outcome for document in report.documents},
            removed=report.removed,
            corpus=report.corpus.model_dump(),
            embedding_space=report.embedding_space,
            vectors=report.vectors,
        )
        return report

    def _failed(request: Request, course_id: str, status: int, code: str, message: str) -> JSONResponse:
        """Log a failed ingestion and answer it in the error envelope with ``status`` and ``code``."""
        log.warning("ingest_failed", course_id=course_id, code=code, error=message)
        return error_response(status, ErrorBody(code=code, message=message), request.state.request_id)

    return router
