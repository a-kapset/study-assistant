"""Admin endpoints: operations that change what the app serves, open only to callers with the admin key."""

import asyncio
import secrets
from collections.abc import Mapping
from pathlib import Path
from typing import Annotated
from uuid import UUID

import structlog
from fastapi import APIRouter, Depends, Header, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict, SecretStr

from study_assistant.courses.models import CourseManifest
from study_assistant.embed.remote import EmbeddingServiceRequestError, EmbeddingServiceUnavailableError
from study_assistant.errors import ErrorBody, ErrorEnvelope, error_response
from study_assistant.exams.loader import QuestionBankError, read_course_bank
from study_assistant.exams.service import ExamService
from study_assistant.ingest.service import CourseIngestor, IngestConflictError, IngestError, IngestReport
from study_assistant.learners.service import Learners

ADMIN_KEY_HEADER = "X-Admin-Key"


class CourseIngestReport(IngestReport):
    """The result of ingesting one course: its corpus, and how many questions its bank now offers to sessions."""

    question_bank_items: int


class LearnerRegistered(BaseModel):
    """A new learner. The key is shown here once and stored nowhere in readable form: a lost key is not recoverable."""

    model_config = ConfigDict(extra="forbid", frozen=True)
    learner_id: UUID
    api_key: str


def admin_router(
    courses: Mapping[str, CourseManifest],
    ingestor: CourseIngestor,
    exams: ExamService,
    learners: Learners,
    source_dir: Path,
    api_key: SecretStr,
) -> APIRouter:
    """Build the router for the admin endpoints; every one of them needs the admin key.

    Question banks are read from the course folders under ``source_dir``, like the other course sources.
    """
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
        response_model=CourseIngestReport,
        responses={
            401: {"model": ErrorEnvelope},
            404: {"model": ErrorEnvelope},
            409: {"model": ErrorEnvelope},
            422: {"model": ErrorEnvelope},
            502: {"model": ErrorEnvelope},
            503: {"model": ErrorEnvelope},
        },
    )
    async def ingest_course(course_id: str, request: Request) -> CourseIngestReport | JSONResponse:
        """Bring the stored corpus and question bank of a course in line with its source files; repeating it
        changes nothing.

        Every failure leaves the stored data as it was: a source or a bank file that cannot be read is a 422,
        another ingestion of the course that changed it meanwhile a 409, an embedding server that answers with an
        error a 502, and one that cannot be reached a 503. The bank is read and checked before anything is written.
        """
        manifest = courses.get(course_id)
        if manifest is None:
            raise HTTPException(status_code=404, detail=f"No course '{course_id}' is served by this app")
        try:
            bank = await asyncio.to_thread(read_course_bank, manifest, source_dir / manifest.id)
        except QuestionBankError as exc:
            return _failed(request, course_id, 422, "question_bank_invalid", str(exc))
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
        # After the corpus: the course's row, which the bank's rows point at, exists from the first ingestion on.
        question_bank_items = await exams.load_bank(course_id, bank)
        log.info(
            "ingest_done",
            course_id=course_id,
            documents={document.source_file: document.outcome for document in report.documents},
            removed=report.removed,
            corpus=report.corpus.model_dump(),
            embedding_space=report.embedding_space,
            vectors=report.vectors,
            question_bank_items=question_bank_items,
        )
        return CourseIngestReport(**report.model_dump(), question_bank_items=question_bank_items)

    @router.post("/learners", status_code=201, responses={401: {"model": ErrorEnvelope}})
    async def register_learner() -> LearnerRegistered:
        """Register a learner and return its id with the key its requests must carry; the key is shown only here."""
        learner_id, key = await learners.register()
        # The key itself is never logged.
        log.info("learner_registered", learner_id=str(learner_id))
        return LearnerRegistered(learner_id=learner_id, api_key=key)

    @router.delete(
        "/learners/{learner_id}",
        status_code=204,
        responses={401: {"model": ErrorEnvelope}, 404: {"model": ErrorEnvelope}, 422: {"model": ErrorEnvelope}},
    )
    async def remove_learner(learner_id: UUID) -> None:
        """Delete a learner with every session and answer it has; its key opens nothing afterwards.

        A learner that does not exist is a 404, so removing twice is told apart from removing once.
        """
        if not await learners.remove(learner_id):
            raise HTTPException(status_code=404, detail=f"No learner '{learner_id}'")
        log.info("learner_removed", learner_id=str(learner_id))

    def _failed(request: Request, course_id: str, status: int, code: str, message: str) -> JSONResponse:
        """Log a failed ingestion and answer it in the error envelope with ``status`` and ``code``."""
        log.warning("ingest_failed", course_id=course_id, code=code, error=message)
        return error_response(status, ErrorBody(code=code, message=message), request.state.request_id)

    return router
