"""Practice exams and progress: every endpoint here acts for the learner whose key the request carries."""

from collections.abc import Awaitable, Callable, Mapping
from typing import Annotated
from uuid import UUID

import structlog
from fastapi import APIRouter, Depends, HTTPException, Query, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict, Field, StringConstraints

from study_assistant.courses.models import COURSE_ID_PATTERN, CourseManifest, ExamSettings
from study_assistant.errors import ErrorBody, ErrorEnvelope, error_response
from study_assistant.exams.service import (
    AlreadyAnsweredError,
    AnswerResult,
    ExamService,
    ItemNotInSessionError,
    NextQuestion,
    Progress,
    QuestionBankEmptyError,
    SessionCreated,
    SessionNotFoundError,
    SessionResult,
    UnknownObjectiveError,
    UnknownOptionError,
)

# Ids come from course files; these limits only stop a request from carrying text that is no id.
_MAX_ID_LENGTH = 200
_MAX_SELECTED = 50

type Identifier = Annotated[str, StringConstraints(min_length=1, max_length=_MAX_ID_LENGTH)]


class SessionRequest(BaseModel):
    """What a new session covers: a course, optionally one learning objective and a number of questions."""

    model_config = ConfigDict(extra="forbid")
    course_id: str = Field(pattern=COURSE_ID_PATTERN)
    # Without it the session covers the whole bank of the course.
    objective_id: Identifier | None = None
    # Without it the course's session size is used; a session never gets more questions than there are in scope.
    size: int | None = Field(default=None, ge=1)


class AnswerRequest(BaseModel):
    """An answer to one question of a session: the ids of the selected options."""

    model_config = ConfigDict(extra="forbid")
    item_id: Identifier
    selected: list[Annotated[str, StringConstraints(max_length=_MAX_ID_LENGTH)]] = Field(
        min_length=1, max_length=_MAX_SELECTED
    )


_ERRORS: dict[int | str, dict[str, object]] = {status: {"model": ErrorEnvelope} for status in (401, 404, 409, 422)}


def exams_router(
    courses: Mapping[str, CourseManifest], exams: ExamService, current_learner: Callable[..., Awaitable[UUID]]
) -> APIRouter:
    """Build the router for practice sessions and progress; every endpoint needs a learner's key."""
    log: structlog.typing.FilteringBoundLogger = structlog.get_logger(__name__)
    router = APIRouter(responses=_ERRORS)

    @router.post("/v1/exams/sessions", status_code=201, response_model=SessionCreated)
    async def create_session(
        body: SessionRequest, learner_id: Annotated[UUID, Depends(current_learner)], request: Request
    ) -> SessionCreated | JSONResponse:
        """Start a practice session over a course's question bank, or over one of its learning objectives.

        A course this app does not serve is a 404, a course without a loaded bank a 409, and an objective no
        question tests a 422.
        """
        manifest = courses.get(body.course_id)
        if manifest is None:
            raise HTTPException(status_code=404, detail=f"No course '{body.course_id}' is served by this app")
        try:
            created = await exams.create_session(
                learner_id, body.course_id, manifest.exam, body.objective_id, body.size
            )
        except QuestionBankEmptyError as exc:
            return _failed(request, 409, "question_bank_empty", str(exc))
        except UnknownObjectiveError as exc:
            return _failed(request, 422, "unknown_objective", str(exc))
        log.info(
            "exam_session_created",
            course_id=body.course_id,
            session_id=str(created.session_id),
            questions=created.question_count,
        )
        return created

    @router.get("/v1/exams/sessions/{session_id}/next", response_model=NextQuestion, response_model_exclude_none=True)
    async def next_question(
        session_id: UUID, learner_id: Annotated[UUID, Depends(current_learner)], request: Request
    ) -> NextQuestion | JSONResponse:
        """Return the session's first question without an answer, never with its key, or "finished"."""
        try:
            return await exams.next_question(learner_id, session_id)
        except SessionNotFoundError as exc:
            return _failed(request, 404, "session_not_found", str(exc))

    @router.post("/v1/exams/sessions/{session_id}/answers", response_model=AnswerResult)
    async def answer(
        session_id: UUID, body: AnswerRequest, learner_id: Annotated[UUID, Depends(current_learner)], request: Request
    ) -> AnswerResult | JSONResponse:
        """Check and store the answer to one question of the session; reply whether it was correct, with the key.

        Sending the same answer again gives the same reply and stores nothing. A different answer to a question
        that already has one is a 409, an option the question does not have a 422, and a question the session
        does not hold a 404.
        """
        try:
            result = await exams.answer(learner_id, session_id, body.item_id, body.selected)
        except SessionNotFoundError as exc:
            return _failed(request, 404, "session_not_found", str(exc))
        except ItemNotInSessionError as exc:
            return _failed(request, 404, "item_not_in_session", str(exc))
        except UnknownOptionError as exc:
            return _failed(request, 422, "unknown_option", str(exc))
        except AlreadyAnsweredError as exc:
            return _failed(request, 409, "already_answered", str(exc))
        return result

    @router.get("/v1/exams/sessions/{session_id}/result", response_model=SessionResult)
    async def result(
        session_id: UUID, learner_id: Annotated[UUID, Depends(current_learner)], request: Request
    ) -> SessionResult | JSONResponse:
        """Return how many questions of the session are answered and correct, and, once all are answered,
        whether the session is passed by the course's pass mark."""
        try:
            return await exams.result(learner_id, session_id, _exam_settings)
        except SessionNotFoundError as exc:
            return _failed(request, 404, "session_not_found", str(exc))

    @router.get("/v1/progress")
    async def progress(
        course_id: Annotated[str, Query(pattern=COURSE_ID_PATTERN)],
        learner_id: Annotated[UUID, Depends(current_learner)],
    ) -> Progress:
        """Return the learner's answers in a course per learning objective, over all sessions."""
        if course_id not in courses:
            raise HTTPException(status_code=404, detail=f"No course '{course_id}' is served by this app")
        return await exams.progress(learner_id, course_id)

    def _exam_settings(course_id: str) -> ExamSettings | None:
        """Return the exam settings of a course as its manifest has them now; none for a course no longer served."""
        manifest = courses.get(course_id)
        return manifest.exam if manifest is not None else None

    def _failed(request: Request, status: int, code: str, message: str) -> JSONResponse:
        """Log a refused exam request and answer it in the error envelope with ``status`` and ``code``."""
        log.info("exam_request_refused", code=code, error=message)
        return error_response(status, ErrorBody(code=code, message=message), request.state.request_id)

    return router
