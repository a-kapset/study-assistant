"""Ask: a learner's question about a course, answered by the answer model."""

from collections.abc import Mapping
from typing import Annotated

import structlog
from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict, Field, StringConstraints

from study_assistant.answer.service import Answerer
from study_assistant.api.search import MAX_QUERY_LENGTH
from study_assistant.courses.models import COURSE_ID_PATTERN, CourseManifest
from study_assistant.errors import ErrorBody, ErrorEnvelope, error_response
from study_assistant.llm.providers import LlmError, LlmUnavailableError


class AskRequest(BaseModel):
    """A learner's question about one course."""

    model_config = ConfigDict(extra="forbid")
    course_id: str = Field(pattern=COURSE_ID_PATTERN)
    # Same limits as a search question: spaces around it are dropped, and a question of spaces only is refused.
    question: Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=MAX_QUERY_LENGTH)]


class AskAnswer(BaseModel):
    """The answer the learner reads."""

    model_config = ConfigDict(extra="forbid")
    answer: str


def ask_router(courses: Mapping[str, CourseManifest], answerer: Answerer) -> APIRouter:
    """Build the router that serves ``POST /v1/ask``."""
    log: structlog.typing.FilteringBoundLogger = structlog.get_logger(__name__)
    router = APIRouter()

    @router.post(
        "/v1/ask",
        response_model=AskAnswer,
        responses={
            404: {"model": ErrorEnvelope},
            422: {"model": ErrorEnvelope},
            502: {"model": ErrorEnvelope},
            503: {"model": ErrorEnvelope},
        },
    )
    async def ask_course(body: AskRequest, request: Request) -> AskAnswer | JSONResponse:
        """Return the model's answer to the question about the course.

        A course this app does not serve is a 404 and the model is not called. A model provider that cannot be
        reached is a 503; a provider that answers with an error, or a reply without text, is a 502.
        """
        course = courses.get(body.course_id)
        if course is None:
            raise HTTPException(status_code=404, detail=f"No course '{body.course_id}' is served by this app")
        try:
            answer = await answerer.answer(course, body.question)
        except LlmUnavailableError as exc:
            message = f"the model provider could not be reached ({exc})"
            return _failed(request, body.course_id, 503, "llm_unavailable", message)
        except LlmError as exc:
            return _failed(request, body.course_id, 502, "llm_failed", str(exc))
        # Neither the question nor the answer is logged: both are the learner's conversation.
        log.info("ask_done", course_id=body.course_id, answer_chars=len(answer))
        return AskAnswer(answer=answer)

    def _failed(request: Request, course_id: str, status: int, code: str, message: str) -> JSONResponse:
        """Log a failed ask and answer it in the error envelope with ``status`` and ``code``."""
        log.warning("ask_failed", course_id=course_id, code=code, error=message)
        return error_response(status, ErrorBody(code=code, message=message), request.state.request_id)

    return router
