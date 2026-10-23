"""Search: the passages of a course that are closest to a question."""

from collections.abc import Mapping
from typing import Annotated

import structlog
from fastapi import APIRouter, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict, Field, StringConstraints

from study_assistant.courses.models import COURSE_ID_PATTERN, CourseManifest
from study_assistant.embed.remote import EmbeddingServiceRequestError, EmbeddingServiceUnavailableError
from study_assistant.errors import ErrorBody, ErrorEnvelope, error_response
from study_assistant.retrieval.models import SearchResult
from study_assistant.retrieval.service import CourseNotSearchableError, VectorSearch
from study_assistant.settings import SearchSettings

# Longer than any question of the first course (about 900 characters); a longer text is not a question.
MAX_QUERY_LENGTH = 2000


class SearchRequest(BaseModel):
    """A question about one course; the number of hits and the lowest score fall back to the settings."""

    model_config = ConfigDict(extra="forbid")
    course_id: str = Field(pattern=COURSE_ID_PATTERN)
    # Spaces around the question are dropped; a question of spaces only is refused.
    query: Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=MAX_QUERY_LENGTH)]
    k: int | None = Field(default=None, ge=1)
    min_score: float | None = Field(default=None, ge=-1, le=1)


def search_router(courses: Mapping[str, CourseManifest], search: VectorSearch, settings: SearchSettings) -> APIRouter:
    """Build the router that serves ``POST /v1/search``."""
    log: structlog.typing.FilteringBoundLogger = structlog.get_logger(__name__)
    router = APIRouter()

    @router.post(
        "/v1/search",
        response_model=SearchResult,
        responses={
            404: {"model": ErrorEnvelope},
            409: {"model": ErrorEnvelope},
            422: {"model": ErrorEnvelope},
            502: {"model": ErrorEnvelope},
            503: {"model": ErrorEnvelope},
        },
    )
    async def search_course(body: SearchRequest, request: Request) -> SearchResult | JSONResponse:
        """Return the passages of the course closest to the question, or "empty" when none is close enough.

        A course this app does not serve is a 404, a course that is not fully embedded in the configured space a
        409, an embedding server that answers with an error a 502, and one that cannot be reached a 503.
        """
        if body.course_id not in courses:
            raise HTTPException(status_code=404, detail=f"No course '{body.course_id}' is served by this app")
        if body.k is not None and body.k > settings.max_k:
            # The limit comes from the settings, so it is checked here, but reported like any invalid field.
            raise RequestValidationError(
                [{"loc": ("body", "k"), "msg": f"Input should be less than or equal to {settings.max_k}"}]
            )
        k = body.k if body.k is not None else settings.default_k
        min_score = body.min_score if body.min_score is not None else settings.min_score
        try:
            result = await search.search(body.course_id, body.query, k, min_score)
        except CourseNotSearchableError as exc:
            return _failed(request, body.course_id, 409, "course_not_searchable", str(exc))
        except EmbeddingServiceRequestError as exc:
            return _failed(request, body.course_id, 502, "embedding_failed", str(exc))
        except EmbeddingServiceUnavailableError as exc:
            message = f"the embedding server could not be reached ({exc})"
            return _failed(request, body.course_id, 503, "embedding_unavailable", message)
        # The question itself is not logged: it is the user's text.
        log.info("search_done", course_id=body.course_id, status=result.status, hits=len(result.hits), k=k)
        return result

    def _failed(request: Request, course_id: str, status: int, code: str, message: str) -> JSONResponse:
        """Log a failed search and answer it in the error envelope with ``status`` and ``code``."""
        log.warning("search_failed", course_id=course_id, code=code, error=message)
        return error_response(status, ErrorBody(code=code, message=message), request.state.request_id)

    return router
