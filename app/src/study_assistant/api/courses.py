"""Courses the app serves: the ones enabled in the settings, as described by their manifests."""

from collections.abc import Mapping

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, ConfigDict

from study_assistant.courses.models import CourseManifest
from study_assistant.errors import ErrorEnvelope
from study_assistant.ingest.service import CorpusCounts, CorpusStore


class CourseSummary(BaseModel):
    """What a client needs to pick a course; structure rules and source files stay inside the app."""

    model_config = ConfigDict(extra="forbid")
    id: str
    title: str
    description: str
    language: str


class CourseDetail(CourseSummary):
    """One course and how much of it is stored; all counts are zero until the course is ingested."""

    corpus: CorpusCounts


class CoursesResponse(BaseModel):
    """The enabled courses, in the order the settings list them."""

    model_config = ConfigDict(extra="forbid")
    courses: list[CourseSummary]


def courses_router(courses: Mapping[str, CourseManifest], store: CorpusStore) -> APIRouter:
    """Build the router that serves ``GET /v1/courses`` and ``GET /v1/courses/{course_id}``."""
    # Courses do not change while the app runs, so their summaries are built once.
    summaries = {
        course.id: CourseSummary(
            id=course.id, title=course.title, description=course.description, language=course.language
        )
        for course in courses.values()
    }
    response = CoursesResponse(courses=list(summaries.values()))
    router = APIRouter()

    @router.get("/v1/courses")
    async def list_courses() -> CoursesResponse:
        """Return the courses this app serves, so a client knows which course ids it can use."""
        return response

    @router.get("/v1/courses/{course_id}", responses={404: {"model": ErrorEnvelope}})
    async def get_course(course_id: str) -> CourseDetail:
        """Return one course with the counts read from the database, so a client sees whether it is ingested."""
        summary = summaries.get(course_id)
        if summary is None:
            raise HTTPException(status_code=404, detail=f"No course '{course_id}' is served by this app")
        return CourseDetail(**summary.model_dump(), corpus=await store.counts(course_id))

    return router
