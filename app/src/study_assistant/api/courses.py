"""Courses the app serves: the ones enabled in the settings, as described by their manifests."""

from collections.abc import Mapping

from fastapi import APIRouter
from pydantic import BaseModel, ConfigDict

from study_assistant.courses.models import CourseManifest


class CourseSummary(BaseModel):
    """What a client needs to pick a course; structure rules and source files stay inside the app."""

    model_config = ConfigDict(extra="forbid")
    id: str
    title: str
    description: str
    language: str


class CoursesResponse(BaseModel):
    """The enabled courses, in the order the settings list them."""

    model_config = ConfigDict(extra="forbid")
    courses: list[CourseSummary]


def courses_router(courses: Mapping[str, CourseManifest]) -> APIRouter:
    """Build the router that serves ``GET /v1/courses`` for the courses loaded at startup."""
    # Courses do not change while the app runs, so the response is built once.
    response = CoursesResponse(
        courses=[
            CourseSummary(id=course.id, title=course.title, description=course.description, language=course.language)
            for course in courses.values()
        ]
    )
    router = APIRouter()

    @router.get("/v1/courses")
    async def list_courses() -> CoursesResponse:
        """Return the courses this app serves, so a client knows which course ids it can use."""
        return response

    return router
