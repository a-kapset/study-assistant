"""The course manifest: metadata and rules for one course, read from courses/<course_id>/course.yaml.

Everything course-specific reaches the app through these models, so the app code holds no course names.
"""

import re
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

# A course id becomes a directory name and part of stored records, so it is limited to safe characters.
COURSE_ID_PATTERN = r"^[a-z0-9_]+$"

# Two-letter language code of the course material, e.g. "en".
COURSE_LANGUAGE_PATTERN = r"^[a-z]{2}$"


# What a source file is for: course text, term definitions, or exam questions in the common format.
type SourceRole = Literal["material", "glossary", "question_bank"]

# Primary sources define what is correct for the course; supplementary ones add background but never override them.
type SourceAuthority = Literal["primary", "supplementary"]


class ManifestModel(BaseModel):
    """Base for manifest parts: an unknown or misspelt key is an error, not silently ignored."""

    model_config = ConfigDict(extra="forbid", frozen=True)


class StructureRules(ManifestModel):
    """How to find the course structure in its source text; each value is a regular expression.

    Invalid expressions are rejected when the manifest is loaded, not when the sources are first read.
    """

    section_heading: re.Pattern[str]
    objective_id: re.Pattern[str]
    # Not every course grades its objectives by knowledge level.
    knowledge_level: re.Pattern[str] | None = None


class CourseSource(ManifestModel):
    """One source file of a course; the file itself stays outside the repository."""

    # Path relative to the course's local source directory.
    file: str = Field(min_length=1)
    role: SourceRole
    authority: SourceAuthority


class CourseManifest(ManifestModel):
    """One course: what it is called, how its sources are structured, and which files they are."""

    id: str = Field(pattern=COURSE_ID_PATTERN)
    title: str = Field(min_length=1)
    # A short description of what the course covers.
    description: str = Field(min_length=1)
    language: str = Field(pattern=COURSE_LANGUAGE_PATTERN)
    structure: StructureRules
    sources: tuple[CourseSource, ...] = Field(min_length=1)
